"""DINO용 MoE 모듈: DINOBatchedExperts + DINOMoEBlock

V-MoE의 구조를 그대로 따르되, DINO FFN에 맞게 수정:
- GELU → ReLU
- bias 포함
- 패딩 토큰 마스킹 (-inf) + aux loss에서 패딩 토큰 제외
- JAX 원본과 동일한 group_size 방식: group_size = moe_group_images * tokens_per_image
  → reshape(-1, group_size, D)로 그룹 구성
  → moe_group_images=1이면 이미지 1장 = 1그룹 (기존 동작과 동일)
- ExpertIndicesDispatcher 사용 (메모리 효율)

Expert Parallelism (EP) 지원:
- expert_parallel=True 시 각 GPU가 num_experts // world_size 개의 local expert만 보유
- all-to-all로 토큰을 교환하여 각 GPU가 local expert만 실행
- Router의 gate는 전체 E experts에 대해 계산 (global routing)
- DINOBatchedExperts는 local_num_experts 개만 생성
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from models.moe import moe
from models.moe import expert_parallel as ep
from models.moe.routing import NoisyTopExpertsPerItemRouter, _weighted_sum


class DINOBatchedExperts(nn.Module):
    """DINO FFN 구조와 동일한 Batched Expert (ReLU 기반).

    원본 DINO FFN: Linear(256→2048) → ReLU → Dropout → Linear(2048→256) → Dropout
    V-MoE Expert:  Linear(D→mlp_dim) → GELU → Linear(mlp_dim→D)

    차이점: GELU → ReLU, dropout 추가, bias 포함
    Expert는 순수 FFN만 담당 — residual/LayerNorm은 외부(Encoder Layer)에서 처리

    EP 시 num_experts = local_num_experts (이 GPU가 보유하는 expert 수)
    """

    def __init__(self, num_experts: int, d_model: int = 256,
                 d_ffn: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout

        # Batched weights: (num_experts, out_dim, in_dim)
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ffn, d_model))
        self.b1 = nn.Parameter(torch.zeros(num_experts, d_ffn))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ffn))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))

        self._init_weights()

    def _init_weights(self):
        """Kaiming init (ReLU 기반이므로) — nn.Linear 기본 init과 동일."""
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            # bias init (nn.Linear 기본과 동일)
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.w1[i])
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.b1[i], -bound1, bound1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.w2[i])
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.b2[i], -bound2, bound2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (E_local, tokens_per_expert, d_model) — dispatcher가 보내준 텐서
               EP 시 E_local = num_experts // world_size

        Returns:
            (E_local, tokens_per_expert, d_model)
        """
        # Linear1: (E, T, d_model) @ (E, d_model, d_ffn) → (E, T, d_ffn)
        x = torch.bmm(x, self.w1.transpose(1, 2)) + self.b1.unsqueeze(1)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Linear2: (E, T, d_ffn) @ (E, d_ffn, d_model) → (E, T, d_model)
        x = torch.bmm(x, self.w2.transpose(1, 2)) + self.b2.unsqueeze(1)
        # 마지막 dropout은 MoE block 밖 residual 부분에서 처리
        return x


class DINOMoEBlock(nn.Module):
    """V-MoE의 MlpMoeBlock을 DINO에 맞게 수정.

    JAX 원본 V-MoE와 동일한 group_size 방식 적용:
    - group_size = moe_group_images * N_total (이미지당 토큰 수)
    - (B, N_total, D) → reshape(-1, group_size, D) → (G, S, D)
    - num_groups = B / moe_group_images
    - moe_group_images=1이면 이미지 1장 = 1그룹 (B가 곧 G)

    Expert Parallelism:
    - expert_parallel=True 시 local expert만 생성 (num_experts // ep_world_size)
    - Router gate는 전체 E experts에 대해 계산 (global)
    - Dispatcher가 all-to-all로 토큰을 local experts에 분배

    forward 반환값: (output, metrics)
    - output: (B, N_total, D) — 순수 FFN 출력 (residual/norm 미포함)
    - metrics: dict — auxiliary_loss 등
    """

    def __init__(self,
                 d_model: int = 256,
                 d_ffn: int = 2048,
                 num_experts: int = 8,
                 num_selected_experts: int = 2,
                 capacity_factor: float = 1.25,
                 noise_std: float = 1.0,
                 dropout: float = 0.0,
                 gshard_loss_weight: float = 0.01,
                 importance_loss_weight: float = 1.0,
                 load_loss_weight: float = 1.0,
                 moe_group_images: int = 1,
                 expert_parallel: bool = False,
                 split_rngs: bool = False,
                 moe_mode: str = 'baseline',
                 num_classes: int = 0,
                 class_routing_loss_weight: float = 0.0,
                 class_routing_alpha: float = 0.1):
        super().__init__()
        self.num_experts = num_experts  # global total
        self.num_selected_experts = num_selected_experts
        self.capacity_factor = capacity_factor
        self.moe_group_images = moe_group_images
        self.expert_parallel = expert_parallel
        self.moe_mode = moe_mode
        self.num_classes = num_classes
        self.class_routing_loss_weight = class_routing_loss_weight
        self.class_routing_alpha = class_routing_alpha

        # EP: local expert 수 계산
        if expert_parallel and ep.get_ep_world_size() > 1:
            W = ep.get_ep_world_size()
            assert num_experts % W == 0, (
                f"num_experts({num_experts})는 ep_world_size({W})로 "
                f"나누어 떨어져야 합니다.")
            self.local_num_experts = num_experts // W
        else:
            self.local_num_experts = num_experts

        # Router — gate는 전체 E experts에 대해 계산 (EP 여부 무관)
        # expert_parallel flag를 dispatcher에 전달
        self.router = NoisyTopExpertsPerItemRouter(
            num_experts=num_experts,
            input_dim=d_model,
            num_selected_experts=num_selected_experts,
            noise_std=noise_std,
            gshard_loss_weight=gshard_loss_weight,
            importance_loss_weight=importance_loss_weight,
            load_loss_weight=load_loss_weight,
            expert_parallel=expert_parallel,
            dispatcher={
                'name': 'indices',               # ExpertIndicesDispatcher
                'batch_priority': True,
                'capacity_factor': capacity_factor,
            },
        )

        # Batched Experts — EP 시 local expert만 생성
        self.experts = DINOBatchedExperts(
            num_experts=self.local_num_experts,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
        )

        # 시각화/모니터링 플래그: True일 때 metrics에 route_map과 load 통계 추가
        # 기본 False (학습 중 오버헤드 없음)
        # 시각화 시: model.transformer._moe_metrics[i]['route_map'] 으로 접근
        self.capture_routing = False

        # JAX V-MoE 원본 방식: split_rngs=False이면 expert 0의 weight를
        # 나머지 expert에 복사하여 모든 expert를 동일한 초기값으로 시작.
        # routing noise가 학습 과정에서 expert 분화를 유도함.
        # split_rngs=True이면 각 expert가 독립적으로 random init.
        if not split_rngs:
            with torch.no_grad():
                for name in ['w1', 'b1', 'w2', 'b2']:
                    param = getattr(self.experts, name)
                    param[1:] = param[0:1].expand_as(param[1:]).clone()

    def set_class_routing_loss_weight(self, weight: float):
        self.class_routing_loss_weight = float(weight)

    def _compute_soft_labels(
            self,
            gt_info: dict,
            spatial_shapes: torch.Tensor,
            B: int,
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """각 토큰에 대한 soft class label과 foreground 마스크를 계산.

        token 좌표는 padded 이미지 기준 정규화, GT box는 원본 이미지 기준.
        valid_ratios[b,l,0]=W_valid/W_pad, [b,l,1]=H_valid/H_pad 로 per-image per-level 보정.

        Returns:
            soft_labels : (B, N_total_orig, C)
            fg_mask     : (B, N_total_orig)  True = foreground
        """
        C = self.num_classes
        gt_boxes_list  = gt_info['boxes']              # list[Tensor(n_i, 4)] cxcywh normalized
        gt_labels_list = gt_info['labels']             # list[Tensor(n_i,)]
        valid_ratios   = gt_info.get('valid_ratios')   # (B, L, 2) or None

        # level별 token 중심 좌표 생성 (padded 이미지 기준)
        token_x_list, token_y_list = [], []
        for l in range(spatial_shapes.shape[0]):
            H_l = int(spatial_shapes[l, 0])
            W_l = int(spatial_shapes[l, 1])
            ys = (torch.arange(H_l, device=device).float() + 0.5) / H_l
            xs = (torch.arange(W_l, device=device).float() + 0.5) / W_l
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            token_x_list.append(xx.reshape(-1))  # (H_l*W_l,)
            token_y_list.append(yy.reshape(-1))

        N_per_level = [tx.shape[0] for tx in token_x_list]
        N = sum(N_per_level)
        soft_labels = torch.zeros(B, N, C, device=device)
        fg_mask     = torch.zeros(B, N, dtype=torch.bool, device=device)

        with torch.no_grad():
            for b in range(B):
                boxes  = gt_boxes_list[b]   # (n_b, 4)
                labels = gt_labels_list[b]  # (n_b,)
                if len(boxes) == 0:
                    continue

                # cxcywh → xyxy (원본 이미지 기준 정규화)
                x1_orig = boxes[:, 0] - boxes[:, 2] / 2  # (n_b,)
                y1_orig = boxes[:, 1] - boxes[:, 3] / 2
                x2_orig = boxes[:, 0] + boxes[:, 2] / 2
                y2_orig = boxes[:, 1] + boxes[:, 3] / 2

                in_box_parts = []
                for l, (tx, ty) in enumerate(zip(token_x_list, token_y_list)):
                    # valid_ratios로 GT 좌표를 padded 이미지 기준으로 변환
                    # token 좌표가 padded 기준이므로 GT도 동일 기준으로 맞춰야 함
                    if valid_ratios is not None:
                        vw = valid_ratios[b, l, 0]  # W_valid/W_pad
                        vh = valid_ratios[b, l, 1]  # H_valid/H_pad
                        x1 = x1_orig * vw
                        y1 = y1_orig * vh
                        x2 = x2_orig * vw
                        y2 = y2_orig * vh
                    else:
                        x1, y1, x2, y2 = x1_orig, y1_orig, x2_orig, y2_orig

                    # in_box_l[i, j]: level l의 token i가 GT box j 안에 있으면 True
                    in_box_l = (
                        (tx.unsqueeze(1) > x1.unsqueeze(0)) &
                        (tx.unsqueeze(1) < x2.unsqueeze(0)) &
                        (ty.unsqueeze(1) > y1.unsqueeze(0)) &
                        (ty.unsqueeze(1) < y2.unsqueeze(0))
                    )  # (N_l, n_b)
                    in_box_parts.append(in_box_l)

                in_box = torch.cat(in_box_parts, dim=0)  # (N, n_b)

                n_boxes_per_token = in_box.sum(dim=1).float()  # (N,)
                fg_mask[b] = n_boxes_per_token > 0

                # n빵: token_class_counts[i,c] = 토큰 i에 해당하는 class c 박스 수
                labels_oh = F.one_hot(labels.clamp(0, C - 1), C).float()  # (n_b, C)
                token_class_counts = in_box.float() @ labels_oh            # (N, C)
                soft_labels[b] = (
                    token_class_counts /
                    n_boxes_per_token.clamp(min=1.0).unsqueeze(1)
                )

        return soft_labels, fg_mask

    def _compute_class_routing_loss(
            self,
            gates_gs:    torch.Tensor,
            soft_labels: torch.Tensor,
            fg_mask_gs:  torch.Tensor,
            pad_mask_gs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Class-aware routing loss (No EMA, batch prototype).

        Args:
            gates_gs    : (G, S, E) gate softmax
            soft_labels : (G, S, C) soft class label
            fg_mask_gs  : (G, S)   True = foreground
            pad_mask_gs : (G, S)   True = EP padding (제외 대상)

        Returns: scalar — intra + alpha * inter
        """
        valid = fg_mask_gs
        if pad_mask_gs is not None:
            valid = valid & (~pad_mask_gs)

        M = int(valid.sum().item())
        if M == 0:
            return gates_gs.sum() * 0.0

        r = gates_gs[valid]      # (M, E)
        q = soft_labels[valid]   # (M, C)

        # 배치 raw prototype: P_hat_c = (q^T @ r) / (q_sum + ε)
        q_sum      = q.sum(dim=0)                                # (C,)
        prototypes = q.T @ r                                     # (C, E)
        prototypes = prototypes / (q_sum.unsqueeze(1) + 1e-8)

        r_norm = F.normalize(r,          dim=-1)  # (M, E)
        p_norm = F.normalize(prototypes, dim=-1)  # (C, E)

        # Intra loss: 같은 class 토큰 → 같은 routing pattern
        cos_sim = r_norm @ p_norm.T                              # (M, C)
        intra   = (q * (1.0 - cos_sim)).sum() / M

        # Inter loss: 다른 class prototype → 다른 routing pattern
        active   = q_sum > 0
        n_active = int(active.sum().item())
        if n_active < 2:
            inter = gates_gs.sum() * 0.0
        else:
            p_active = p_norm[active]                            # (n_active, E)
            sim_mat  = p_active @ p_active.T                    # (n_active, n_active)
            off_diag = ~torch.eye(n_active, dtype=torch.bool,
                                  device=gates_gs.device)
            inter = sim_mat[off_diag].sum() / (n_active * (n_active - 1))

        return intra + self.class_routing_alpha * inter

    def forward(self, src: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                gt_info: Optional[dict] = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            src: (B, N_total, D) — 인코더 토큰
            key_padding_mask: (B, N_total) — True = 패딩 위치
            gt_info: {'boxes': list[Tensor(n,4)], 'labels': list[Tensor(n,)]}
                     학습 시에만 전달, inference 시 None

        Returns:
            output: (B, N_total, D) — 순수 FFN 출력 (residual/norm 미포함)
            metrics: dict — auxiliary_loss, gshard_loss, importance_loss, load_loss
        """
        B, N_total, D = src.shape
        orig_N_total = N_total
        per_scale_pad_info = None  # per-scale 패딩 strip 시 사용

        # ── EP 환경에서 GPU 간 Sequence Length 동기화 ──
        # DINO는 배치 내 최대 이미지 크기로 패딩하므로 GPU마다 N_total이 다를 수 있음.
        # all-to-all은 모든 rank에서 텐서 크기가 동일해야 하므로 패딩 필요.
        #
        # [Per-Scale 패딩] spatial_shapes가 있으면 scale별로 독립 패딩:
        #   변경 전: [scale1 | scale2 | scale3 | scale4 | PAD PAD PAD]  (맨 뒤 flat)
        #   변경 후: [scale1+PAD | scale2+PAD | scale3+PAD | scale4+PAD] (scale별)
        # → 아이디어 2(scale별 라우팅)로의 확장이 자연스러움.
        # → 패딩 토큰은 key_padding_mask=True로 마스킹되어 라우팅/loss에 영향 없음.
        if self.expert_parallel and ep.get_ep_world_size() > 1 and spatial_shapes is not None:
            num_levels = spatial_shapes.shape[0]
            #spatial_shapes.shape=(4,2)=> (scale갯수,(H,W))
            local_tokens_per_level = spatial_shapes[:, 0] * spatial_shapes[:, 1]  # (L,)
        

            # 각 scale별 max 토큰 수를 GPU간 동기화
            max_tokens_per_level = local_tokens_per_level.clone() #원본저장
            dist.all_reduce(max_tokens_per_level, op=dist.ReduceOp.MAX,
                            group=ep.get_ep_group())
            #scale간 가장 많은 토큰 갯수로 통일 =>4scale 모두                

            pad_per_level = max_tokens_per_level - local_tokens_per_level  # (L,)
            #pad_per_level은 gpu통신간에 같은 크기를 위해 각 scale마다 필요한 padding 토큰 갯수 저장

            if pad_per_level.sum().item() > 0:
                # scale별로 split → pad → concat
                src_splits = src.split(local_tokens_per_level.tolist(), dim=1)
                #src shape=(B,전체 토큰갯수 N_total,D) => scale별로 다시 나누기
                mask_splits = (key_padding_mask.split(local_tokens_per_level.tolist(), dim=1)
                               if key_padding_mask is not None else [None] * num_levels)

                new_src, new_mask = [], []
                for lvl in range(num_levels):
                    p = pad_per_level[lvl].item() # p= 얼마나 많은 패딩을 각 scale마다 더 추가를 할지
                    if p > 0:
                        new_src.append(F.pad(src_splits[lvl], (0, 0, 0, p))) #F.pad=>텐서의 테두리에 패딩해줌
                        #pad는 뒤에서부터 적용  현재 src_splits는 (B,scale 토큰갯수,D) => 0,0은 D차원에 패딩X
                        #그다음 0,p는 앞에는 0개 뒤에 p개만큼 0 붙히기
                        if mask_splits[lvl] is not None:
                            new_mask.append(F.pad(mask_splits[lvl], (0, p), value=True)) #true=padding O
                        else: #한디바이스 안에서 배치 안에서는 이미지 사이즈가 같아서  none인데 다른 디바이스가 더 큰경우
                            m = torch.zeros(B, local_tokens_per_level[lvl].item() + p,
                                            dtype=torch.bool, device=src.device)
                            m[:, local_tokens_per_level[lvl].item():] = True
                            new_mask.append(m)
                    else:
                        new_src.append(src_splits[lvl])
                        new_mask.append(mask_splits[lvl] if mask_splits[lvl] is not None
                                        else torch.zeros(B, local_tokens_per_level[lvl].item(),
                                                         dtype=torch.bool, device=src.device))

                src = torch.cat(new_src, dim=1)
                key_padding_mask = torch.cat(new_mask, dim=1) #gpu간 통신을 위한 패딩된 이미지,mask를 대입
                N_total = src.shape[1]
                per_scale_pad_info = (local_tokens_per_level, max_tokens_per_level)

        elif self.expert_parallel and ep.get_ep_world_size() > 1:
            # fallback: spatial_shapes 없으면 기존 flat-end 방식 유지
            local_N = torch.tensor([N_total], dtype=torch.long, device=src.device)
            dist.all_reduce(local_N, op=dist.ReduceOp.MAX, group=ep.get_ep_group())
            max_N = local_N.item()

            if max_N > N_total:
                pad_len = max_N - N_total
                src = F.pad(src, (0, 0, 0, pad_len))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, pad_len), value=True)
                else:
                    key_padding_mask = torch.zeros(B, max_N, dtype=torch.bool, device=src.device)
                    key_padding_mask[:, N_total:] = True
                N_total = max_N

        # ── soft_labels 계산 (baseline/scale_aware 공용, 한 번만) ──
        soft_labels_flat = fg_mask_flat = None
        if (gt_info is not None and self.training and
                self.num_classes > 0 and self.class_routing_loss_weight > 0 and
                spatial_shapes is not None):
            soft_labels_flat, fg_mask_flat = self._compute_soft_labels(
                gt_info, spatial_shapes, B, src.device)
            # soft_labels_flat: (B, N_total_orig, C)
            orig_N = soft_labels_flat.shape[1]
            if orig_N < N_total:   # EP 패딩이 적용된 경우
                if per_scale_pad_info is not None:
                    # per-scale 패딩: level별로 맞춰서 pad
                    orig_per, padded_per = per_scale_pad_info
                    sl_sp = soft_labels_flat.split(orig_per.tolist(), dim=1)
                    fm_sp = fg_mask_flat.split(orig_per.tolist(), dim=1)
                    sl_list, fm_list = [], []
                    for lvl_idx in range(len(orig_per)):
                        p = int(padded_per[lvl_idx]) - int(orig_per[lvl_idx])
                        sl_list.append(F.pad(sl_sp[lvl_idx], (0, 0, 0, p)) if p > 0 else sl_sp[lvl_idx])
                        fm_list.append(F.pad(fm_sp[lvl_idx], (0, p))       if p > 0 else fm_sp[lvl_idx])
                    soft_labels_flat = torch.cat(sl_list, dim=1)
                    fg_mask_flat     = torch.cat(fm_list, dim=1)
                else:
                    # flat-end 패딩
                    pad_len = N_total - orig_N
                    soft_labels_flat = F.pad(soft_labels_flat, (0, 0, 0, pad_len))
                    fg_mask_flat     = F.pad(fg_mask_flat,     (0, pad_len))

        # ── 모드 분기: scale_aware이면 별도 메서드로 처리 ──
        if self.moe_mode == 'scale_aware' and spatial_shapes is not None:
            return self._forward_scale_aware(
                src, key_padding_mask, spatial_shapes,
                per_scale_pad_info, orig_N_total,
                soft_labels=soft_labels_flat,
                fg_mask=fg_mask_flat)

        # ── 0. JAX 원본 V-MoE 방식 (baseline): group_size = moe_group_images * N_total ──
        group_size = self.moe_group_images * N_total
        assert B % self.moe_group_images == 0, (
            f"batch_size({B})는 moe_group_images({self.moe_group_images})로 "
            f"나누어 떨어져야 합니다.")
        num_groups = B // self.moe_group_images

        src_grouped = src.reshape(num_groups, group_size, D)       # (G, S, D)

        # key_padding_mask도 동일하게 reshape: (B, N_total) → (G, S)
        if key_padding_mask is not None:
            mask_grouped = key_padding_mask.reshape(num_groups, group_size)
            #정답지인 key_padding도 reshape (B, N_total) → (G, S) ==mask_grouped
        else:
            mask_grouped = None

        # ── 1. 패딩 마스킹: gate logit에 -1e4 → softmax 후 gate≈0 ──
        # -inf를 사용하면 softmax([-inf,...,-inf]) = NaN (0/0) 발생.
        # -1e4를 사용하면 e^(-1e4) ≈ 0 이므로 NaN 없이 near-zero 확률 출력.
        # 0.0은 사용 불가: e^0 = 1 이므로 패딩 토큰이 유효 확률을 뺏어감.
        gate_logits = self.router.dense(src_grouped)  # (G, S, E) =>패딩도 로짓값 구해짐
        if mask_grouped is not None:
            gate_logits = gate_logits.masked_fill(
                mask_grouped.unsqueeze(-1), -1e4) #mask_group인경우 로직값이 무엇이든 -1e4로 -inf대신 =>NAN안나옴

        # ── 2. Router: softmax + aux loss 계산 (패딩 토큰 제외) ──
        gates_softmax, metrics = self._compute_gates_and_metrics(
            gate_logits, mask_grouped)
        #deterministic일때 return== gates_softmax, metrics
        #non-detministic일때 return ==gates_softmax_noisy, metrics  즉 노이즈가 낀 값을 넘겨줌

        # ── 2.5. Class Routing Loss (baseline) ──
        if soft_labels_flat is not None:
            sl_gs = soft_labels_flat.reshape(num_groups, group_size, self.num_classes)
            fm_gs = fg_mask_flat.reshape(num_groups, group_size)
            class_loss = self._compute_class_routing_loss(
                gates_softmax, sl_gs, fm_gs, pad_mask_gs=mask_grouped)
            metrics['class_routing_loss'] = class_loss
            metrics['auxiliary_loss'] = (
                metrics['auxiliary_loss'] +
                self.class_routing_loss_weight * class_loss)

        # ── 3. Dispatcher 생성 ──
        # capacity는 gates.shape[1] = group_size (패딩 포함 가능) 기준으로 자동 계산.
        # EP 환경에서는 all_reduce(MAX)로 모든 rank의 N_total이 동일하게 맞춰졌으므로
        # padded group_size 기반 capacity가 모든 rank에서 같음 → ALLTOALL 크기 일치 보장.
        # (per-rank valid 토큰 기반으로 계산하면 rank마다 capacity가 달라져 deadlock 발생)
        dispatcher = self.router._create_dispatcher(gates_softmax)
        #routing함수의 _create_dispatcher 만 가져다 씀! 나머지X

        # ── 3.5 라우팅 캡처(capture_routing=True일 때만) ──
        # capture_routing은 기본 False. main.py에서 5-epoch 윈도우 체크 시에만
        # 잠깐 True로 켜고 고정 이미지 N장 forward 후 다시 False로 복원.
        if self.capture_routing:
            with torch.no_grad():
                E_glob = self.num_experts
                # gate_logits: 노이즈 추가 전 clean logit → 라우터의 순수한 의도
                # topk(2)로 top-1, top-2 expert 인덱스 동시 추출
                top2_idx = gate_logits.topk(2, dim=-1).indices  # (G, S, 2)
                top1 = top2_idx[..., 0]                         # (G, S): 1순위 expert
                top2 = top2_idx[..., 1]                         # (G, S): 2순위 expert
                if mask_grouped is not None:
                    top1 = top1.masked_fill(mask_grouped, -1)   # 패딩 토큰은 -1 → 회색으로 표시
                    top2 = top2.masked_fill(mask_grouped, -1)

                route_map_top1 = top1.reshape(B, N_total)       # (B, N_maybe_padded)
                route_map_top2 = top2.reshape(B, N_total)

                # overflow_map: K개 assignment 모두 drop된 토큰 (= 실제 미처리)
                # 이 토큰들 위에 시각화에서 빨간 X 오버레이
                if hasattr(dispatcher, 'indices'):
                    idx       = dispatcher.indices
                    exp_idx   = idx[..., 0]
                    buf_idx   = idx[..., 1]
                    cap       = dispatcher.capacity
                    valid_exp  = (exp_idx >= 0) & (exp_idx < E_glob)
                    valid_buf  = (buf_idx  >= 0) & (buf_idx  < cap)
                    valid_both = valid_exp & valid_buf

                    # 시각화 stats: overflow_rate, load_cv, tokens_per_expert
                    # overflow: (G, S, K) — [..., 0]=top-1 배정, [..., 1]=top-2 배정
                    overflow = valid_exp & (~valid_buf)
                    if mask_grouped is not None:
                        overflow_v = overflow & (~mask_grouped.unsqueeze(-1))
                        valid_exp_v = valid_exp & (~mask_grouped.unsqueeze(-1))
                    else:
                        overflow_v = overflow
                        valid_exp_v = valid_exp
                    metrics['overflow_rate'] = (
                        overflow_v.sum().float() / valid_exp_v.sum().float().clamp(min=1))
                    metrics['overflow_rate_top1'] = (
                        overflow_v[..., 0].sum().float() / valid_exp_v[..., 0].sum().float().clamp(min=1))
                    metrics['overflow_rate_top2'] = (
                        overflow_v[..., 1].sum().float() / valid_exp_v[..., 1].sum().float().clamp(min=1))
                    # tokens_per_expert: router가 실제로 각 expert에 배정한 토큰 수
                    # (buffer 상한과 무관 → router의 배분 의도를 그대로 반영)
                    # 패딩 토큰 제외: mask_grouped=True 위치는 카운트에서 빼야 함
                    if mask_grouped is not None:
                        routing_valid = valid_exp & (~mask_grouped.unsqueeze(-1))
                    else:
                        routing_valid = valid_exp
                    one_hot_e = F.one_hot(exp_idx.clamp(0, E_glob - 1), E_glob).float()
                    # one_hot_e: (G, S, K, E_glob), routing_valid: (G, S, K)
                    # unsqueeze(-1) → (G, S, K, 1)로 만들어야 (G, S, K, E_glob)와 broadcast 가능
                    one_hot_e = one_hot_e * routing_valid.float().unsqueeze(-1)
                    # one_hot_e: (G, S, K, E_glob)
                    # K=0: top-1 배정, K=1: top-2 배정
                    tpe       = one_hot_e.sum(dim=(0, 1, 2))          # (E,) 전체 합산
                    tpe_top1  = one_hot_e[:, :, 0, :].sum(dim=(0, 1)) # (E,) top-1만
                    tpe_top2  = one_hot_e[:, :, 1, :].sum(dim=(0, 1)) # (E,) top-2만
                    metrics['tokens_per_expert']      = tpe
                    metrics['tokens_per_expert_top1'] = tpe_top1
                    metrics['tokens_per_expert_top2'] = tpe_top2
                    metrics['load_cv']      = tpe.std()      / tpe.mean().clamp(min=1e-8)
                    metrics['load_cv_top1'] = tpe_top1.std() / tpe_top1.mean().clamp(min=1e-8)
                    metrics['load_cv_top2'] = tpe_top2.std() / tpe_top2.mean().clamp(min=1e-8)

                    any_processed = valid_both.any(dim=-1)       # (G, S)
                    if mask_grouped is not None:
                        completely_dropped = (~any_processed) & (~mask_grouped)
                    else:
                        completely_dropped = ~any_processed
                    overflow_map = completely_dropped.reshape(B, N_total)

                    # top-1, top-2 각각 overflow된 토큰 위치 (buffer 꽉 참, 패딩 제외)
                    if mask_grouped is not None:
                        overflow_map_top1 = (overflow[..., 0] & ~mask_grouped).reshape(B, N_total)
                        overflow_map_top2 = (overflow[..., 1] & ~mask_grouped).reshape(B, N_total)
                    else:
                        overflow_map_top1 = overflow[..., 0].reshape(B, N_total)
                        overflow_map_top2 = overflow[..., 1].reshape(B, N_total)
                else:
                    overflow_map      = torch.zeros(B, N_total, dtype=torch.bool, device=src.device)
                    overflow_map_top1 = torch.zeros(B, N_total, dtype=torch.bool, device=src.device)
                    overflow_map_top2 = torch.zeros(B, N_total, dtype=torch.bool, device=src.device)

                # EP 패딩 제거: GPU간 통신용으로 추가했던 패딩 토큰 strip
                if per_scale_pad_info is not None and orig_N_total < N_total:
                    orig_per_level, padded_per_level = per_scale_pad_info
                    ppl = padded_per_level.tolist()
                    opl = [orig_per_level[l].item() for l in range(len(orig_per_level))]
                    def _trim(t):
                        parts = t.split(ppl, dim=1)
                        return torch.cat(
                            [parts[l][:, :opl[l]] for l in range(len(opl))], dim=1)
                    route_map_top1    = _trim(route_map_top1)
                    route_map_top2    = _trim(route_map_top2)
                    overflow_map      = _trim(overflow_map)
                    overflow_map_top1 = _trim(overflow_map_top1)
                    overflow_map_top2 = _trim(overflow_map_top2)
                elif orig_N_total < N_total:
                    route_map_top1    = route_map_top1[:, :orig_N_total]
                    route_map_top2    = route_map_top2[:, :orig_N_total]
                    overflow_map      = overflow_map[:, :orig_N_total]
                    overflow_map_top1 = overflow_map_top1[:, :orig_N_total]
                    overflow_map_top2 = overflow_map_top2[:, :orig_N_total]

                # flat (B, N_total) → scale별 (B, H_l, W_l) 리스트로 분리
                if spatial_shapes is not None:
                    scale_sizes = [
                        int(spatial_shapes[l, 0]) * int(spatial_shapes[l, 1])
                        for l in range(spatial_shapes.shape[0])]
                    rm1_s  = route_map_top1.split(scale_sizes, dim=1)
                    rm2_s  = route_map_top2.split(scale_sizes, dim=1)
                    om_s   = overflow_map.split(scale_sizes, dim=1)
                    om1_s  = overflow_map_top1.split(scale_sizes, dim=1)
                    om2_s  = overflow_map_top2.split(scale_sizes, dim=1)
                    route_maps_top1, route_maps_top2 = [], []
                    overflow_maps, overflow_maps_top1, overflow_maps_top2 = [], [], []
                    for l in range(spatial_shapes.shape[0]):
                        H_l = int(spatial_shapes[l, 0])
                        W_l = int(spatial_shapes[l, 1])
                        route_maps_top1.append(rm1_s[l].reshape(B, H_l, W_l).cpu())
                        route_maps_top2.append(rm2_s[l].reshape(B, H_l, W_l).cpu())
                        overflow_maps.append(om_s[l].reshape(B, H_l, W_l).cpu())
                        overflow_maps_top1.append(om1_s[l].reshape(B, H_l, W_l).cpu())
                        overflow_maps_top2.append(om2_s[l].reshape(B, H_l, W_l).cpu())
                    metrics['route_maps_top1']    = route_maps_top1   # list[(B,H,W)]
                    metrics['route_maps_top2']    = route_maps_top2
                    metrics['overflow_maps']      = overflow_maps      # 둘 다 실패한 토큰
                    metrics['overflow_maps_top1'] = overflow_maps_top1 # top-1만 실패
                    metrics['overflow_maps_top2'] = overflow_maps_top2 # top-2만 실패
                    metrics['spatial_shapes']     = spatial_shapes.cpu()
                else:
                    metrics['route_map_top1_flat'] = route_map_top1.cpu()
                    metrics['route_map_top2_flat'] = route_map_top2.cpu()
                    metrics['overflow_map_flat']   = overflow_map.cpu()

        # ── 4. Dispatch → Expert 처리 → Combine ──
        # EP 시: dispatch → all-to-all → (E_local, W*G*C, D)
        #         experts → (E_local, W*G*C, D)
        #         combine → all-to-all → (G, S, D)
        dispatched = dispatcher.dispatch(src_grouped)   # (E_local, G*C, D) or (E_local, W*G*C, D)
        expert_out = self.experts(dispatched)            # same shape
        output = dispatcher.combine(expert_out)          # (G, S, D)
  
        # ── 5. 원래 shape으로 복원: (G, S, D) → (B, N_total, D) ──
        output = output.reshape(B, N_total, D)

        # ── 패딩했던 부분 제거: 원래 N_total 길이로 복원 ──
        if per_scale_pad_info is not None and orig_N_total < N_total:
            # per-scale strip: 각 scale에서 패딩 제거 후 다시 concat
            orig_per_level, padded_per_level = per_scale_pad_info
            out_splits = output.split(padded_per_level.tolist(), dim=1)
            trimmed = [out_splits[lvl][:, :orig_per_level[lvl].item(), :]
                       for lvl in range(len(orig_per_level))]
            output = torch.cat(trimmed, dim=1).contiguous()
        elif orig_N_total < N_total:
            # flat-end fallback
            output = output[:, :orig_N_total, :].contiguous()

        return output, metrics

    def _compute_gates_and_metrics(
            self, gate_logits: torch.Tensor,
            mask_grouped: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """이미 계산된 gate_logits로부터 softmax + auxiliary loss 계산.

        패딩 마스킹이 이미 적용된 gate_logits을 받으므로
        Router의 _compute_gates_softmax_and_metrics를 풀어서 사용.

        패딩 토큰은 aux loss 계산에서 제외:
        - gshard_loss: 패딩 토큰의 argmax가 expert 0으로 집중되는 bias 방지
        - importance_loss: 패딩 토큰이 기여하지 않도록 제외
        - load_loss: 패딩 토큰의 threshold 계산 제외
        """
        router = self.router
        # NaN 방지: softmax 후 padding 위치를 0.0으로 교체
        
        gates_softmax = F.softmax(gate_logits, dim=-1) #gate_logits (G,S,E)
        if mask_grouped is not None:
            gates_softmax = gates_softmax.masked_fill(mask_grouped.unsqueeze(-1), 0.0)#아예 softmax값 자체도 0으로
        #======여기까지는 expert에 대한 패딩포함(0으로 해놓음) softmax들 모음 ====

        #아래는 matrix loss구해놓는부분
        if router.deterministic or router.noise_std == 0.0 or not self.training:
            # 패딩 제외한 loss 계산
            importance_loss = self._importance_loss_masked(gates_softmax, mask_grouped)
            gshard_loss = self._gshard_loss_masked(gates_softmax, mask_grouped)
            metrics = {
                "auxiliary_loss": _weighted_sum(
                    (router.gshard_loss_weight, gshard_loss),
                    (router.importance_loss_weight, importance_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
            }
            return gates_softmax, metrics
        else:
            noise_std = (1.0 / router.num_experts) * router.noise_std
            logits_noise = noise_std * torch.randn_like(gate_logits)
            gates_logits_noisy = gate_logits + logits_noise
            gates_softmax_noisy = F.softmax(gates_logits_noisy, dim=-1)
            if mask_grouped is not None:
                gates_softmax_noisy = gates_softmax_noisy.masked_fill(mask_grouped.unsqueeze(-1), 0.0)

            # 패딩 제외한 loss 계산
            importance_loss = self._importance_loss_masked(gates_softmax, mask_grouped)
            load_loss = self._load_loss_masked(
                gate_logits, gates_logits_noisy, noise_std, mask_grouped)
            gshard_loss = self._gshard_loss_masked(gates_softmax_noisy, mask_grouped)

            metrics = {
                "auxiliary_loss": _weighted_sum(
                    (router.gshard_loss_weight, gshard_loss),
                    (router.importance_loss_weight, importance_loss),
                    (router.load_loss_weight, load_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
                "load_loss": load_loss,
            }
            return gates_softmax_noisy, metrics

    # ── 패딩 제외 auxiliary loss 함수들 ──

    def _gshard_loss_masked(self, gates: torch.Tensor,
                            mask: Optional[torch.Tensor]) -> torch.Tensor:
        """G-Shard loss에서 패딩 토큰을 제외.

        gates: (G, S, E)
        mask: (G, S) — True = 패딩
        """
        _, _, num_experts = gates.shape

        if mask is None:
            return self.router._gshard_auxiliary_loss_batched(gates)

        # 패딩이 아닌 토큰만 사용
        # ~mask: True = 유효한 토큰
        valid = (~mask).float().unsqueeze(-1)  # (G, S, 1)
        valid_count = valid.sum(dim=1).clamp(min=1.0)  # (G, 1) — 그룹별 유효 토큰 수

        # mean_gates_per_expert: 유효 토큰의 gate 평균
        mean_gates = (gates * valid).sum(dim=1) / valid_count  # (G, E)

        # mean_top1_per_expert: 유효 토큰 중 top-1 비율
        # 패딩 토큰의 gate는 이미 0이므로 argmax 결과에 영향을 주지 않도록
        # 패딩 토큰의 argmax를 무시
        gates_masked = gates.clone()
        gates_masked[mask] = -1.0  # 패딩 위치를 -1로 (argmax에서 선택 안됨)
        top1 = F.one_hot(gates_masked.argmax(dim=2), num_experts).float()  # (G, S, E)
        mean_top1 = (top1 * valid).sum(dim=1) / valid_count  # (G, E)

        auxiliary_loss = (mean_top1 * mean_gates).mean(dim=1)  # (G,)
        auxiliary_loss = auxiliary_loss * num_experts ** 2
        return auxiliary_loss.mean()

    def _importance_loss_masked(self, gates: torch.Tensor,
                                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Importance loss에서 패딩 토큰을 제외.

        gates: (G, S, E)
        mask: (G, S) — True = 패딩
        """
        if mask is None:
            return self.router._importance_auxiliary_loss_batched(gates)
            #마스크없음 바로 통화

        valid = (~mask).float().unsqueeze(-1)  # (G, S, 1) 
        #=> ~(반대)를통해 원래 true=패딩 false=픽셀 =>> true=픽셀 false=패딩으로 교체 .float => 픽셀=1.0 ,패딩=0.0이된다
        # 유효 토큰의 gate 합 = importance per expert
        importance = (gates * valid).sum(dim=1)  # (G, E)
        std_imp = importance.std(dim=-1, correction=0)  # (G,)
        mean_imp = importance.mean(dim=-1)  # (G,)
        cv_sq = (std_imp / mean_imp.clamp(min=1e-8)) ** 2  # (G,)
        return cv_sq.mean()

    def _load_loss_masked(self, logits: torch.Tensor,
                          logits_noisy: torch.Tensor,
                          noise_std: float,
                          mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Load loss에서 패딩 토큰을 제외.

        logits, logits_noisy: (G, S, E)
        mask: (G, S) — True = 패딩

        패딩 토큰의 gate_logits = -inf이므로 noise를 더해도 -inf.
        threshold = -inf, gate = -inf → noise_required = -inf - (-inf) = NaN 발생.
        Normal.cdf(NaN) → 런타임 에러.
        따라서 유효한 토큰만으로 계산하고, 패딩 토큰은 완전히 제외한다.
        """
        if mask is None:
            return self.router._load_auxiliary_loss_batched(
                logits, logits_noisy, noise_std)

        num_experts = logits_noisy.shape[-1]
        valid = (~mask).float()  # (G, S) — 1.0 = 유효, 0.0 = 패딩

        # 패딩 위치를 -1e4로 대체하여 topk에서 선택되지 않도록 함.
        # 0.0을 사용하면 유효 토큰의 logits가 음수일 때 패딩이 topk에 진입하여
        # threshold 계산을 왜곡함. -1e4는 충분히 작아서 topk에서 절대 선택 안 됨.
        logits_safe = logits.clone()
        logits_noisy_safe = logits_noisy.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(logits)  # (G, S, E)
        logits_safe[mask_expanded] = -1e4
        logits_noisy_safe[mask_expanded] = -1e4

        # Threshold: K-th largest noisy logit per item
        _, top_indices = torch.topk(
            logits_noisy_safe, self.router.num_selected_experts, dim=-1)
        threshold_index = top_indices[..., -1] #top-k중 마지막 ==threshold 인덱스
        threshold_per_item = (
            F.one_hot(threshold_index, num_experts).float() * logits_noisy_safe
        ).sum(dim=-1)  # (G, S)  logits_noisy_safe 이거 노이즈만있는것이 X  **gate_logits + logits_noise 이값임**

        # Noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits_safe
        noise_required_to_win = noise_required_to_win / noise_std

        # Probability via Gaussian CDF (NaN-free)
        normal = torch.distributions.Normal(0, 1)
        p = 1.0 - normal.cdf(noise_required_to_win)  # (G, S, E)

        # 패딩 토큰 제외: 유효 토큰만 평균
        valid_3d = valid.unsqueeze(-1)  # (G, S, 1)
        valid_count = valid_3d.sum(dim=1).clamp(min=1.0)  # (G, 1)
        p_mean = (p * valid_3d).sum(dim=1) / valid_count  # (G, E)

        std_p = p_mean.std(dim=-1, correction=0)
        mean_p = p_mean.mean(dim=-1)
        cv_sq = (std_p / mean_p.clamp(min=1e-8)) ** 2
        return cv_sq.mean()

    def _forward_scale_aware(
            self,
            src: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor],
            spatial_shapes: torch.Tensor,
            per_scale_pad_info,
            orig_N_total: int,
            soft_labels: Optional[torch.Tensor] = None,
            fg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Scale-Aware MoE: scale별로 독립적으로 라우팅.

        각 scale의 토큰들이 자기들끼리만 expert 경쟁을 하도록 처리.
        scale4(25토큰)도 자체 capacity를 받아 drop 없이 처리됨.

        흐름:
          src (B, N_total, D)
          → scale별 split: [(B, S_1, D), (B, S_2, D), ...]
          → 각 scale마다 독립적으로 group → gate → dispatch → expert → combine
          → torch.cat(dim=1) → (B, N_total, D)
        """
        B, N_total, D = src.shape
        num_levels = spatial_shapes.shape[0] # 4scale (L(4),2(h,w))

        # per_scale_pad_info: EP 패딩이 있으면 (orig_per_level, padded_per_level)
        if per_scale_pad_info is not None:
            orig_per_level, padded_per_level = per_scale_pad_info 
            #orig_per_level =local_tokens_per_level, padded_per_level=max_tokens_per_level
        else:
            orig_per_level = spatial_shapes[:, 0] * spatial_shapes[:, 1]   # (L,)
            padded_per_level = orig_per_level

        # (B, N_total, D) → scale별 분리 (패딩 포함 크기 기준)
        src_splits = src.split(padded_per_level.tolist(), dim=1)
        mask_splits = (key_padding_mask.split(padded_per_level.tolist(), dim=1)
                       if key_padding_mask is not None else [None] * num_levels)

        assert B % self.moe_group_images == 0, (
            f"batch_size({B})는 moe_group_images({self.moe_group_images})로 "
            f"나누어 떨어져야 합니다.")
        num_groups = B // self.moe_group_images   # G

        # soft_labels를 level별로 미리 split.
        # src와 동일하게 padded_per_level 기준으로 split해야 함.
        # forward()에서 이미 EP 패딩이 반영된 채로 전달되므로 padded 크기가 일치.
        sl_splits = fm_splits = None
        if soft_labels is not None:
            sl_splits = soft_labels.split(padded_per_level.tolist(), dim=1)  # list[(B, S_l_padded, C)]
            fm_splits = fg_mask.split(padded_per_level.tolist(),     dim=1)  # list[(B, S_l_padded)]

        all_output_splits = []
        all_metrics_lists = {}

        # capture_routing=True일 때 scale별 route_map 수집용 리스트
        if self.capture_routing:
            route_maps_top1, route_maps_top2 = [], []
            overflow_maps, overflow_maps_top1, overflow_maps_top2 = [], [], []

        for lvl in range(num_levels):
            src_l  = src_splits[lvl]    # (B, S_l, D) #padding이 포함된 원본상태
            mask_l = mask_splits[lvl]   # (B, S_l) or None
            S_l    = src_l.shape[1] #패딩이 포함된 해당 scale의 토큰 수.

            # ── 그룹화: (B, S_l, D) → (G, group_size_l, D) ──
            group_size_l = self.moe_group_images * S_l
            src_l_grouped = src_l.reshape(num_groups, group_size_l, D)
            mask_l_grouped = (mask_l.reshape(num_groups, group_size_l)
                              if mask_l is not None else None)

            # ── Gate logit + 패딩 마스킹 ──
            gate_logits_l = self.router.dense(src_l_grouped)   # (G, group_size_l, E)
            if mask_l_grouped is not None:
                gate_logits_l = gate_logits_l.masked_fill(
                    mask_l_grouped.unsqueeze(-1), -1e4)

            # ── Softmax + aux loss (패딩 제외) ──
            gates_l, metrics_l = self._compute_gates_and_metrics(
                gate_logits_l, mask_l_grouped)

            # ── Class Routing Loss (scale_aware, level별) ──
            if sl_splits is not None:
                sl_l = sl_splits[lvl]   # (B, S_l, C) — 이미 padded_per_level 크기
                fm_l = fm_splits[lvl]   # (B, S_l)
                sl_l_gs = sl_l.reshape(num_groups, group_size_l, self.num_classes)
                fm_l_gs = fm_l.reshape(num_groups, group_size_l)
                class_loss_l = self._compute_class_routing_loss(
                    gates_l, sl_l_gs, fm_l_gs, pad_mask_gs=mask_l_grouped)
                metrics_l['class_routing_loss'] = class_loss_l
                metrics_l['auxiliary_loss'] = (
                    metrics_l['auxiliary_loss'] +
                    self.class_routing_loss_weight * class_loss_l)

            # ── Dispatcher 생성 ──
            # capacity는 gates.shape[1] = group_size_l (EP padding 포함 가능) 기준으로 자동 계산.
            # EP 환경에서는 all_reduce(MAX)로 모든 rank의 S_l이 동일하게 맞춰졌으므로
            # padded S_l 기반 capacity가 모든 rank에서 같음 → ALLTOALL 크기 일치 보장.
            # (per-rank valid 토큰 기반으로 계산하면 rank마다 capacity가 달라져 deadlock 발생)
            dispatcher_l = self.router._create_dispatcher(gates_l)

            # ── 캡처(capture_routing=True일 때만) ──
            if self.capture_routing:
                with torch.no_grad():
                    E_glob = self.num_experts
                    top2_idx_l = gate_logits_l.topk(2, dim=-1).indices  # (G, group_size_l, 2)
                    top1_l = top2_idx_l[..., 0]
                    top2_l = top2_idx_l[..., 1]
                    if mask_l_grouped is not None:
                        top1_l = top1_l.masked_fill(mask_l_grouped, -1)
                        top2_l = top2_l.masked_fill(mask_l_grouped, -1)

                    H_l = int(spatial_shapes[lvl, 0])
                    W_l = int(spatial_shapes[lvl, 1])
                    # EP 패딩 포함 S_l → 원래 orig 크기로 잘라서 (B, H, W)로 reshape
                    S_l_orig = (orig_per_level[lvl].item()
                                if per_scale_pad_info is not None else S_l)
                    route_maps_top1.append(
                        top1_l.reshape(B, S_l)[:, :S_l_orig].reshape(B, H_l, W_l).cpu())
                    route_maps_top2.append(
                        top2_l.reshape(B, S_l)[:, :S_l_orig].reshape(B, H_l, W_l).cpu())

                    if hasattr(dispatcher_l, 'indices'):
                        idx_l       = dispatcher_l.indices
                        exp_idx_l   = idx_l[..., 0]
                        buf_idx_l   = idx_l[..., 1]
                        cap_l       = dispatcher_l.capacity
                        valid_exp_l  = (exp_idx_l >= 0) & (exp_idx_l < E_glob)
                        valid_buf_l  = (buf_idx_l  >= 0) & (buf_idx_l  < cap_l)
                        valid_both_l = valid_exp_l & valid_buf_l

                        # 시각화 stats: overflow_rate, load_cv, tokens_per_expert
                        # overflow_l: (G, S_l, K) — [..., 0]=top-1, [..., 1]=top-2
                        overflow_l = valid_exp_l & (~valid_buf_l)
                        if mask_l_grouped is not None:
                            overflow_l_v = overflow_l & (~mask_l_grouped.unsqueeze(-1))
                            valid_exp_l_v = valid_exp_l & (~mask_l_grouped.unsqueeze(-1))
                        else:
                            overflow_l_v = overflow_l
                            valid_exp_l_v = valid_exp_l
                        metrics_l['overflow_rate'] = (
                            overflow_l_v.sum().float() / valid_exp_l_v.sum().float().clamp(min=1))
                        metrics_l['overflow_rate_top1'] = (
                            overflow_l_v[..., 0].sum().float() / valid_exp_l_v[..., 0].sum().float().clamp(min=1))
                        metrics_l['overflow_rate_top2'] = (
                            overflow_l_v[..., 1].sum().float() / valid_exp_l_v[..., 1].sum().float().clamp(min=1))
                        # tokens_per_expert: router의 배분 의도 (buffer 상한 무관, 패딩 제외)
                        if mask_l_grouped is not None:
                            routing_valid_l = valid_exp_l & (~mask_l_grouped.unsqueeze(-1))
                        else:
                            routing_valid_l = valid_exp_l
                        one_hot_l = F.one_hot(exp_idx_l.clamp(0, E_glob - 1), E_glob).float()
                        # one_hot_l: (G, S_l, K, E_glob), routing_valid_l: (G, S_l, K)
                        # unsqueeze(-1) → (G, S_l, K, 1)로 만들어야 broadcast 가능
                        one_hot_l = one_hot_l * routing_valid_l.float().unsqueeze(-1)
                        tpe_l      = one_hot_l.sum(dim=(0, 1, 2))
                        tpe_l_top1 = one_hot_l[:, :, 0, :].sum(dim=(0, 1))
                        tpe_l_top2 = one_hot_l[:, :, 1, :].sum(dim=(0, 1))
                        metrics_l['tokens_per_expert']      = tpe_l
                        metrics_l['tokens_per_expert_top1'] = tpe_l_top1
                        metrics_l['tokens_per_expert_top2'] = tpe_l_top2
                        metrics_l['load_cv']      = tpe_l.std()      / tpe_l.mean().clamp(min=1e-8)
                        metrics_l['load_cv_top1'] = tpe_l_top1.std() / tpe_l_top1.mean().clamp(min=1e-8)
                        metrics_l['load_cv_top2'] = tpe_l_top2.std() / tpe_l_top2.mean().clamp(min=1e-8)

                        any_proc_l = valid_both_l.any(dim=-1)     # (G, group_size_l)
                        if mask_l_grouped is not None:
                            drop_l = (~any_proc_l) & (~mask_l_grouped)
                        else:
                            drop_l = ~any_proc_l
                        overflow_maps.append(
                            drop_l.reshape(B, S_l)[:, :S_l_orig].reshape(B, H_l, W_l).cpu())
                        # top-1, top-2 각각 overflow 위치 (패딩 제외)
                        overflow_maps_top1.append(
                            overflow_l_v[..., 0].reshape(B, S_l)[:, :S_l_orig].reshape(B, H_l, W_l).cpu())
                        overflow_maps_top2.append(
                            overflow_l_v[..., 1].reshape(B, S_l)[:, :S_l_orig].reshape(B, H_l, W_l).cpu())
                    else:
                        overflow_maps.append(torch.zeros(B, H_l, W_l, dtype=torch.bool))
                        overflow_maps_top1.append(torch.zeros(B, H_l, W_l, dtype=torch.bool))
                        overflow_maps_top2.append(torch.zeros(B, H_l, W_l, dtype=torch.bool))

            # ── Dispatch → Expert → Combine ──
            dispatched_l = dispatcher_l.dispatch(src_l_grouped)
            expert_out_l = self.experts(dispatched_l)
            output_l     = dispatcher_l.combine(expert_out_l)   # (G, group_size_l, D)

            # (G, group_size_l, D) → (B, S_l, D)
            output_l = output_l.reshape(B, S_l, D)
            all_output_splits.append(output_l)

            # 메트릭 수집 (scale별 평균 내기 위해 리스트로)
            for k, v in metrics_l.items():
                all_metrics_lists.setdefault(k, []).append(v)

        # ── auxiliary loss: scale 단순 평균 ──
        metrics = {k: torch.stack(v).mean() for k, v in all_metrics_lists.items()}

        # tokens_per_expert: scale 평균이 아닌 합산 (시각화 bar chart: 전체 토큰 수)
        for tpe_key in ('tokens_per_expert', 'tokens_per_expert_top1', 'tokens_per_expert_top2'):
            if tpe_key in all_metrics_lists:
                metrics[tpe_key] = torch.stack(
                    all_metrics_lists[tpe_key]).sum(dim=0)

        if self.capture_routing:
            metrics['route_maps_top1']    = route_maps_top1
            metrics['route_maps_top2']    = route_maps_top2
            metrics['overflow_maps']      = overflow_maps       # 둘 다 실패
            metrics['overflow_maps_top1'] = overflow_maps_top1 # top-1만 실패
            metrics['overflow_maps_top2'] = overflow_maps_top2 # top-2만 실패
            metrics['spatial_shapes']     = spatial_shapes.cpu()

        # ── Concat: (B, N_padded, D) ──
        output = torch.cat(all_output_splits, dim=1)

        # ── per-scale 패딩 제거 ──
        if per_scale_pad_info is not None and orig_N_total < output.shape[1]:
            out_splits = output.split(padded_per_level.tolist(), dim=1)
            trimmed = [out_splits[lvl][:, :orig_per_level[lvl].item(), :]
                       for lvl in range(num_levels)]
            output = torch.cat(trimmed, dim=1).contiguous()

        return output, metrics
