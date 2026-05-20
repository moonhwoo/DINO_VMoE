"""DINO Deformable Transformer Encoder Layer with MoE FFN.

기존 DeformableTransformerEncoderLayer에서 FFN을 DINOMoEBlock으로 교체.
구조: Deformable Self-Attention → MoE(FFN) → (Optional) Channel Attention

반환값 변경:
- 기존: return src
- MoE:  return src, moe_metrics
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.dino.ops.modules import MSDeformAttn
from models.moe.dino_moe_block import DINOMoEBlock


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """DINO의 activation 함수 생성 (deformable_transformer.py에서 가져옴)."""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return nn.SELU(inplace=True)
    raise RuntimeError(f"activation should be relu/gelu/glu/prelu/selu, not {activation}.")


class DeformableTransformerEncoderMoELayer(nn.Module):
    """DeformableTransformerEncoderLayer에서 FFN → MoE 교체.

    기존 layer와 동일한 forward 시그니처를 유지하되,
    반환값이 (src, moe_metrics) 튜플로 변경됨.
    TransformerEncoder에서 isinstance(result, tuple)로 분기 처리.
    """

    def __init__(self,
                 d_model: int = 256,
                 d_ffn: int = 2048,
                 dropout: float = 0.0,
                 activation: str = "relu",
                 n_levels: int = 4,
                 n_heads: int = 8,
                 n_points: int = 4,
                 add_channel_attention: bool = False,
                 use_deformable_box_attn: bool = False,
                 box_attn_type: str = 'roi_align',
                 # MoE params
                 num_experts: int = 8,
                 num_selected_experts: int = 2,
                 capacity_factor: float = 1.25,
                 noise_std: float = 1.0,
                 gshard_loss_weight: float = 0.01,
                 importance_loss_weight: float = 1.0,
                 load_loss_weight: float = 1.0,
                 moe_group_images: int = 1,
                 expert_parallel: bool = False,
                 split_rngs: bool = False,
                 moe_mode: str = 'baseline'):
        super().__init__()

        # ── Self Attention (기존과 동일) ──
        if use_deformable_box_attn:
            from models.dino.attention import MSDeformableBoxAttention
            self.self_attn = MSDeformableBoxAttention(
                d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ── MoE Block (FFN 대체) ──
        self.moe_block = DINOMoEBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            num_experts=num_experts,
            num_selected_experts=num_selected_experts,
            capacity_factor=capacity_factor,
            noise_std=noise_std,
            dropout=dropout,
            gshard_loss_weight=gshard_loss_weight,
            importance_loss_weight=importance_loss_weight,
            load_loss_weight=load_loss_weight,
            moe_group_images=moe_group_images,
            expert_parallel=expert_parallel,
            split_rngs=split_rngs,
            moe_mode=moe_mode,
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ── Channel Attention (호환 유지) ──
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, key_padding_mask=None):
        """
        Args:
            src: (bs, sum(hi*wi), d_model)
            pos: (bs, sum(hi*wi), d_model)
            reference_points: (bs, sum(hi*wi), n_levels, 2)
            spatial_shapes: (n_levels, 2)
            level_start_index: (n_levels,)
            key_padding_mask: (bs, sum(hi*wi)) — True = 패딩

        Returns:
            src: (bs, sum(hi*wi), d_model)
            moe_metrics: dict — auxiliary_loss 등
        """
        # ── 1. Deformable Self-Attention (기존과 동일) ──
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points, src, spatial_shapes,
            level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ── 2. MoE Block (FFN 대체) ──
        #key_padding_mask: (B, N_total) =>concat되어있는 1D시퀀스 padding이면 true , 아니면 false
        src2, moe_metrics = self.moe_block(src, key_padding_mask=key_padding_mask, spatial_shapes=spatial_shapes)
        src = src + self.dropout3(src2)   # residual connection
        src = self.norm2(src)             # layer norm

        # ── 3. Channel Attention (호환 유지) ──
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src, moe_metrics
