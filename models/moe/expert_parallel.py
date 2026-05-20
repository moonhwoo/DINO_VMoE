"""Expert Parallelism (EP) utilities for DINO-MoE.

2 GPU + 8 experts 예시:
  GPU 0: expert 0,1,2,3   GPU 1: expert 4,5,6,7
  all-to-all로 토큰을 교환하여 각 GPU가 local expert만 실행

Usage:
    # 학습 시작 시 (torch.distributed.init_process_group 이후)
    from models.moe.expert_parallel import init_expert_parallel
    init_expert_parallel()

    # DDP 대신 backward 후:
    from models.moe.expert_parallel import sync_shared_gradients
    loss.backward()
    sync_shared_gradients(model)
    optimizer.step()
"""
import torch
import torch.distributed as dist
from typing import Optional


# ── Module-level EP state ──────────────────────────────────────────────────
_ep_group: Optional[dist.ProcessGroup] = None
_ep_world_size: int = 1
_ep_rank: int = 0


def init_expert_parallel(group: Optional[dist.ProcessGroup] = None):
    """Initialize expert parallelism state.

    Call this AFTER torch.distributed.init_process_group().
    Args:
        group: Optional custom process group for EP.
               If None, uses the default (WORLD) group.
    """
    global _ep_group, _ep_world_size, _ep_rank
    if group is not None:
        _ep_group = group
    elif dist.is_initialized():
        _ep_group = dist.group.WORLD
    else:
        _ep_group = None
    _ep_world_size = dist.get_world_size(_ep_group) if _ep_group else 1
    _ep_rank = dist.get_rank(_ep_group) if _ep_group else 0


def get_ep_group() -> Optional[dist.ProcessGroup]:
    return _ep_group


def get_ep_world_size() -> int:
    return _ep_world_size


def get_ep_rank() -> int:
    return _ep_rank


# ── Autograd-compatible all-to-all ─────────────────────────────────────────

class _AllToAll(torch.autograd.Function):
    """All-to-all with proper backward (transpose all-to-all).

    Forward:  각 rank가 W개 chunk을 보내고 W개 chunk을 받음
    Backward: forward의 역방향 = 다시 all-to-all (chunk 순서 반전)

    예시 (2 GPU, 8 experts):
      Forward dispatch:
        GPU 0이 expert 4,5,6,7로 보낼 토큰 → GPU 1로 전송
        GPU 1이 expert 0,1,2,3로 보낼 토큰 → GPU 0으로 전송
      Backward:
        gradient가 역방향으로 동일한 all-to-all을 수행
        → 각 GPU가 자기 local expert의 올바른 gradient를 받음
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, ep_group: dist.ProcessGroup,
                world_size: int):
        ctx.ep_group = ep_group
        ctx.world_size = world_size

        # 🚀 Fix 1: NCCL 통신 전 확실한 contiguous 보장 및 zeros_like 사용
        input = input.contiguous()
        input_list = [chunk.contiguous() for chunk in input.chunk(world_size, dim=0)]
        output_list = [torch.zeros_like(chunk) for chunk in input_list]

        dist.all_to_all(output_list, input_list, group=ep_group)
        return torch.cat(output_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        grad_input_list = [chunk.contiguous() for chunk in grad_output.chunk(ctx.world_size, dim=0)]
        grad_output_list = [torch.zeros_like(chunk) for chunk in grad_input_list]

        # barrier: forward alltoall과 동일하게 backward alltoall도 모든 rank가
        # 같은 시점에 통신에 진입하도록 보장. 없으면 NCCL SeqNum 불일치 데드락 가능.
        dist.barrier(group=ctx.ep_group)
        dist.all_to_all(grad_output_list, grad_input_list, group=ctx.ep_group)
        dist.barrier(group=ctx.ep_group)
        return torch.cat(grad_output_list, dim=0), None, None


def all_to_all(input: torch.Tensor) -> torch.Tensor:
    """Autograd-compatible all-to-all over the EP group.

    Args:
        input: (W, ...) tensor where dim-0 is split across W ranks.

    Returns:
        (W, ...) tensor after exchange.
    """
    if _ep_group is None or _ep_world_size <= 1:
        return input
    # 모든 rank가 이 지점에 도달했는지 확인한 뒤 all-to-all 실행.
    # 없으면 한쪽이 아직 forward/backward 중인데 다른 쪽이 이미
    # sync_shared_gradients의 all_reduce를 호출해 통신 순서가 꼬임.
    dist.barrier(group=_ep_group)
    output = _AllToAll.apply(input, _ep_group, _ep_world_size)
    # all-to-all 완료 후에도 barrier로 모든 rank 동기화 (다음 통신과 순서 꼬임 방지)
    dist.barrier(group=_ep_group)
    return output


# ── Checkpoint: gather all experts to rank 0 ─────────────────────────────

def gather_expert_state_dict(model: torch.nn.Module) -> dict:
    """모든 rank의 local expert weight를 rank 0으로 gather하여 full state_dict 반환.

    EP 학습 시 각 GPU는 num_experts // W 개의 local expert만 보유.
    Checkpoint 저장 전 이 함수를 호출하면:
      - rank 0: 전체 expert(8개)가 포함된 state_dict 반환
      - rank != 0: 빈 dict 반환 (save_on_master에서 무시됨)

    예시 (2 GPU, 8 experts):
      rank 0: experts.w1 shape (4, 2048, 256) → gather → (8, 2048, 256)
      rank 1: experts.w1 shape (4, 2048, 256) → rank 0으로 전송

    EP가 비활성화된 경우(single GPU): 그냥 model.state_dict() 반환.
    """
    state_dict = model.state_dict()

    if _ep_group is None or _ep_world_size <= 1:
        return state_dict

    full_state_dict = {}
    for name, tensor in state_dict.items():
        if is_expert_parameter(name):
            # 모든 rank의 local expert tensor를 rank 0으로 gather
            gathered = [torch.zeros_like(tensor) for _ in range(_ep_world_size)]
            dist.all_gather(gathered, tensor.contiguous(), group=_ep_group)
            if _ep_rank == 0:
                # dim=0이 expert 차원: (local_E, ...) → cat → (global_E, ...)
                full_state_dict[name] = torch.cat(gathered, dim=0).cpu()
        else:
            # shared parameter: 모든 rank에서 동일하므로 그대로 사용
            if _ep_rank == 0:
                full_state_dict[name] = tensor.cpu()

    # rank != 0은 빈 dict (save_on_master에서 저장 안 됨)
    if _ep_rank != 0:
        return {}

    return full_state_dict


def scatter_expert_state_dict(full_state_dict: dict, model: torch.nn.Module) -> dict:
    """Full state_dict에서 현재 rank의 local expert만 추출하여 반환.

    gather_expert_state_dict로 저장된 checkpoint를 EP eval에서 로드할 때 사용.
    전체 expert(8개)가 포함된 state_dict에서 현재 rank에 해당하는
    local expert(4개)만 slice하여 model.load_state_dict()에 전달.

    예시 (2 GPU, 8 experts):
      full: experts.w1 shape (8, 2048, 256)
      rank 0 → (0:4, ...) = experts 0-3
      rank 1 → (4:8, ...) = experts 4-7

    EP가 비활성화된 경우: full_state_dict 그대로 반환.
    """
    if _ep_group is None or _ep_world_size <= 1:
        return full_state_dict

    # model의 state_dict로 local expert 크기 파악
    local_sd = model.state_dict()
    scattered = {}

    for name, full_tensor in full_state_dict.items():
        if is_expert_parameter(name):
            local_num_experts = local_sd[name].shape[0]
            start = _ep_rank * local_num_experts
            end = start + local_num_experts
            scattered[name] = full_tensor[start:end]
        else:
            scattered[name] = full_tensor

    return scattered


# ── Gradient sync for EP training ─────────────────────────────────────────

def is_expert_parameter(name: str) -> bool:
    """Check if a parameter name belongs to an expert module.

    Expert 파라미터 식별 규칙:
    - DINOMoEBlock의 experts (DINOBatchedExperts): 'moe_block.experts.'
    - SparseMoELayer의 experts: 'moe_layer.experts.'

    이 파라미터들은 각 GPU에 서로 다른 expert가 있으므로
    all-reduce하면 안 됨.
    """
    return '.experts.' in name


def sync_shared_gradients(model: torch.nn.Module,
                          group: Optional[dist.ProcessGroup] = None):
    """All-reduce gradients of shared (non-expert) parameters only.

    ┌─────────────────────────────────────────────────────────────┐
    │  왜 DDP 대신 이것을 써야 하는가?                              │
    │                                                             │
    │  DDP: 모든 파라미터의 gradient를 all-reduce                  │
    │  → Expert 파라미터도 all-reduce됨                            │
    │  → GPU 0의 expert 0 gradient와 GPU 1의 expert 4 gradient가  │
    │    평균됨 → 완전히 잘못된 update!                             │
    │                                                             │
    │  sync_shared_gradients:                                     │
    │  - Expert 파라미터: all-to-all backward가 이미 올바른        │
    │    gradient를 계산함. 건드리지 않음.                          │
    │  - Shared 파라미터 (router, attention, LayerNorm):           │
    │    모든 GPU에 동일하게 복제되어 있으므로                       │
    │    gradient를 all-reduce(평균)해서 동기화                     │
    └─────────────────────────────────────────────────────────────┘

    Usage:
        loss.backward()
        sync_shared_gradients(model)  # DDP 대신
        optimizer.step()
    """
    if group is None:
        group = _ep_group
    if group is None or _ep_world_size <= 1:
        return

    # backward의 MoE all-to-all이 모든 rank에서 완료될 때까지 대기.
    # 없으면 한쪽 rank가 아직 backward all-to-all 중인데
    # 다른 rank가 여기서 all_reduce를 호출하여 NCCL SeqNum 불일치 → 데드락.
    dist.barrier(group=group)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if is_expert_parameter(name):
            # all-to-all backward는 W개 GPU의 gradient를 SUM으로 합산.
            # Shared params는 all_reduce(AVG)로 1/W 스케일이므로,
            # Expert도 W로 나눠 동일한 스케일로 맞춘다.
            param.grad.div_(get_ep_world_size())
        else:
            # 🚀 Fix 2: all_reduce 전에 그래디언트가 불연속적이면 강제로 펴준다! (NCCL 쓰레기값 방지)
            if not param.grad.is_contiguous():
                param.grad.data = param.grad.data.contiguous()
            # Shared parameter: 모든 GPU에서 동일해야 하므로 gradient 평균.
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=group)


def ep_clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """EP 환경에서 모든 GPU가 동일한 global norm으로 gradient clipping.

    문제: 기본 clip_grad_norm_은 GPU-local norm 사용 → 각 GPU의 local expert가
    다르므로 norm이 달라짐 → shared params에 다른 clip factor 적용 → 동기화 깨짐.

    해결: expert_norm²를 all_reduce(SUM)으로 전체 합산, shared_norm²는
    sync_shared_gradients 이후 이미 동일하므로, 합산하면 모든 GPU에서
    동일한 total_norm이 산출됨.

    Args:
        model: gradient를 clipping할 모델.
        max_norm: 최대 허용 gradient norm.
        norm_type: norm 타입 (default: 2.0 = L2 norm).
        group: EP process group (default: module-level _ep_group).

    Returns:
        Clipping 전 total gradient norm.
    """
    if group is None:
        group = _ep_group

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = next(model.parameters()).device
    shared_norm_sq = torch.tensor(0.0, device=device)
    expert_norm_sq = torch.tensor(0.0, device=device)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        param_norm_sq = param.grad.detach().norm(norm_type) ** norm_type
        if is_expert_parameter(name):
            expert_norm_sq += param_norm_sq
        else:
            shared_norm_sq += param_norm_sq

    # 각 GPU의 local expert norm을 합산 → 전역 expert norm
    if group is not None and _ep_world_size > 1:
        dist.all_reduce(expert_norm_sq, op=dist.ReduceOp.SUM, group=group)

    total_norm = (shared_norm_sq + expert_norm_sq) ** (1.0 / norm_type)

    # 🚀 Fix 3: 혹시라도 total_norm이 NaN/Inf라면 곱하기(clip)를 건너뛰도록 PyTorch Native 방어막 추가
    if torch.isinf(total_norm) or torch.isnan(total_norm):
        return total_norm

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach().mul_(clip_coef_clamped)

    return total_norm
