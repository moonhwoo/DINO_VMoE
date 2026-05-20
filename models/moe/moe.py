"""Core Sparse MoE utils for PyTorch.

The following abbreviations are sometimes used to name the size of different
axes in the arrays.

G = num_groups. It must be a multiple of num_experts.
S = group_size.
E = num_experts.
C = capacity.
K = num_selected_experts. It must be <= num_experts.
"""
import abc
import math
import logging
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe import expert_parallel as ep

CeilOrRound = Literal["ceil", "round"]


# ============================================================================
# Base Dispatcher
# ============================================================================

class BaseDispatcher(abc.ABC):
    """Base class for different dispatcher implementations.

    Dispatchers are in charge of preparing the data to be dispatched to the
    different experts, and then combining the outputs of each expert for each
    item. There are different ways of doing so with different memory / flops /
    runtime implications.

    In all cases, when dispatching data, they take a tensor of shape (G, S, ...).
    The groups (G) are dispatched independently of each other. The items in each
    group (S) will take place in the buffer (of capacity C) of items to be
    processed by each expert (E). The output is a tensor of shape (E, G*C, ...)
    with the elements to be processed by each expert.

    When combining data, they take a tensor of shape (E, G*C, ...) and output
    a tensor of shape (G, S, ...).
    """

    @abc.abstractmethod
    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        """Dispatches data to experts.

        Args:
          data: (G, S, ...) tensor with the data to dispatch to the experts.

        Returns:
          (E, G*C, ...) tensor with the data to be processed by each expert.
        """

    @abc.abstractmethod
    def combine(self, data: torch.Tensor) -> torch.Tensor:
        """Combines outputs from multiple experts.

        Args:
          data: (E, G*C, ...) tensor with the output data from each expert.

        Returns:
          (G, S, ...) tensor with the combined outputs from each expert.
        """


# ============================================================================
# DenseEinsumDispatcher
# ============================================================================

class DenseEinsumDispatcher(BaseDispatcher):
    """Dispatcher using Einsum, dispatching data to all experts.

    This is similar to EinsumDispatcher, but with the assumption that C = S.

    Attributes:
      combine_weights: (G, S, E) tensor with the combine weights.
      expert_parallel: Whether to use expert parallelism.
    """

    def __init__(self, combine_weights: torch.Tensor,
                 expert_parallel: bool = False):
        self.combine_weights = combine_weights
        self.expert_parallel = expert_parallel

    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        dispatch_weights = torch.ones_like(self.combine_weights, dtype=torch.bool)
        data = torch.einsum("GSE,GS...->GES...", dispatch_weights.float(), data)
        return _dispatch(data, expert_parallel=self.expert_parallel)

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        num_groups = self.combine_weights.shape[0]
        data = _receive(data, num_groups, expert_parallel=self.expert_parallel)
        return torch.einsum("GSE,GES...->GS...", self.combine_weights, data)


# ============================================================================
# EinsumDispatcher
# ============================================================================

class EinsumDispatcher(BaseDispatcher):
    """Dispatcher using Einsum.

    Attributes:
      combine_weights: (G, S, E, C) tensor with the combine weights.
      dispatch_weights: Optional (G, S, E, C) tensor with the dispatch weights.
      expert_parallel: Whether to use expert parallelism.
    """

    def __init__(self, combine_weights: torch.Tensor,
                 dispatch_weights: Optional[torch.Tensor] = None,
                 expert_parallel: bool = False):
        self.combine_weights = combine_weights
        self.dispatch_weights = dispatch_weights
        self.expert_parallel = expert_parallel

    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        dw = (self.combine_weights > 0).float() if self.dispatch_weights is None \
            else self.dispatch_weights.float()
        data = torch.einsum("GSEC,GS...->GEC...", dw, data)
        return _dispatch(data, expert_parallel=self.expert_parallel)

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        num_groups = self.combine_weights.shape[0]
        data = _receive(data, num_groups, expert_parallel=self.expert_parallel)
        return torch.einsum("GSEC,GEC...->GS...", self.combine_weights, data)


# ============================================================================
# ExpertIndicesDispatcher
# ============================================================================

class ExpertIndicesDispatcher(BaseDispatcher):
    """Dispatcher using scatter/gather with (expert, buffer) indices.

    Attributes:
      indices: (G, S, K, 2) int tensor with (expert, buffer) indices.
      combine_weights: (G, S, K) tensor with combine weights.
      num_experts: Number of experts (global, not per-rank).
      capacity: Capacity of each expert's buffer per group.
      expert_parallel: Whether to use expert parallelism.
    """

    def __init__(self, indices: torch.Tensor, combine_weights: torch.Tensor,
                 num_experts: int, capacity: int,
                 expert_parallel: bool = False):
        self.indices = indices
        self.combine_weights = combine_weights
        self.num_experts = num_experts
        self.capacity = capacity
        self.expert_parallel = expert_parallel

    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        num_groups, _, num_selected_experts, _ = self.indices.shape
        _, _, *item_shape = data.shape
        data = data.repeat_interleave(num_selected_experts, dim=1)
        #data (G,S*k,D) k개 갯수로 복사함   즉 1지망용,2지망용
        indices = self.indices.reshape(num_groups, -1, 2)
        #(G, S, K, 2) -> (G, S*K, 2)

        expert_idx = indices[:, :, 0] #몇번 전문가한테 가는지
        buffer_idx = indices[:, :, 1] #그 전문가의 몇번째 buffer인지
        valid = ((expert_idx >= 0) & (expert_idx < self.num_experts) &
                 (buffer_idx >= 0) & (buffer_idx < self.capacity))

        g_idx = torch.arange(num_groups, device=data.device
                             ).unsqueeze(1).expand_as(expert_idx)

        # scatter는 전체 E experts 기준으로 수행 (routing은 global)
        result = torch.zeros(num_groups, self.num_experts, self.capacity,
                             *item_shape, dtype=data.dtype, device=data.device)
        result.index_put_((g_idx[valid], expert_idx[valid], buffer_idx[valid]),
                          data[valid], accumulate=True)
        # _dispatch가 EP일 때 all-to-all로 local experts에 해당하는 토큰만 남김
        return _dispatch(result, expert_parallel=self.expert_parallel)

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        num_groups = self.combine_weights.shape[0]
        # _receive가 EP일 때 all-to-all로 결과를 원래 rank에 돌려줌
        data = _receive(data, num_groups,
                        expert_parallel=self.expert_parallel)  # (G, E, C, ...)

        expert_idx = self.indices[..., 0].clamp(0, self.num_experts - 1)
        buffer_idx = self.indices[..., 1].clamp(0, self.capacity - 1)
        g_idx = torch.arange(num_groups, device=data.device).reshape(-1, 1, 1)

        gathered = data[g_idx, expert_idx, buffer_idx]

        mask = ((self.indices[..., 0] < self.num_experts) &
                (self.indices[..., 1] < self.capacity))
        extra_dims = gathered.ndim - 3
        gathered = gathered * mask.reshape(*mask.shape, *([1] * extra_dims)).float()

        return torch.einsum("GSK...,GSK->GS...", gathered, self.combine_weights)


# ============================================================================
# Bfloat16Dispatcher
# ============================================================================

class Bfloat16Dispatcher(BaseDispatcher):
    """Dispatcher wrapper converting data to bfloat16 to save bandwidth."""

    def __init__(self, dispatcher: BaseDispatcher):
        self.dispatcher = dispatcher

    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        dtype = data.dtype
        if data.is_floating_point():
            data = data.to(torch.bfloat16)
        data = self.dispatcher.dispatch(data)
        return data.to(dtype)

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        dtype = data.dtype
        if data.is_floating_point():
            data = data.to(torch.bfloat16)
        data = self.dispatcher.combine(data)
        return data.to(dtype)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_capacity(
        num_tokens: int,
        num_experts: int,
        capacity_factor: float,
        ceil_or_round: CeilOrRound = "ceil",
        multiple_of: Optional[int] = 4) -> int:
    """Returns the capacity per expert needed to distribute num_tokens among num_experts."""
    if ceil_or_round == "ceil":
        capacity = int(math.ceil(num_tokens * capacity_factor / num_experts))
    elif ceil_or_round == "round":
        capacity = int(round(num_tokens * capacity_factor / num_experts))
    else:
        raise ValueError(f"Unsupported {ceil_or_round=}")
    if capacity < 1:
        raise ValueError(
            f"The values num_tokens = {num_tokens}, num_experts = "
            f"{num_experts} and capacity_factor = {capacity_factor} "
            f"lead to capacity = {capacity}, but it must be >= 1.")
    if multiple_of and multiple_of > 0:
        capacity += (-capacity) % multiple_of
    actual_capacity_factor = capacity * num_experts / num_tokens
    if abs(actual_capacity_factor - capacity_factor) > 1e-1:
        logging.warning(
            "The target capacity_factor is %f, but with num_tokens=%d and "
            "num_experts=%d the actual capacity_factor is %f.",
            capacity_factor, num_tokens, num_experts, actual_capacity_factor)
    return capacity


def _dispatch(data: torch.Tensor, expert_parallel: bool = False) -> torch.Tensor:
    """Dispatches data to experts via reshape+transpose (+ all-to-all for EP).

    Input:  (G, E, C, ...) -> Output: (E_local, G_total*C, ...)

    When expert_parallel=False (default): E_local = E, G_total = G (기존 동작).
    When expert_parallel=True:
      - E를 (W, E_local)로 분할 (W = EP world size)
      - all-to-all로 각 rank의 토큰을 해당 local experts로 전송
      - E_local = E // W, G_total = G * W
    """
    num_groups, num_experts, capacity, *item_shape = data.shape

    if expert_parallel and ep.get_ep_world_size() > 1:
        W = ep.get_ep_world_size()
        E_local = num_experts // W
        assert num_experts % W == 0, (
            f"num_experts ({num_experts}) must be divisible by "
            f"ep_world_size ({W})")

        # (G, E, C, ...) → (G, W, E_local, C, ...) → (W, G, E_local, C, ...)
        data = data.reshape(num_groups, W, E_local, capacity, *item_shape)
        data = data.permute(1, 0, 2, 3, *range(4, 4 + len(item_shape)))
        # data is now (W, G, E_local, C, ...): chunk[w] = 이 rank→rank w 데이터
        data = data.contiguous()

        # all-to-all: 각 rank가 자기 local experts에 해당하는 토큰을 받음
        # After: (W, G, E_local, C, ...) where chunk[w] = rank w→이 rank 데이터
        data = ep.all_to_all(data)

        # (W, G, E_local, C, ...) → (E_local, W*G*C, ...)
        data = data.permute(2, 0, 1, 3, *range(4, 4 + len(item_shape)))
        return data.reshape(E_local, W * num_groups * capacity, *item_shape)
    else:
        # 기존 동작 (단일 GPU 또는 DDP)
        if num_groups % num_experts == 0:
            data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
            data = data.permute(2, 1, 0, 3, *range(4, 4 + len(item_shape)))
        else:
            data = data.permute(1, 0, 2, *range(3, 3 + len(item_shape)))
        return data.reshape(num_experts, num_groups * capacity, *item_shape)


def _receive(data: torch.Tensor, num_groups: int,
             expert_parallel: bool = False) -> torch.Tensor:
    """Receives data from experts via reshape+transpose (+ all-to-all for EP).

    Input:  (E_local, G_total*C, ...) -> Output: (G, E, C, ...)

    When expert_parallel=True: E_local < E, G_total = G * W.
    Performs all-to-all to return results to originating ranks.
    """
    if expert_parallel and ep.get_ep_world_size() > 1:
        W = ep.get_ep_world_size()
        E_local = data.shape[0]
        num_experts = E_local * W
        total_tokens = data.shape[1]
        capacity = total_tokens // (W * num_groups)
        item_shape = list(data.shape[2:])

        # (E_local, W*G*C, ...) → (E_local, W, G, C, ...) → (W, G, E_local, C, ...)
        data = data.reshape(E_local, W, num_groups, capacity, *item_shape)
        data = data.permute(1, 2, 0, 3, *range(4, 4 + len(item_shape)))
        data = data.contiguous()

        # all-to-all (역방향): 결과를 원래 rank로 반환
        data = ep.all_to_all(data)

        # (W, G, E_local, C, ...) → (G, W, E_local, C, ...) → (G, E, C, ...)
        data = data.permute(1, 0, 2, 3, *range(4, 4 + len(item_shape)))
        return data.reshape(num_groups, num_experts, capacity, *item_shape)
    else:
        # 기존 동작
        num_experts, num_groups_times_capacity, *item_shape = data.shape
        capacity = num_groups_times_capacity // num_groups
        data = data.reshape(num_experts, num_groups, capacity, *item_shape)
        if num_groups % num_experts == 0:
            data = data.reshape(
                num_experts, num_experts, -1, capacity, *item_shape)
            data = data.permute(2, 1, 0, 3, *range(4, 4 + len(item_shape)))
            data = data.reshape(num_groups, num_experts, capacity, *item_shape)
        else:
            data = data.permute(1, 0, 2, *range(3, 3 + len(item_shape)))
        return data


def _scatter_nd(indices: torch.Tensor, updates: torch.Tensor,
                shape: tuple) -> torch.Tensor:
    """PyTorch implementation of scatter_nd (analogous to tf.scatter_nd).

    Args:
      indices: (B, ndim) int tensor of indices.
      updates: (B, ...) tensor of values.
      shape: Output shape (ndim-dimensional).

    Returns:
      Tensor of `shape` with accumulated values at given indices.
    """
    result = torch.zeros(shape, dtype=updates.dtype, device=updates.device)
    # Clamp indices to valid range to avoid out-of-bounds errors.
    # Out-of-range indices contribute zeros (matching JAX behavior).
    idx_list = []
    valid_mask = torch.ones(indices.shape[0], dtype=torch.bool,
                            device=indices.device)
    for dim in range(indices.shape[1]):
        idx = indices[:, dim]
        valid_mask = valid_mask & (idx >= 0) & (idx < shape[dim])
        idx_list.append(idx)

    if valid_mask.any():
        valid_indices = tuple(idx[valid_mask] for idx in idx_list)
        valid_updates = updates[valid_mask]
        result.index_put_(valid_indices, valid_updates, accumulate=True)
    return result


# ============================================================================
# Top-Experts-Per-Item Routing
# ============================================================================

def get_dense_einsum_dispatcher(gates: torch.Tensor,
                                **kwargs) -> DenseEinsumDispatcher:
    """Returns a DenseEinsumDispatcher (all tokens to all experts)."""
    return DenseEinsumDispatcher(combine_weights=gates)


def get_top_experts_per_item_dispatcher(
        gates: torch.Tensor,
        name: str,
        num_selected_experts: int,
        batch_priority: bool,
        capacity: Optional[int] = None,
        capacity_factor: Optional[float] = None,
        capacity_ceil_or_round: CeilOrRound = "ceil",
        capacity_multiple_of: Optional[int] = 4,
        **dispatcher_kwargs) -> BaseDispatcher:
    """Returns a dispatcher implementing Top-Experts-Per-Item routing.

    For each item, the `num_selected_experts` experts with the largest gating
    score are selected. Each expert has a fixed `capacity`.

    Args:
      gates: (S, E) tensor with the gating values for each (item, expert).
      name: Type of dispatcher ("einsum" or "indices").
      num_selected_experts: Maximum number of experts to select per item (K).
      batch_priority: Whether to use batch priority routing.
      capacity: Maximum items per expert. Either this or capacity_factor.
      capacity_factor: Sets capacity relative to S * K / E.
      capacity_ceil_or_round: How to compute capacity ("ceil" or "round").
      capacity_multiple_of: Make capacity a multiple of this.

    Returns:
      A dispatcher.
    """
    if (capacity is None) == (capacity_factor is None):
        raise ValueError(
            "You must specify either 'capacity' or 'capacity_factor', not both."
            f" Current values: capacity={capacity!r}, "
            f"capacity_factor={capacity_factor!r}")
    if not capacity:
        group_size, num_experts = gates.shape
        capacity = compute_capacity(
            num_tokens=group_size * num_selected_experts,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            ceil_or_round=capacity_ceil_or_round,
            multiple_of=capacity_multiple_of)

    fn_map = {
        "einsum": _get_top_experts_per_item_einsum_dispatcher,
        "indices": _get_top_experts_per_item_expert_indices_dispatcher,
    }
    if name not in fn_map:
        raise ValueError(f"Unknown dispatcher type: {name!r}")
    return fn_map[name](gates, num_selected_experts, capacity, batch_priority,
                        **dispatcher_kwargs)


def get_top_items_per_expert_dispatcher(
        gates: torch.Tensor,
        name: str,
        capacity: Optional[int] = None,
        capacity_factor: Optional[float] = None,
        capacity_ceil_or_round: CeilOrRound = "ceil",
        capacity_multiple_of: Optional[int] = 4,
        **dispatcher_kwargs) -> Tuple[BaseDispatcher, Dict[str, torch.Tensor]]:
    """Returns a dispatcher implementing Top-Items-Per-Expert (Expert Choice).

    For each expert, the top `capacity` items with the largest gating score
    are selected. This ensures perfectly balanced load.

    Args:
      gates: (S, E) tensor with gating values.
      name: Type of dispatcher ("einsum").
      capacity: Maximum items per expert.
      capacity_factor: Sets capacity relative to S / E.
      capacity_ceil_or_round: How to compute capacity.
      capacity_multiple_of: Make capacity a multiple of this.

    Returns:
      A (dispatcher, metrics) tuple.
    """
    if (capacity is None) == (capacity_factor is None):
        raise ValueError(
            "You must specify either 'capacity' or 'capacity_factor', not both.")
    if not capacity:
        group_size, num_experts = gates.shape
        capacity = compute_capacity(
            num_tokens=group_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            ceil_or_round=capacity_ceil_or_round,
            multiple_of=capacity_multiple_of)

    fn_map = {
        "einsum": _get_top_items_per_expert_einsum_dispatcher,
    }
    if name not in fn_map:
        raise ValueError(f"Unknown dispatcher type: {name!r}")
    return fn_map[name](gates, capacity, **dispatcher_kwargs)


def _get_top_experts_per_item_common(
        gates: torch.Tensor, num_selected_experts: int,
        batch_priority: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns common arrays used by Top-Experts-Per-Item routing.

    Args:
      gates: (S, E) tensor with gating values.
      num_selected_experts: Maximum number of experts to select per item.
      batch_priority: Whether to use batch priority routing.

    Returns:
      - combine_weights: (S, K)
      - expert_index: (S, K)
      - buffer_index: (S, K, E)
    """
    group_size, num_experts = gates.shape
    combine_weights, expert_index = torch.topk(gates, num_selected_experts, dim=-1)

    if batch_priority:
        perm = torch.argsort(-combine_weights[:, 0])
        expert_index = expert_index[perm]

    # (S, K) -> (K, S) -> flatten to (K*S,)
    expert_index_flat = expert_index.t().reshape(-1)

    # Convert to one-hot: (K*S, E)
    expert_one_hot = F.one_hot(expert_index_flat, num_experts).int()

    # Cumulative buffer index within each expert's buffer
    buffer_index = torch.cumsum(expert_one_hot, dim=0) * expert_one_hot - 1
    buffer_index = buffer_index.reshape(-1, group_size, num_experts)  # (K, S, E)
    buffer_index = buffer_index.permute(1, 0, 2)  # (S, K, E)

    # Revert expert_index to original shape
    expert_index = expert_index_flat.reshape(-1, group_size).t()  # (S, K)

    if batch_priority:
        inv_perm = torch.argsort(perm)
        expert_index = expert_index[inv_perm]
        buffer_index = buffer_index[inv_perm]

    return combine_weights, expert_index, buffer_index


def _get_top_experts_per_item_common_batched(
        gates: torch.Tensor, num_selected_experts: int,
        batch_priority: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched version of _get_top_experts_per_item_common.

    Processes all groups in parallel using tensor operations instead of a
    Python for-loop (equivalent to jax.vmap over the group dimension).

    Args:
      gates: (G, S, E) tensor with gating values.
      num_selected_experts: Maximum number of experts to select per item.
      batch_priority: Whether to use batch priority routing.

    Returns:
      - combine_weights: (G, S, K)
      - expert_index: (G, S, K)
      - buffer_index: (G, S, K, E)
    """
    G, group_size, num_experts = gates.shape
    # Top-K over expert dimension: (G, S, K)
    combine_weights, expert_index = torch.topk(gates, num_selected_experts, dim=-1)

    if batch_priority:
        # Sort items by their top-1 gate value within each group
        perm = torch.argsort(-combine_weights[:, :, 0], dim=1)  # (G, S) 
        #perm=combine weight에서 전문가 관점이 아닌 그룹내에서 가장 큰 값 순으로  인덱스 정렬
        #즉 sequence차원에서 누가 제일 높은 지망 점수를 가졌나 토큰끼리
        expert_index = torch.gather(
            expert_index, 1, perm.unsqueeze(-1).expand_as(expert_index))
        #perm.unsqueeze(-1).expand_as(expert_index)는 이제 perm이 그룹안에서 1등 토큰만 했으므로 1,2등 토큰도 보게
        #gather을 통해서 높은순으로 top k쌍 만들음

    # (G, S, K) -> (G, K, S) -> (G, K*S): make K the leading axis so top-1
    # choices get priority over top-2 etc.
    expert_index_flat = expert_index.transpose(1, 2).reshape(G, -1)  # (G, K*S)

    # One-hot: (G, K*S, E)
    expert_one_hot = F.one_hot(expert_index_flat, num_experts).int()
    #각자 expert에 대한 투표용지 가짐(one-hot)

    # Cumsum along K*S axis (dim=1) — each group is independent
    buffer_index = torch.cumsum(expert_one_hot, dim=1) * expert_one_hot - 1
    #곱한후는 여전히 (G, K*S, E) 
    # (G, K*S, E) -> (G, K, S, E) -> (G, S, K, E)
    buffer_index = buffer_index.reshape(G, -1, group_size, num_experts)
    buffer_index = buffer_index.permute(0, 2, 1, 3)

    # Restore expert_index: (G, K*S) -> (G, K, S) -> (G, S, K)
    expert_index = expert_index_flat.reshape(G, -1, group_size).transpose(1, 2)

    if batch_priority:
        inv_perm = torch.argsort(perm, dim=1)  # (G, S)
        expert_index = torch.gather(
            expert_index, 1,
            inv_perm.unsqueeze(-1).expand_as(expert_index))
        buffer_index = torch.gather(
            buffer_index, 1,
            inv_perm.unsqueeze(-1).unsqueeze(-1).expand_as(buffer_index))

    return combine_weights, expert_index, buffer_index


def _get_top_experts_per_item_expert_indices_dispatcher_batched(
        gates: torch.Tensor, num_selected_experts: int, capacity: int,
        batch_priority: bool, **dispatcher_kwargs) -> ExpertIndicesDispatcher:
    """Batched version: returns ExpertIndicesDispatcher from (G, S, E) gates."""
    expert_parallel = dispatcher_kwargs.pop("expert_parallel", False)
    G, _, num_experts = gates.shape
    combine_weights, expert_idx, buffer_idx = _get_top_experts_per_item_common_batched(
        gates, num_selected_experts, batch_priority)
    buffer_idx, _ = buffer_idx.max(dim=3)
    return ExpertIndicesDispatcher(
        indices=torch.stack([expert_idx, buffer_idx], dim=-1),
        combine_weights=combine_weights,
        num_experts=num_experts,
        capacity=capacity,
        expert_parallel=expert_parallel)


def _get_top_experts_per_item_einsum_dispatcher_batched(
        gates: torch.Tensor, num_selected_experts: int, capacity: int,
        batch_priority: bool, **dispatcher_kwargs) -> EinsumDispatcher:
    """Batched version: returns EinsumDispatcher from (G, S, E) gates."""
    expert_parallel = dispatcher_kwargs.pop("expert_parallel", False)
    G, _, num_experts = gates.shape
    _, _, buffer_idx = _get_top_experts_per_item_common_batched(
        gates, num_selected_experts, batch_priority)
    buffer_idx, _ = buffer_idx.max(dim=2)

    valid = (buffer_idx >= 0) & (buffer_idx < capacity)
    buffer_idx_clamped = buffer_idx.clamp(0, capacity - 1)
    dispatch_weights = F.one_hot(buffer_idx_clamped, capacity).bool()
    dispatch_weights = dispatch_weights & valid.unsqueeze(-1)

    combine_weights = torch.einsum(
        "GSE,GSEC->GSEC", gates, dispatch_weights.float())

    return EinsumDispatcher(
        combine_weights=combine_weights,
        dispatch_weights=dispatch_weights.float(),
        expert_parallel=expert_parallel)


def get_top_experts_per_item_dispatcher_batched(
        gates: torch.Tensor,
        name: str,
        num_selected_experts: int,
        batch_priority: bool,
        capacity: Optional[int] = None,
        capacity_factor: Optional[float] = None,
        capacity_ceil_or_round: CeilOrRound = "ceil",
        capacity_multiple_of: Optional[int] = 4,
        **dispatcher_kwargs) -> "BaseDispatcher":
    """Batched dispatcher creation from (G, S, E) gates — no Python loops."""
    if (capacity is None) == (capacity_factor is None):
        raise ValueError(
            "You must specify either 'capacity' or 'capacity_factor', not both."
            f" Current values: capacity={capacity!r}, "
            f"capacity_factor={capacity_factor!r}")
    if not capacity:
        group_size, num_experts = gates.shape[1], gates.shape[2]
        capacity = compute_capacity(
            num_tokens=group_size * num_selected_experts,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            ceil_or_round=capacity_ceil_or_round,
            multiple_of=capacity_multiple_of)

    fn_map = {
        "einsum": _get_top_experts_per_item_einsum_dispatcher_batched,
        "indices": _get_top_experts_per_item_expert_indices_dispatcher_batched,
    } #위에서 입력받은 name에 따라 결정이됌
    if name not in fn_map:
        raise ValueError(f"Unknown dispatcher type: {name!r}")
    return fn_map[name](gates, num_selected_experts, capacity, batch_priority,
                        **dispatcher_kwargs)


def _get_top_experts_per_item_einsum_dispatcher(
        gates: torch.Tensor, num_selected_experts: int, capacity: int,
        batch_priority: bool, **dispatcher_kwargs) -> EinsumDispatcher:
    """Returns an EinsumDispatcher performing Top-Experts-Per-Item routing."""
    expert_parallel = dispatcher_kwargs.pop("expert_parallel", False)
    _, _, buffer_idx = _get_top_experts_per_item_common(
        gates, num_selected_experts, batch_priority)
    buffer_idx, _ = buffer_idx.max(dim=1)

    valid = (buffer_idx >= 0) & (buffer_idx < capacity)
    buffer_idx_clamped = buffer_idx.clamp(0, capacity - 1)
    dispatch_weights = F.one_hot(buffer_idx_clamped, capacity).bool()
    dispatch_weights = dispatch_weights & valid.unsqueeze(-1)

    combine_weights = torch.einsum(
        "SE,SEC->SEC", gates, dispatch_weights.float())

    return EinsumDispatcher(
        combine_weights=combine_weights,
        dispatch_weights=dispatch_weights.float(),
        expert_parallel=expert_parallel)


def _get_top_experts_per_item_expert_indices_dispatcher(
        gates: torch.Tensor, num_selected_experts: int, capacity: int,
        batch_priority: bool, **dispatcher_kwargs) -> ExpertIndicesDispatcher:
    """Returns an ExpertIndicesDispatcher performing Top-Experts-Per-Item routing."""
    expert_parallel = dispatcher_kwargs.pop("expert_parallel", False)
    _, num_experts = gates.shape
    combine_weights, expert_idx, buffer_idx = _get_top_experts_per_item_common(
        gates, num_selected_experts, batch_priority)
    buffer_idx, _ = buffer_idx.max(dim=2)
    return ExpertIndicesDispatcher(
        indices=torch.stack([expert_idx, buffer_idx], dim=-1),
        combine_weights=combine_weights,
        num_experts=num_experts,
        capacity=capacity,
        expert_parallel=expert_parallel)


def _get_top_items_per_expert_einsum_dispatcher(
        gates: torch.Tensor, capacity: int,
        **dispatcher_kwargs) -> Tuple[EinsumDispatcher, Dict[str, torch.Tensor]]:
    """Returns an EinsumDispatcher performing Top-Items-Per-Expert routing."""
    expert_parallel = dispatcher_kwargs.pop("expert_parallel", False)
    group_size, num_experts = gates.shape

    top_items_gates, top_items_index = torch.topk(gates.t(), capacity, dim=-1)

    dispatch_weights = F.one_hot(top_items_index, group_size)
    dispatch_weights = dispatch_weights.permute(2, 0, 1).bool()

    combine_weights = torch.einsum(
        "SE,SEC->SEC", gates, dispatch_weights.float())

    dispatcher = EinsumDispatcher(
        dispatch_weights=dispatch_weights.float(),
        combine_weights=combine_weights,
        expert_parallel=expert_parallel)

    # Compute monitoring metrics
    num_experts_per_item = dispatch_weights.sum(dim=(1, 2)).int()
    metrics = {
        "num_experts_per_item_min": num_experts_per_item.min(),
        "num_experts_per_item_max": num_experts_per_item.max(),
        "min_selected_gate": top_items_gates.min(),
        "max_selected_gate": top_items_gates.max(),
    }
    log2_num_experts = int(math.log2(num_experts))
    for t in [2**i for i in range(log2_num_experts + 1)] + [num_experts]:
        ratio = (num_experts_per_item >= t).sum().float() / group_size
        metrics[f"ratio_processed_items_by_at_least_{t}_experts"] = ratio

    return dispatcher, metrics


# ============================================================================
# Sparse MoE SPMD - PyTorch Implementation
# ============================================================================

class SparseMoELayer(nn.Module):
    """Sparse MoE layer that wraps an expert module.

    This is the PyTorch equivalent of JAX's sparse_moe_spmd lift transform.

    When the expert class is MlpBlock (the common case) and has_aux=False,
    all expert weights are stacked into single tensors and processed via
    torch.bmm for parallel execution — equivalent to JAX's vmap over experts.

    For custom expert classes or has_aux=True, falls back to nn.ModuleList
    with sequential execution.

    Args:
      expert_cls: The nn.Module class to use as each expert (e.g., MlpBlock).
      num_experts: Number of experts (global total).
      expert_kwargs: Keyword arguments for instantiating each expert.
      split_rngs: Whether each expert gets different random initialization.
      has_aux: Whether the expert returns auxiliary outputs.
      expert_parallel: If True, each rank only creates local experts
          (num_experts // ep_world_size). Requires EP to be initialized.
    """

    def __init__(self, expert_cls: type, num_experts: int,
                 expert_kwargs: Optional[dict] = None,
                 split_rngs: bool = False,
                 has_aux: bool = False,
                 expert_parallel: bool = False):
        super().__init__()
        self.num_experts = num_experts  # global
        self.has_aux = has_aux
        self.expert_parallel = expert_parallel
        expert_kwargs = expert_kwargs or {}

        # EP: 이 rank가 보유할 local expert 수 계산
        if expert_parallel and ep.get_ep_world_size() > 1:
            W = ep.get_ep_world_size()
            assert num_experts % W == 0, (
                f"num_experts ({num_experts}) must be divisible by "
                f"ep_world_size ({W})")
            self.local_num_experts = num_experts // W
        else:
            self.local_num_experts = num_experts

        # expert_cls가 nn.Module인 경우 ModuleList로 생성
        # DINOBatchedExperts처럼 batched expert를 직접 전달할 때는
        # DINOMoEBlock에서 직접 처리하므로 여기서는 ModuleList 방식만 지원
        self.batched = False
        self.experts = nn.ModuleList([
            expert_cls(**expert_kwargs)
            for _ in range(self.local_num_experts)
        ])
        if not split_rngs:
            state_dict_0 = self.experts[0].state_dict()
            for i in range(1, self.local_num_experts):
                self.experts[i].load_state_dict(state_dict_0)

    def forward(self, dispatcher: BaseDispatcher,
                inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Forward pass through the Sparse MoE layer.

        Args:
          dispatcher: A dispatcher that handles routing.
          inputs: (G, S, ...) tensor to route through experts.

        Returns:
          (G, S, ...) combined output, optionally with auxiliary outputs.

        When expert_parallel=True:
          dispatch → (E_local, W*G*C, ...) — local experts만 처리
          combine → all-to-all로 결과 반환 → (G, S, ...)
        """
        # Dispatch: (G, S, ...) -> (E_local, N, ...) where N = G*C or W*G*C
        dispatched = dispatcher.dispatch(inputs)

        if self.batched:
            expert_outputs = self.experts(dispatched)
        else:
            expert_outputs = []
            aux_outputs = []
            for e in range(self.local_num_experts):
                expert_input = dispatched[e]
                output = self.experts[e](expert_input)
                if self.has_aux:
                    output, aux = output
                    aux_outputs.append(aux)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs, dim=0)

        # Combine: (E_local, N, ...) -> (G, S, ...)
        combined = dispatcher.combine(expert_outputs)

        if self.has_aux and not self.batched:
            return combined, aux_outputs
        return combined
