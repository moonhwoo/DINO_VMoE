"""Module with routing layers for PyTorch."""
import functools
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe import moe

BaseDispatcher = moe.BaseDispatcher
KwArgs = Mapping[str, Any]
Metrics = Dict[str, torch.Tensor]


class NoisyTopExpertsPerItemRouter(nn.Module):
    """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2106.05974.

    First, a dense (gating) layer computes logits for each pair of (item, expert).
    Noise is added to these logits. The logits are normalized using a softmax over
    the expert dimension. This score determines which items are dispatched to
    which experts and how the outputs are combined.

    Auxiliary losses (G-Shard, Importance, Load) are computed to train the gating
    layer since the routing algorithm itself is non-differentiable.
    """

    def __init__(self,
                 num_experts: int,
                 input_dim: int,
                 num_selected_experts: int = 1,
                 noise_std: float = 1.0,
                 gshard_loss_weight: float = 0.0,
                 importance_loss_weight: float = 1.0,
                 load_loss_weight: float = 1.0,
                 dispatcher: Optional[KwArgs] = None,
                 deterministic: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 expert_parallel: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.noise_std = noise_std
        self.gshard_loss_weight = gshard_loss_weight
        self.importance_loss_weight = importance_loss_weight
        self.load_loss_weight = load_loss_weight
        self.dispatcher_kwargs = dict(dispatcher) if dispatcher else {}
        self.deterministic = deterministic
        self.dtype = dtype
        self.expert_parallel = expert_parallel

        # Gating dense layer: 전체 E experts에 대해 gate 계산 (EP 여부 무관)
        self.dense = nn.Linear(input_dim, num_experts, bias=False)
        if dtype is not None:
            self.dense = self.dense.to(dtype)

    def forward(self, inputs: torch.Tensor) -> Tuple[BaseDispatcher, Metrics]:
        """Forward pass.

        Args:
          inputs: (G, S, hidden_size) tensor.

        Returns:
          (dispatcher, metrics) tuple.
        """
        gates_softmax, metrics = self._compute_gates_softmax_and_metrics(inputs)
        dispatcher = self._create_dispatcher(gates_softmax)
        return dispatcher, metrics

    def _compute_gates_softmax_and_metrics(
            self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Metrics]:
        if inputs.ndim != 3:
            raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
        if not self.num_experts >= self.num_selected_experts >= 1:
            raise ValueError(
                f"num_experts >= num_selected_experts >= 1, but got "
                f"num_experts = {self.num_experts} and "
                f"num_selected_experts = {self.num_selected_experts}.")

        # Compute gating logits: (G, S, hidden) -> (G, S, E)
        gates_logits = self.dense(inputs)

        # Softmax over the expert dimension
        gates_softmax = F.softmax(gates_logits, dim=-1)

        # Importance loss (always computed)
        importance_loss = self._importance_auxiliary_loss_batched(gates_softmax)

        if self.deterministic or self.noise_std == 0.0:
            gshard_loss = self._gshard_auxiliary_loss_batched(gates_softmax)
            metrics = {
                "auxiliary_loss": _weighted_sum(
                    (self.gshard_loss_weight, gshard_loss),
                    (self.importance_loss_weight, importance_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
            }
            return gates_softmax, metrics
        else:
            noise_std = (1.0 / self.num_experts) * self.noise_std
            logits_noise = noise_std * torch.randn_like(gates_logits)
            gates_logits_noisy = gates_logits + logits_noise
            gates_softmax_noisy = F.softmax(gates_logits_noisy, dim=-1)

            load_loss = self._load_auxiliary_loss_batched(
                gates_logits, gates_logits_noisy, noise_std)
            gshard_loss = self._gshard_auxiliary_loss_batched(gates_softmax_noisy)

            metrics = {
                "auxiliary_loss": _weighted_sum(
                    (self.gshard_loss_weight, gshard_loss),
                    (self.importance_loss_weight, importance_loss),
                    (self.load_loss_weight, load_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
                "load_loss": load_loss,
            }
            return gates_softmax_noisy, metrics

    def _create_dispatcher(self, gates_dispatch: torch.Tensor,
                           capacity: Optional[int] = None) -> BaseDispatcher:
        """Creates a dispatcher implementing TopExpertsPerItem routing.
        #dispatcher = self.router._create_dispatcher(gates_softmax) 여기서 호출
        #gates_dispatch== gates_softmax

        All groups are processed in parallel using batched tensor operations
        (equivalent to jax.vmap over the group dimension).

        Args:
            gates_dispatch: (G, S, E) gating tensor.
            capacity: 명시적 capacity override. 지정하면 capacity_factor 대신 사용.
                      EP 환경에서 S에 padding이 포함된 경우, 실제 유효 토큰 수 기반의
                      capacity를 외부에서 계산해 전달할 때 사용.
                      (JAX SPMD는 padding이 없으므로 이 인자가 불필요하지만,
                       PyTorch EP의 all-to-all padding 문제를 보정하기 위해 추가)
        """
        dispatcher_kwargs = dict(self.dispatcher_kwargs)
        use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
        # EP flag를 dispatcher에 전달
        dispatcher_kwargs["expert_parallel"] = self.expert_parallel

        # capacity 명시 지정 시 capacity_factor 대신 사용.
        # EP per-scale 패딩으로 S가 부풀었을 때 유효 토큰 기반 capacity를 보정.
        if capacity is not None:
            dispatcher_kwargs.pop("capacity_factor", None)
            dispatcher_kwargs["capacity"] = capacity

        dispatcher = moe.get_top_experts_per_item_dispatcher_batched(
            gates=gates_dispatch,
            num_selected_experts=self.num_selected_experts,
            **dispatcher_kwargs)

        if use_bfloat16:
            dispatcher = moe.Bfloat16Dispatcher(dispatcher)
        return dispatcher

    @staticmethod
    def _gshard_auxiliary_loss(gates: torch.Tensor) -> torch.Tensor:
        """G-Shard auxiliary loss from Algorithm 1 in https://arxiv.org/pdf/2006.16668.pdf."""
        _, num_experts = gates.shape
        mean_gates_per_expert = gates.mean(dim=0)
        mean_top1_per_expert = F.one_hot(
            gates.argmax(dim=1), num_experts).float().mean(dim=0)
        auxiliary_loss = (mean_top1_per_expert * mean_gates_per_expert).mean()
        auxiliary_loss *= num_experts ** 2
        return auxiliary_loss

    def _gshard_auxiliary_loss_batched(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute G-Shard loss per group and average. Vectorized over G dim."""
        # gates: (G, S, E)
        _, _, num_experts = gates.shape
        mean_gates_per_expert = gates.mean(dim=1)  # (G, E)
        mean_top1_per_expert = F.one_hot(
            gates.argmax(dim=2), num_experts).float().mean(dim=1)  # (G, E)
        auxiliary_loss = (mean_top1_per_expert * mean_gates_per_expert).mean(dim=1)  # (G,)
        auxiliary_loss = auxiliary_loss * num_experts ** 2
        return auxiliary_loss.mean()

    @staticmethod
    def _importance_auxiliary_loss(gates: torch.Tensor) -> torch.Tensor:
        """Importance auxiliary loss: coefficient of variation squared."""
        axis = tuple(range(gates.ndim - 1))
        importance_per_expert = gates.sum(dim=axis)
        # Use correction=0 to match JAX's jnp.std (population std, ddof=0).
        std_importance = importance_per_expert.std(correction=0)
        mean_importance = importance_per_expert.mean()
        return (std_importance / mean_importance) ** 2

    def _importance_auxiliary_loss_batched(self, gates: torch.Tensor) -> torch.Tensor:
        """Importance auxiliary loss vectorized over G dim."""
        # gates: (G, S, E)
        importance_per_expert = gates.sum(dim=1)  # (G, E) — sum over S (items)
        # Use correction=0 to match JAX's jnp.std (population std, ddof=0).
        std_importance = importance_per_expert.std(dim=-1, correction=0)  # (G,)
        mean_importance = importance_per_expert.mean(dim=-1)  # (G,)
        cv_sq = (std_importance / mean_importance) ** 2  # (G,)
        return cv_sq.mean()

    @staticmethod
    def _load_auxiliary_loss(logits: torch.Tensor, logits_noisy: torch.Tensor,
                             noise_std: float,
                             num_selected_experts: int) -> torch.Tensor:
        """Load auxiliary loss using Gaussian CDF."""
        num_experts = logits_noisy.shape[-1]
        # Get threshold: the K-th largest noisy logit value
        _, top_indices = torch.topk(logits_noisy, num_selected_experts, dim=-1)
        threshold_index = top_indices[..., -1]
        threshold_per_item = (
            F.one_hot(threshold_index, num_experts).float() * logits_noisy
        ).sum(dim=-1)

        # How far each (item, expert) is from the threshold, normalized
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win = noise_required_to_win / noise_std

        # Probability of being above threshold: 1 - CDF(noise_required)
        normal = torch.distributions.Normal(0, 1)
        p = 1.0 - normal.cdf(noise_required_to_win)

        # Average probability per expert, then CV squared.
        # Use correction=0 to match JAX's jnp.std (population std, ddof=0).
        p_mean = p.mean(dim=0)
        return (p_mean.std(correction=0) / p_mean.mean()) ** 2

    def _load_auxiliary_loss_batched(self, logits: torch.Tensor,
                                     logits_noisy: torch.Tensor,
                                     noise_std: float) -> torch.Tensor:
        """Load auxiliary loss vectorized over G dim."""
        # logits, logits_noisy: (G, S, E)
        num_experts = logits_noisy.shape[-1]

        # Step 1: Threshold — K-th largest noisy logit per item
        _, top_indices = torch.topk(
            logits_noisy, self.num_selected_experts, dim=-1)  # (G, S, K)
        threshold_index = top_indices[..., -1]  # (G, S)
        threshold_per_item = (
            F.one_hot(threshold_index, num_experts).float() * logits_noisy
        ).sum(dim=-1)  # (G, S)

        # Step 2: Noise required to win, normalized
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits  # (G, S, E)
        noise_required_to_win = noise_required_to_win / noise_std

        # Step 3: Probability via Gaussian CDF
        normal = torch.distributions.Normal(0, 1)
        p = 1.0 - normal.cdf(noise_required_to_win)  # (G, S, E)

        # Step 4: CV² per group, then average
        p_mean = p.mean(dim=1)  # (G, E) — average over S (items)
        std_p = p_mean.std(dim=-1, correction=0)  # (G,)
        mean_p = p_mean.mean(dim=-1)  # (G,)
        cv_sq = (std_p / mean_p) ** 2  # (G,)
        return cv_sq.mean()


class NoisyTopItemsPerExpertRouter(nn.Module):
    """Noisy TopItemsPerExpert router (Expert Choice Routing).

    Instead of picking the Top-K experts for each item, here we pick the
    Top-C items for each expert. This ensures balanced load but the number
    of experts per item can vary.

    Coined "Experts Choice Routing" in https://arxiv.org/abs/2202.09368.
    """

    def __init__(self,
                 num_experts: int,
                 input_dim: int,
                 noise_std: float = 1.0,
                 dispatcher: Optional[KwArgs] = None,
                 deterministic: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 expert_parallel: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.dispatcher_kwargs = dict(dispatcher) if dispatcher else {}
        self.deterministic = deterministic
        self.dtype = dtype
        self.expert_parallel = expert_parallel

        self.dense = nn.Linear(input_dim, num_experts, bias=False)
        if dtype is not None:
            self.dense = self.dense.to(dtype)

    def forward(self, inputs: torch.Tensor) -> Tuple[BaseDispatcher, Metrics]:
        if inputs.ndim != 3:
            raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")

        gates_softmax = self._compute_gates_softmax(inputs)
        dispatcher, metrics = self._create_dispatcher_and_metrics(gates_softmax)
        metrics["auxiliary_loss"] = torch.tensor(0.0, device=inputs.device)
        return dispatcher, metrics

    def _compute_gates_softmax(self, inputs: torch.Tensor) -> torch.Tensor:
        gates_logits = self.dense(inputs)
        if self.deterministic or self.noise_std == 0.0:
            return F.softmax(gates_logits, dim=-1)
        else:
            noise_std = (1.0 / self.num_experts) * self.noise_std
            logits_noise = noise_std * torch.randn_like(gates_logits)
            return F.softmax(gates_logits + logits_noise, dim=-1)

    def _create_dispatcher_and_metrics(
            self, gates_dispatch: torch.Tensor
    ) -> Tuple[BaseDispatcher, Metrics]:
        """Creates a dispatcher implementing TopItemsPerExpert routing.

        All groups are processed in parallel using batched tensor operations
        (equivalent to jax.vmap over the group dimension).
        """
        dispatcher_kwargs = dict(self.dispatcher_kwargs)
        use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
        dispatcher_kwargs["expert_parallel"] = self.expert_parallel

        dispatcher, metrics = moe.get_top_items_per_expert_dispatcher_batched(
            gates=gates_dispatch, **dispatcher_kwargs)

        if use_bfloat16:
            dispatcher = moe.Bfloat16Dispatcher(dispatcher)
        return dispatcher, metrics


# ============================================================================
# Helpers
# ============================================================================

def _weighted_sum(*args) -> torch.Tensor:
    """Returns a weighted sum of [(weight, element), ...] for weights > 0."""
    result = None
    for w, x in args:
        if w > 0:
            term = x * w
            result = term if result is None else result + term
    if result is not None:
        return result
    # 모든 가중치 0인 경우: 입력 텐서의 device/dtype 유지 (CUDA/CPU mismatch 방지)
    for w, x in args:
        if isinstance(x, torch.Tensor):
            return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.tensor(0.0)


def _batch_dispatchers(dispatchers: list) -> BaseDispatcher:
    """Combines per-group dispatchers into a single batched dispatcher.

    For EinsumDispatchers, stacks combine_weights along a new leading dim (G).
    """
    if not dispatchers:
        raise ValueError("No dispatchers to batch")

    d0 = dispatchers[0]
    if isinstance(d0, moe.EinsumDispatcher):
        combine_weights = torch.stack(
            [d.combine_weights for d in dispatchers], dim=0)
        dispatch_weights = None
        if d0.dispatch_weights is not None:
            dispatch_weights = torch.stack(
                [d.dispatch_weights for d in dispatchers], dim=0)
        return moe.EinsumDispatcher(
            combine_weights=combine_weights,
            dispatch_weights=dispatch_weights,
            expert_parallel=d0.expert_parallel)
    elif isinstance(d0, moe.ExpertIndicesDispatcher):
        indices = torch.stack([d.indices for d in dispatchers], dim=0)
        combine_weights = torch.stack(
            [d.combine_weights for d in dispatchers], dim=0)
        return moe.ExpertIndicesDispatcher(
            indices=indices,
            combine_weights=combine_weights,
            num_experts=d0.num_experts,
            capacity=d0.capacity,
            expert_parallel=d0.expert_parallel)
    elif isinstance(d0, moe.DenseEinsumDispatcher):
        combine_weights = torch.stack(
            [d.combine_weights for d in dispatchers], dim=0)
        return moe.DenseEinsumDispatcher(
            combine_weights=combine_weights,
            expert_parallel=d0.expert_parallel)
    else:
        raise TypeError(f"Unsupported dispatcher type: {type(d0)}")
