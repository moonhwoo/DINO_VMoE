"""
JAX V-MoE 방식의 padding sampler / dataset view.

MoE 모델은 배치 크기가 moe_group_images의 배수여야 capacity 계산이 일관됨.
_PaddedSampler 로 인덱스를 배수에 맞게 padding하고,
_PaddedDatasetView 로 padding 샘플에 _is_padding=True 플래그를 달아
engine.evaluate() 에서 metric 계산에서만 제외한다.

사용처: tools/main.py (validation), tools/eval_on_test.py (test)
"""
import math
import torch


class _PaddedSampler(torch.utils.data.Sampler):
    """Sequential sampler padded to (world_size × moe_group) divisibility.

    JAX V-MoE와 동일한 방식: 실제 이미지 먼저, 부족한 만큼 앞 인덱스로 padding.
    패딩된 샘플은 실제 이미지 인덱스를 재사용하므로 dataset 자체는 변경 불필요.

    n_real_local: 이 rank에서 실제로 평가해야 할 이미지 수.

    예시 (N=5, world_size=2, moe_group=2):
      all_indices = [0,1,2,3,4, 0,1,2]  (padding 3개)
      Rank 0: [0,1,2,3] → n_real_local=4  (전부 real)
      Rank 1: [4,0,1,2] → n_real_local=1  (index 4만 real, 0,1,2는 padding)

    Args:
        n_total   : 전체 실제 이미지 수
        pad_to    : moe_group_images (한 배치 내 이미지 수)
        rank      : 현재 프로세스 rank (distributed 시)
        world_size: 전체 GPU 수 (distributed 시)
    """
    def __init__(self, n_total: int, pad_to: int, rank: int = 0, world_size: int = 1):
        divisor = world_size * pad_to
        n_padded = math.ceil(n_total / divisor) * divisor
        all_indices = list(range(n_total)) + [i % n_total for i in range(n_padded - n_total)]
        if world_size > 1:
            per_rank = n_padded // world_size
            rank_start = rank * per_rank
            rank_end = (rank + 1) * per_rank
            indices = all_indices[rank_start:rank_end]
            self.n_real_local = max(0, min(n_total, rank_end) - rank_start)
        else:
            indices = all_indices
            self.n_real_local = n_total
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class _PaddedDatasetView(torch.utils.data.Dataset):
    """JAX VALID_KEY 방식: 모든 샘플(real + padding)을 모델에 통과시키되,
    padding 샘플에 _is_padding=True 플래그를 달아 metric 계산에서만 제외.

    - pos < n_real_local  → _is_padding=False (real)
    - pos >= n_real_local → _is_padding=True  (padding, MoE capacity용)

    engine.evaluate()의 res 딕셔너리 생성 시 _is_padding=True는 제외됨.
    """
    def __init__(self, base_dataset, indices: list, n_real_local: int):
        self.base = base_dataset
        self.indices = indices
        self.n_real_local = n_real_local

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        img, target = self.base[self.indices[pos]]
        target = dict(target)
        target['_is_padding'] = torch.tensor(pos >= self.n_real_local, dtype=torch.bool)
        return img, target
