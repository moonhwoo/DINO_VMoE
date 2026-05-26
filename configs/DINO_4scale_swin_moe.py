_base_ = ['DINO_4scale_swin.py']

# ============================================================================
# MoE 설정
# ============================================================================
use_moe = True
moe_layers = [1,3,5,7,9,11]              # 6개 인코더 layer 중 (2)  (4),6번째에 MoE 적용
moe_num_experts = 8                  # expert 수
moe_num_selected_experts = 2         # top-K (각 토큰 → 2개 expert)
moe_capacity_factor = 1.25        # 25% 여유 버퍼 학습할떈 1.25로 학습
moe_noise_std = 1.0                  # routing noise (exploration)  =>지금은 non-deterministic
                                        #=> 0.0 으로하면 deterministic 과 똑같이 동작
moe_loss_coef = 0.005                # auxiliary loss 가중치 (weight_dict)
moe_gshard_loss_weight = 0        # G-Shard loss weight (router 내부)
moe_importance_loss_weight = 1.0     # importance loss weight (router 내부)
moe_load_loss_weight = 1.0           # load loss weight (router 내부)
moe_mode = 'scale_aware'                # 1.'baseline' (전체 토큰 단일 그룹) —
                                     # 2.'scale_aware'(scale 별로 토큰 그룹)
moe_group_images = 4                # 그룹당 이미지 수 (JAX 원본의 min_batch_size_per_device)
                                     # group_size = moe_group_images * N_total
                                     # 1이면 이미지 1장 = 1그룹, 4이면 4장 묶어서 1그룹
moe_expert_parallel = True           # Expert Parallelism: True이면 expert를 GPU에 분산
                                     # 2 GPU + 8 experts → 각 GPU에 4 experts
                                     # all-to-all로 토큰 교환
                                     # DDP 대신 sync_shared_gradients() 사용
moe_split_rngs = False               # False: 모든 expert를 동일 weight로 초기화 (JAX V-MoE 원본 방식)
                                     #   → routing noise가 학습 중 expert 분화를 유도
                                     # True:  각 expert를 독립적으로 random init

# ============================================================================
# Class-Aware Routing Loss (No EMA, Setup B)
# ============================================================================
moe_class_routing_loss_weight_init  = 0.05   # λ 초기값 (epoch 0)
moe_class_routing_loss_weight_final = 0.10   # λ 최종값 (warmup 완료 후)
moe_class_routing_warmup_epochs     = 2      # 선형 warmup 기간 (epoch)
moe_class_routing_alpha             = 1.0    # inter loss 비중 (intra:inter = 1:1)
# num_classes는 기존 args.num_classes 재활용
# 실효 비중 = moe_loss_coef(0.005) × λ(0.10) = 0.0005

# ============================================================================
# Ablation 대상 (주석 해제하여 실험)
# ============================================================================
# moe_layers = [5]                   # MoE 1개 layer만
# moe_layers = [0, 1, 2, 3, 4, 5]   # 전체 layer MoE
# moe_num_experts = 4                # expert 수 줄이기
# moe_num_experts = 16               # expert 수 늘리기
# moe_num_selected_experts = 1       # top-1
# moe_capacity_factor = 1.5          # 더 큰 여유 버퍼
# moe_group_images = 4               # 4장 묶어서 1그룹
