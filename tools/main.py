# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from util.padded_sampler import _PaddedSampler, _PaddedDatasetView
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--encoder_init', default='random',
                        choices=['random', 'tile', 'last'],
                        help='7층 이후 encoder 초기화 방식: '
                             'random=랜덤(기본), '
                             'tile=1~6층 패턴 반복(0→6,1→7,...), '
                             'last=마지막 6층을 7~12층 전체에 복사')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # routing 시각화 옵션
    parser.add_argument('--vis_image_ids', type=int, nargs='+', default=[],
        help='COCO val image ID 목록. 매 윈도우마다 이 이미지들로 routing 시각화 (예: 199 1006 6462 7714  5 8 33 52  429 449 458 573)')
    parser.add_argument('--vis_every_n_epochs', type=int, default=5,
        help='몇 epoch 윈도우마다 시각화 저장 여부를 체크할지 (default: 5)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--early_stop_patience', default=0, type=int,
                        help='stop training if no improvement after this many epochs (0 disables)')
    
    return parser


def _tile_encoder_layers(pretrained_st, model_enc_layers, mode='tile', logger=None):
    """pretrained N층 encoder 가중치를 model_enc_layers층에 맞게 복사.

    mode='tile': 0→6, 1→7, 2→8, ... 패턴 반복
    mode='last': 마지막 층(N-1)을 N~model_enc_layers-1 전체에 복사
    """
    from collections import OrderedDict
    new_st = OrderedDict(pretrained_st)

    existing = set()
    for k in new_st:
        if k.startswith('transformer.encoder.layers.'):
            existing.add(int(k.split('.')[3]))
    if not existing:
        return new_st
    pretrain_n = max(existing) + 1  # e.g. 6

    if model_enc_layers <= pretrain_n:
        return new_st

    if mode == 'tile':
        for src_idx in range(pretrain_n):
            dst_idx = src_idx + pretrain_n
            while dst_idx < model_enc_layers:
                for k, v in list(pretrained_st.items()):
                    prefix = f'transformer.encoder.layers.{src_idx}.'
                    if k.startswith(prefix):
                        new_key = f'transformer.encoder.layers.{dst_idx}.' + k[len(prefix):]
                        new_st[new_key] = v.clone()
                        if logger:
                            logger.info(f'  [tile_layers] {k} → {new_key}')
                dst_idx += pretrain_n
    elif mode == 'last':
        src_idx = pretrain_n - 1  # 마지막 pretrained 층 (e.g. layer 5)
        for dst_idx in range(pretrain_n, model_enc_layers):
            for k, v in list(pretrained_st.items()):
                prefix = f'transformer.encoder.layers.{src_idx}.'
                if k.startswith(prefix):
                    new_key = f'transformer.encoder.layers.{dst_idx}.' + k[len(prefix):]
                    new_st[new_key] = v.clone()
                    if logger:
                        logger.info(f'  [last_layer] {k} → {new_key}')

    return new_st


def _expand_pretrained_ffn_to_moe(pretrained_st, model_st, moe_layers, logger=None):
    """JAX V-MoE의 expand_tile 방식: pretrained FFN weight → MoE expert weight로 복사.

    JAX 원본 (vmoe/initialization/rules.py):
        array = jnp.expand_dims(W, axis=0)       # (out, in) → (1, out, in)
        array = jnp.tile(array, [E, 1, 1])       # → (E, out, in)

    PyTorch 동등:
        W.unsqueeze(0).expand(E, -1, -1).clone()

    Pretrained key 매핑 (MoE layer i):
        layers.{i}.linear1.weight (d_ffn, d_model)  → layers.{i}.moe_block.experts.w1 (E, d_ffn, d_model)
        layers.{i}.linear1.bias   (d_ffn,)           → layers.{i}.moe_block.experts.b1 (E, d_ffn)
        layers.{i}.linear2.weight (d_model, d_ffn)   → layers.{i}.moe_block.experts.w2 (E, d_model, d_ffn)
        layers.{i}.linear2.bias   (d_model,)          → layers.{i}.moe_block.experts.b2 (E, d_model)
    """
    from collections import OrderedDict
    new_st = OrderedDict(pretrained_st)

    # MoE layer에 해당하는 FFN key → expert key로 변환
    ffn_to_expert = {
        'linear1.weight': 'moe_block.experts.w1',
        'linear1.bias':   'moe_block.experts.b1',
        'linear2.weight': 'moe_block.experts.w2',
        'linear2.bias':   'moe_block.experts.b2',
    }

    for layer_idx in moe_layers:
        for ffn_key_suffix, expert_key_suffix in ffn_to_expert.items():
            # pretrained key: transformer.encoder.layers.{i}.linear1.weight
            pretrained_key = f'transformer.encoder.layers.{layer_idx}.{ffn_key_suffix}'
            # model key: transformer.encoder.layers.{i}.moe_block.experts.w1
            model_key = f'transformer.encoder.layers.{layer_idx}.{expert_key_suffix}'

            if pretrained_key not in new_st:
                continue
            if model_key not in model_st:
                continue

            pretrained_w = new_st.pop(pretrained_key)  # 원본 key 제거
            num_experts = model_st[model_key].shape[0]  # E (local_num_experts)

            # expand_tile: (out, in) → (E, out, in) / (out,) → (E, out)
            expanded = pretrained_w.unsqueeze(0).expand(num_experts, *pretrained_w.shape).clone()

            new_st[model_key] = expanded
            if logger:
                logger.info(
                    f"  [expand_tile] {pretrained_key} {list(pretrained_w.shape)} "
                    f"→ {model_key} {list(expanded.shape)}")

    # MoE layer의 나머지 FFN key (linear1/linear2) 중 변환 안 된 것 정리
    # (norm2 등은 key가 같으므로 그대로 매칭됨)
    removed = []
    for k in list(new_st.keys()):
        for layer_idx in moe_layers:
            prefix = f'transformer.encoder.layers.{layer_idx}.'
            if k.startswith(prefix) and ('linear1' in k or 'linear2' in k):
                new_st.pop(k)
                removed.append(k)
    if removed and logger:
        logger.info(f"  [expand_tile] Removed unmapped FFN keys: {removed}")

    return new_st


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)

    # EP 초기화: distributed가 활성화된 후에 호출해야 함
    # (config 로드 전이지만, EP init은 distributed 상태만 필요)
    if args.distributed:
        from models.moe.expert_parallel import init_expert_parallel
        init_expert_parallel()

    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # EP 모드에서 DDP broadcast 없이 shared param(router 등)이 rank간 동일하게 초기화되도록
    # 모델 init용 seed는 모든 rank 동일하게 설정
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    use_ep = getattr(args, 'moe_expert_parallel', False) and getattr(args, 'use_moe', False)
    if args.distributed:
        if use_ep:
            # EP 모드: DDP 사용하지 않음
            # - Expert 파라미터: 각 GPU가 서로 다른 expert를 보유하므로 all-reduce 불가
            # - Shared 파라미터: sync_shared_gradients()로 수동 동기화
            # → engine.py의 train_one_epoch에서 backward 후 sync_shared_gradients() 호출
            logger.info("Using Expert Parallelism (EP) — DDP disabled, "
                        "sync_shared_gradients() will be used instead")
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # 데이터 로딩/augmentation용 seed: rank마다 다르게 설정
    # GPU마다 다른 augmentation을 적용하여 학습 다양성 확보
    data_seed = args.seed + utils.get_rank()
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    dataset_train = build_dataset(image_set='train', args=args)
    # allow evaluating on the COCO `test` split when --test is passed
    val_image_set = 'test' if getattr(args, 'test', False) else 'val'
    dataset_val = build_dataset(image_set=val_image_set, args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # JAX V-MoE 방식: tail 샘플을 버리지 않고 padding하여 전부 모델에 통과.
    # padding 샘플은 _is_padding=True 플래그로 마킹 → engine.evaluate()에서 metric 제외.
    val_batch_size = getattr(args, 'moe_group_images', 1)
    _val_sampler = _PaddedSampler(
        len(dataset_val), val_batch_size,
        rank=utils.get_rank(), world_size=utils.get_world_size(),
    )
    padded_val_view = _PaddedDatasetView(dataset_val, _val_sampler.indices, _val_sampler.n_real_local)
    data_loader_val = DataLoader(
        padded_val_view, val_batch_size,
        sampler=torch.utils.data.SequentialSampler(padded_val_view),
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
    )

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    # prefer an explicit --resume passed by user; only fallback to output_dir/checkpoint.pth
    if not args.resume and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # EP: full checkpoint(8 experts) → 현재 rank의 local experts만 추출
        resume_sd = checkpoint['model']
        if use_ep:
            from models.moe.expert_parallel import scatter_expert_state_dict
            resume_sd = scatter_expert_state_dict(resume_sd, model_without_ddp)
        model_without_ddp.load_state_dict(resume_sd)
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        # ── encoder layer 초기화: tile / last / random ──
        _enc_init = getattr(args, 'encoder_init', 'random')
        if _enc_init in ('tile', 'last'):
            model_enc_layers = getattr(args, 'enc_layers', 6)
            logger.info(f'[encoder_init={_enc_init}] pretrained layers → {model_enc_layers}층으로 복사')
            _tmp_st = _tile_encoder_layers(_tmp_st, model_enc_layers, mode=_enc_init, logger=logger)

        # ── JAX V-MoE expand_tile: pretrained FFN → MoE expert 가중치 복사 ──
        # pretrained의 linear1/linear2를 MoE layer의 expert w1/b1/w2/b2로 expand_tile.
        # JAX 원본: jnp.expand_dims(W, axis=0) → jnp.tile(W, [E,1,1])
        # PyTorch:  W.unsqueeze(0).expand(E, ...).clone()
        if getattr(args, 'use_moe', False):
            _tmp_st = _expand_pretrained_ffn_to_moe(
                _tmp_st, model_without_ddp.state_dict(),
                moe_layers=getattr(args, 'moe_layers', []),
                logger=logger)

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    no_improve_epochs = 0
    patience = getattr(args, 'early_stop_patience', 0)

    # ── routing 시각화 윈도우 추적 변수 ──
    # vis_last_best_map: 마지막으로 시각화했을 때의 global best mAP
    # vis_window_had_best: 현재 N-epoch 윈도우 안에서 global best가 갱신됐는지
    from visualize_routing import run_routing_visualization
    vis_last_best_map   = 0.0
    vis_window_had_best = False

    def compute_class_routing_lambda(epoch: int, args) -> float:
        warmup = getattr(args, 'moe_class_routing_warmup_epochs', 0)
        init   = getattr(args, 'moe_class_routing_loss_weight_init',  0.0)
        final  = getattr(args, 'moe_class_routing_loss_weight_final', 0.0)
        if warmup <= 0:
            return final
        return init + (final - init) * min(epoch / warmup, 1.0)

    def update_class_routing_weight(model, lambda_val: float):
        for m in model.modules():
            if hasattr(m, 'set_class_routing_loss_weight'):
                m.set_class_routing_loss_weight(lambda_val)

    for epoch in range(args.start_epoch, args.epochs):
        # ── Class Routing Loss λ warmup ──
        lam = compute_class_routing_lambda(epoch, args)
        update_class_routing_weight(model_without_ddp, lam)
        if utils.get_rank() == 0:
            logger.info(f"[Epoch {epoch}] class_routing_lambda = {lam:.4f}")

        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            #if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
            #    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # EP: 모든 rank의 expert를 rank 0으로 gather하여 전체 모델 저장
            if use_ep:
                from models.moe.expert_parallel import gather_expert_state_dict
                gathered_model_sd = gather_expert_state_dict(model_without_ddp)
            else:
                gathered_model_sd = model_without_ddp.state_dict()

            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': gathered_model_sd,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest_regular = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest_regular:
            vis_window_had_best = True   # 이 윈도우에서 global best 갱신됨
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            # EP: gather 재사용 (같은 epoch 내에서 이미 gather한 state_dict 사용)
            if use_ep:
                from models.moe.expert_parallel import gather_expert_state_dict
                best_model_sd = gather_expert_state_dict(model_without_ddp)
            else:
                best_model_sd = model_without_ddp.state_dict()
            utils.save_on_master({
                'model': best_model_sd,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        improved_ema = False
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest_ema = best_map_holder.update(map_ema, epoch, is_ema=True)
            improved_ema = _isbest_ema
            if _isbest_ema:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        # early stopping logic: consider improvement from regular or ema evaluation
        improved = bool(_isbest_regular) or bool(improved_ema)
        if improved:
            no_improve_epochs = 0
        else:
            if patience > 0:
                no_improve_epochs += 1

        if patience > 0 and no_improve_epochs > patience:
            print(f"Early stopping triggered: no improvement for {no_improve_epochs} epochs (patience={patience})")
            break

        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # ── 5-epoch 윈도우 단위 routing 시각화 ──
            # 조건: N epoch 윈도우 끝 + vis_image_ids 지정 + rank 0
            # 이 윈도우 안에서 global best 갱신이 있었고 이전 vis보다 높아야 저장
            if ((epoch + 1) % args.vis_every_n_epochs == 0
                    and getattr(args, 'vis_image_ids', [])):

                current_best = best_map_holder.best_all.best_res

                if vis_window_had_best and current_best > vis_last_best_map:
                    best_ckpt_path = output_dir / 'checkpoint_best_regular.pth' #best_ckpt→ 지금까지 최고 성능 모델
                    curr_ckpt_path = output_dir / 'checkpoint.pth'  #curr_ckpt → 현재 학습 중 모델
                    if best_ckpt_path.exists() and curr_ckpt_path.exists():
                        # best 가중치 로드 후 즉시 해제 (deepcopy 대신 디스크 재활용)
                        best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
                        best_epoch = best_ckpt.get('epoch', epoch)
                        model_without_ddp.load_state_dict(best_ckpt['model'])
                        del best_ckpt   # 즉시 해제 (~2.6GB CPU RAM 반환)

                        # 시각화 실행
                        tag = f'ep{best_epoch}_map{current_best:.3f}'
                        run_routing_visualization(
                            model_without_ddp, dataset_val,
                            args.vis_image_ids, best_epoch,
                            args.output_dir, device, tag=tag)

                        # 학습 가중치 복원: 이미 저장된 checkpoint.pth 재활용
                        curr_ckpt = torch.load(curr_ckpt_path, map_location='cpu')
                        model_without_ddp.load_state_dict(curr_ckpt['model'])
                        del curr_ckpt   # 즉시 해제

                    vis_last_best_map = current_best

                vis_window_had_best = False   # 다음 윈도우를 위해 초기화

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
