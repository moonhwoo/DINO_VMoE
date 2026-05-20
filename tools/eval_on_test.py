#!/usr/bin/env python3
"""
Evaluate checkpoint on COCO `test` split (expects GT in instances_test2017.json).

MoE + Expert Parallelism 지원:
  PYTHONPATH=/home/moonhw/논문코드공부/V_MoE_study/dino-moe-project:/home/moonhw/논문코드공부/V_MoE_study/dino-moe-project/tools torchrun --nproc_per_node=2 tools/eval_on_test.py   --config_file configs/DINO_4scale_swin_moe.py   --coco_path /home/moonhw/SIXray-D/SIXray_COCO_ROOT   --checkpoint outputs/dino_SIXray_swin_moe/checkpoint_best_regular.pth   --output_dir outputs/dino_SIXray_swin_moe/SIXray_test   --device cuda   --num_workers 4   --per_class   --save_per_class   --also_ap50_ap75   --save_results   --test_dataset sixray

Non-MoE (single GPU):
  python tools/eval_on_test.py \
    --config_file config/DINO/DINO_4scale_swin.py \
    --coco_path /home/moonhw/SIXray-D/SIXray_COCO_ROOT \
    --checkpoint outputs/.../checkpoint_best_regular.pth \
    --output_dir outputs/.../SIXray_test \
    --device cuda --num_workers 4 \
    --per_class --save_per_class --also_ap50_ap75 --save_results \
    --test_dataset sixray
"""
import argparse
import math
import os
import random

import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from util.slconfig import SLConfig
import util.misc as utils
from util.utils import to_device
from util.padded_sampler import _PaddedSampler, _PaddedDatasetView
from main import build_model_main
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, test


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True)
    p.add_argument('--coco_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--fix_size', action='store_true')
    p.add_argument('--dataset_file', default='coco')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save_results', action='store_true',
               help='Save detection results json during evaluation.')

    # ✅ 추가: 클래스별 AP 출력/저장
    p.add_argument('--per_class', action='store_true',
                   help='Print per-class AP (mAP@[0.50:0.95]) and optionally AP50/AP75.')
    p.add_argument('--save_per_class', action='store_true',
                   help='Save per-class AP to CSV under output_dir/eval/per_class_ap.csv')
    p.add_argument('--also_ap50_ap75', action='store_true',
                   help='Also compute per-class AP50 and AP75.')

    p.add_argument('--test_dataset', default='sixray',
                   choices=['sixray', 'pidray'],
                   help='Test dataset to use (sixray or pidray)')

    # distributed (torchrun 시 자동 설정)
    p.add_argument('--dist_url', default='env://')
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--rank', default=0, type=int)
    return p.parse_args()


def setup_distributed(args):
    """torchrun 환경이면 distributed + EP 초기화."""
    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ.get('RANK', '0'))
        args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        args.distributed = True

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

        # EP 초기화
        if getattr(args, 'moe_expert_parallel', False):
            from models.moe.expert_parallel import init_expert_parallel
            init_expert_parallel()
            print(f"[Rank {args.rank}] EP initialized (world_size={args.world_size})")
    else:
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1


def _safe_get_cat_names(base_ds, cat_ids):
    """Return category names aligned with cat_ids. Fallback to str(cat_id)."""
    try:
        cats = base_ds.loadCats(cat_ids)
        id2name = {c["id"]: c.get("name", str(c["id"])) for c in cats}
        return [id2name.get(i, str(i)) for i in cat_ids]
    except Exception:
        return [str(i) for i in cat_ids]


def _per_class_ap_from_cocoeval(coco_eval, base_ds, also_ap50_ap75=False):
    """
    Returns list of dict per class:
      {cat_id, name, ap_50_95, ap50 (opt), ap75 (opt)}
    """
    import numpy as np

    # precision: [T, R, K, A, M]
    precision = coco_eval.eval.get("precision", None)
    if precision is None:
        raise RuntimeError("coco_eval.eval['precision'] not found. Evaluation may have failed.")

    cat_ids = coco_eval.params.catIds  # length K (category ids)
    cat_names = _safe_get_cat_names(base_ds, cat_ids)

    # area=all => aidx=0, maxDets=100 usually midx=-1
    aidx = 0
    midx = -1

    # AP@[0.50:0.95]
    prec_all = precision[:, :, :, aidx, midx]  # [T, R, K]
    results = []

    for k, (cid, name) in enumerate(zip(cat_ids, cat_names)):
        p = prec_all[:, :, k]
        p = p[p > -1]
        ap_50_95 = float(np.mean(p)) if p.size else float("nan")

        row = {"cat_id": int(cid), "name": name, "ap_50_95": ap_50_95}

        if also_ap50_ap75:
            # IoU thresholds index: 0 -> 0.50, 5 -> 0.75
            p50 = precision[0, :, k, aidx, midx]
            p50 = p50[p50 > -1]
            ap50 = float(np.mean(p50)) if p50.size else float("nan")

            p75 = precision[5, :, k, aidx, midx]
            p75 = p75[p75 > -1]
            ap75 = float(np.mean(p75)) if p75.size else float("nan")

            row["ap50"] = ap50
            row["ap75"] = ap75

        results.append(row)

    # 보기 좋게 ap_50_95 내림차순 정렬(원하면 cat_id 정렬로 바꿔도 됨)
    results.sort(key=lambda d: (-(d["ap_50_95"] if d["ap_50_95"] == d["ap_50_95"] else -1e9)))
    return results


def main():
    args = parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Distributed + EP 초기화 (torchrun 환경이면 자동 활성화)
    setup_distributed(args)

    if args.test_dataset == 'pidray':
        print(f"[INFO] Switching to PIDray test dataset")
        args.coco_path = '/home/moonhw/PIDray/PIDray_COCO_ROOT'  # PIDray 경로로 변경
        print(f"[INFO] Updated coco_path to: {args.coco_path}")
    else:
        if args.rank == 0:
            print(f"[INFO] Using SIXray test dataset: {args.coco_path}")

    # fix the seed for reproducibility
    # main.py와 동일한 2단계 seed 정책:
    #   1단계) 모델 init 전: 모든 rank 동일 seed → shared param(router 등) rank간 동일 초기화
    #   2단계) 데이터 로딩 전: rank별 다른 seed → 데이터 다양성 확보
    # (eval에서는 바로 checkpoint를 load해 파라미터를 덮어쓰므로 1단계 seed가
    #  모델 동작에 영향을 주지는 않지만, main.py와 철학을 통일)
    seed = getattr(args, 'seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else args.device)

    # build model (EP 활성화 시 local expert만 생성)
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)

    # load checkpoint weights
    ck = torch.load(args.checkpoint, map_location='cpu')
    model_without_ddp = model
    ck_state = ck['model'] if 'model' in ck else ck

    # EP: full checkpoint(8 experts) → 현재 rank의 local experts만 추출
    use_ep = getattr(args, 'moe_expert_parallel', False) and args.distributed
    if use_ep:
        from models.moe.expert_parallel import scatter_expert_state_dict
        ck_state = scatter_expert_state_dict(ck_state, model_without_ddp)
        if args.rank == 0:
            print("[INFO] Expert weights scattered to local ranks")

    model_without_ddp.load_state_dict(ck_state)

    # 데이터 로딩 전: rank별 다른 seed로 재설정 (main.py data_seed와 동일)
    data_seed = getattr(args, 'seed', 42) + utils.get_rank()
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # build test dataset
    dataset_test = build_dataset(image_set='test', args=args)

    # JAX V-MoE 방식: 모든 샘플을 모델에 통과 (MoE capacity 일관성),
    # padding 샘플은 _is_padding=True로 마킹 → metric 계산에서만 제외.
    moe_group = getattr(args, 'moe_group_images', 1)
    n_total = len(dataset_test)
    _sampler = _PaddedSampler(
        n_total, moe_group,
        rank=args.rank if args.distributed else 0,
        world_size=args.world_size if args.distributed else 1,
    )
    padded_view = _PaddedDatasetView(dataset_test, _sampler.indices, _sampler.n_real_local)
    data_loader_test = DataLoader(
        padded_view, batch_size=moe_group,
        sampler=torch.utils.data.SequentialSampler(padded_view),
        drop_last=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    base_ds = get_coco_api_from_dataset(dataset_test)

    os.makedirs(args.output_dir, exist_ok=True)

    # ensure common flags expected by `evaluate()` exist
    if not hasattr(args, 'debug'):
        args.debug = False
    if not hasattr(args, 'amp'):
        args.amp = False
    if not hasattr(args, 'print_freq'):
        args.print_freq = 10
    if not hasattr(args, 'save_log'):
        args.save_log = False

    # If user requested saving detection results, run an in-script inference
    # loop (avoid calling engine.test()). This writes a results JSON.
    if args.save_results:
        if args.rank == 0:
            print("Running inference loop to dump per-image detection results (results.json)")
        import json
        final_res = []
        model.eval()
        with torch.no_grad():
            for samples, targets in data_loader_test:
                samples = samples.to(device)
                targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

                outputs = model(samples)
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
                if 'segm' in postprocessors.keys():
                    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

                for target, out in zip(targets, results):
                    # padding 샘플은 MoE capacity용으로만 forward했으므로 결과 저장 제외
                    if target.get('_is_padding', torch.tensor(False)).item():
                        continue
                    image_id = target['image_id'].item()
                    _scores = out['scores']
                    _labels = out['labels']
                    _boxes = out['boxes']
                    if not isinstance(_scores, list):
                        _scores = _scores.tolist()
                    if not isinstance(_labels, list):
                        _labels = _labels.tolist()
                    if not isinstance(_boxes, list):
                        _boxes = _boxes.tolist()
                    for s, l, b in zip(_scores, _labels, _boxes):
                        itemdict = {
                            "image_id": int(image_id),
                            "category_id": int(l),
                            "bbox": b,
                            "score": float(s),
                        }
                        final_res.append(itemdict)

        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f'results{args.rank}.json'
        with open(outpath, 'w') as f:
            json.dump(final_res, f)
        print(f"[Rank {args.rank}] Saved detection json to {outpath}")

    # run evaluation; padding 샘플은 engine.py에서 _is_padding 플래그로 자동 제외 (JAX VALID_KEY 방식)
    orig_save_results = getattr(args, 'save_results', False)
    args.save_results = False
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessors,
        data_loader_test, base_ds, device,
        args.output_dir, wo_class_error=False, args=args
    )
    args.save_results = orig_save_results

    # 이하 rank 0에서만 결과 출력/저장
    if args.rank != 0:
        return

    print("Test stats:", test_stats)

    if coco_evaluator is None or "bbox" not in coco_evaluator.coco_eval:
        print("No coco_evaluator or bbox results found. Can't compute per-class AP.")
        return

    coco_eval_bbox = coco_evaluator.coco_eval["bbox"]

    # save raw coco eval dict
    savep = Path(args.output_dir) / "eval" / "test_latest.pth"
    savep.parent.mkdir(parents=True, exist_ok=True)
    torch.save(coco_eval_bbox.eval, savep)
    print("Saved coco eval to", savep)

    # ✅ 전체 메트릭 출력 및 저장
    if 'coco_eval_bbox' in test_stats:
        bbox_stats = test_stats['coco_eval_bbox']

        print("\n" + "="*70)
        print("Overall COCO Metrics:")
        print("="*70)
        print(f"  mAP@[0.50:0.95] (IoU=0.50:0.95, area=all, maxDets=100) = {bbox_stats[0]:.4f}")
        print(f"  mAP@0.50        (IoU=0.50,      area=all, maxDets=100) = {bbox_stats[1]:.4f}")
        print(f"  mAP@0.75        (IoU=0.75,      area=all, maxDets=100) = {bbox_stats[2]:.4f}")
        print(f"  mAR             (IoU=0.50:0.95, area=all, maxDets=100) = {bbox_stats[8]:.4f}")
        print("="*70 + "\n")

        # CSV로 저장
        import csv
        summary_csv = Path(args.output_dir) / "eval" / "overall_metrics.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["mAP@[0.50:0.95]", f"{bbox_stats[0]:.4f}"])
            writer.writerow(["mAP@0.50", f"{bbox_stats[1]:.4f}"])
            writer.writerow(["mAP@0.75", f"{bbox_stats[2]:.4f}"])
            writer.writerow(["mAP (small)", f"{bbox_stats[3]:.4f}"])
            writer.writerow(["mAP (medium)", f"{bbox_stats[4]:.4f}"])
            writer.writerow(["mAP (large)", f"{bbox_stats[5]:.4f}"])
            writer.writerow(["mAR@[0.50:0.95] (maxDets=1)", f"{bbox_stats[6]:.4f}"])
            writer.writerow(["mAR@[0.50:0.95] (maxDets=10)", f"{bbox_stats[7]:.4f}"])
            writer.writerow(["mAR@[0.50:0.95] (maxDets=100)", f"{bbox_stats[8]:.4f}"])
            writer.writerow(["mAR (small)", f"{bbox_stats[9]:.4f}"])
            writer.writerow(["mAR (medium)", f"{bbox_stats[10]:.4f}"])
            writer.writerow(["mAR (large)", f"{bbox_stats[11]:.4f}"])
        print(f"Saved overall metrics to {summary_csv}\n")

    # ✅ 클래스별 AP 출력/저장
    if args.per_class or args.save_per_class:
        per_class = _per_class_ap_from_cocoeval(
            coco_eval_bbox, base_ds, also_ap50_ap75=args.also_ap50_ap75
        )

        # print
        if args.per_class:
            if args.also_ap50_ap75:
                print("\nPer-class AP:")
                print(f"{'Category':25s} {'ID':>6s} {'AP50:95':>8s} {'AP50':>8s} {'AP75':>8s}")
                print("-"*70)
                for r in per_class:
                    print(f"{r['name'][:25]:25s} {r['cat_id']:6d} {r['ap_50_95']:8.4f} {r['ap50']:8.4f} {r['ap75']:8.4f}")
            else:
                print("\nPer-class AP (mAP@[0.50:0.95]):")
                print(f"{'Category':25s} {'ID':>6s} {'AP50:95':>8s}")
                print("-"*70)
                for r in per_class:
                    print(f"{r['name'][:25]:25s} {r['cat_id']:6d} {r['ap_50_95']:8.4f}")

        # save CSV
        if args.save_per_class:
            import csv
            csv_path = Path(args.output_dir) / "eval" / "per_class_ap.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(per_class[0].keys()) if per_class else ["cat_id", "name", "ap_50_95"]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(per_class)
            print(f"Saved per-class AP to {csv_path}\n")


if __name__ == '__main__':
    main()
