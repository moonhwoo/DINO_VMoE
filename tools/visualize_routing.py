"""MoE Router 시각화 모듈.

학습 중 자동 호출 (main.py):
    run_routing_visualization(model, dataset_val, image_ids, epoch, output_dir, device)

학습 후 체크포인트로 단독 실행:
    python tools/visualize_routing.py \\
        -c configs/DINO_4scale_swin_moe.py \\
        --checkpoint output/checkpoint_best_regular.pth \\
        --coco_path /path/to/coco \\
        --image_ids 199 1006 6462 7714  5 8 33 52  429 449 458 573 \\
        --output_dir output/vis_test

# 선택 이미지 12장 (SIXray-D val 기준, moe_group_images=4 기준 3그룹):
#   그룹1 (기존): id=199 Knife  /  id=1006 Gun+Pliers  /  id=6462 Wrench+Pliers  /  id=7714 Knife
#   그룹2:        id=5   Knife  /  id=8    Knife        /  id=33   Gun+Knife      /  id=52   Gun
#   그룹3:        id=429 Wrench /  id=449  Wrench       /  id=458  Wrench+Pliers  /  id=573  Scissors
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
import matplotlib
matplotlib.use('Agg')   # GUI 없는 서버 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# expert 색상 팔레트 (tab10, 최대 10개 expert)
EXPERT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3]   # (10, 3) RGB 0~1


# ── capture_routing 플래그 제어 ──────────────────────────────────────

def enable_capture(model):
    """모든 DINOMoEBlock.capture_routing = True.
    이 순간부터 forward() 시 route_map_top1/top2, overflow_map이 metrics에 저장됨."""
    from models.moe.dino_moe_block import DINOMoEBlock
    for m in model.modules():
        if isinstance(m, DINOMoEBlock):
            m.capture_routing = True


def disable_capture(model):
    """모든 DINOMoEBlock.capture_routing = False.
    학습 배치에는 이 상태를 유지 → route_map 계산 오버헤드 없음."""
    from models.moe.dino_moe_block import DINOMoEBlock
    for m in model.modules():
        if isinstance(m, DINOMoEBlock):
            m.capture_routing = False


# ── 메인 시각화 함수 ─────────────────────────────────────────────────

def run_routing_visualization(model, dataset_val, image_ids, epoch,
                               output_dir, device, tag='',
                               postprocessors=None,
                               match_score_thr=0.05, vis_score_thr=0.3, iou_thr=0.5):
    """고정 이미지들로 routing 시각화 생성 및 저장.

    main.py에서 5-epoch 윈도우 체크 시 호출 흐름:
      1. 현재 학습 가중치 백업 (main.py에서 수행)
      2. checkpoint_best_regular.pth 가중치 로드 (main.py에서 수행)
      3. 이 함수 호출 → capture_routing=True → 고정 이미지 forward → 그림 저장
      4. 학습 가중치 복원 (main.py에서 수행)

    Args:
        model:       model_without_ddp (DDP 래퍼 제거된 모델)
        dataset_val: val 데이터셋 (CocoDetection)
        image_ids:   COCO val image ID 리스트 (예: [139, 285, 632, 1000])
        epoch:       현재 best epoch 번호 (폴더명, 그림 제목에 사용)
        output_dir:  저장 루트 디렉토리
        device:      torch.device
        tag:         폴더명 접미사 (예: 'ep3_map0.423')

    저장 경로:
        output_dir/routing_vis/epoch_XXXX_TAG/img_YYYYYY_top1.png
                                               img_YYYYYY_top2.png
    """
    from util.misc import nested_tensor_from_tensor_list

    was_training = model.training
    model.eval()
    enable_capture(model)

    folder = f'epoch_{epoch:04d}' + (f'_{tag}' if tag else '')
    save_dir = Path(output_dir) / 'routing_vis' / folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # moe_group_images 값 조회 (학습과 동일한 그룹 크기로 배치 구성)
    moe_group = 1
    for m in model.modules():
        if hasattr(m, 'moe_group_images'):
            moe_group = m.moe_group_images
            break

    # 모든 유효 이미지 수집
    valid_entries = []  # list of (img_id, img_tensor, orig_img)
    for img_id in image_ids:
        try:
            idx = dataset_val.ids.index(img_id)
        except ValueError:
            print(f'[vis] image_id={img_id} not in dataset, skip')
            continue
        img_tensor, _ = dataset_val[idx]
        orig_img = _load_original_pil(dataset_val, idx)
        valid_entries.append((img_id, img_tensor, orig_img))

    if not valid_entries:
        disable_capture(model)
        if was_training:
            model.train()
        return

    with torch.no_grad():
        # moe_group 단위로 forward — 학습과 동일한 그룹 구조 유지
        i = 0
        summary = {'correct': 0, 'partial': 0, 'incorrect': 0, 'unknown': 0}
        while i < len(valid_entries):
            chunk = list(valid_entries[i:i + moe_group])
            # 마지막 배치가 moe_group보다 작으면 마지막 이미지 반복 패딩
            while len(chunk) < moe_group:
                chunk.append(chunk[-1])

            tensors = [e[1] for e in chunk]
            samples = nested_tensor_from_tensor_list(tensors).to(device)

            # forward: route_map이 _moe_metrics에 저장됨
            outputs = model(samples)

            moe_metrics = getattr(model.transformer, '_moe_metrics', [])
            if not moe_metrics:
                print('[vis] _moe_metrics 없음 — capture_routing이 동작하지 않음')
                break

            # chunk 내 실제 이미지(패딩 제외)만 저장
            real_count = min(moe_group, len(valid_entries) - i)
            coco_api = getattr(dataset_val, 'coco', None)

            # postprocessors로 예측 결과 계산 (원본 이미지 크기 기준)
            pred_results = [None] * len(chunk)
            if postprocessors is not None and 'bbox' in postprocessors and outputs is not None:
                # PIL.size = (W, H) → postprocessor expects (H, W)
                chunk_orig_sizes = torch.tensor(
                    [[e[2].size[1], e[2].size[0]] for e in chunk],
                    dtype=torch.float32, device=device)
                pp_outputs = {k: v for k, v in outputs.items()
                              if k in ('pred_logits', 'pred_boxes')}
                pred_results = postprocessors['bbox'](pp_outputs, chunk_orig_sizes)

            for b in range(real_count):
                img_id, _, orig_img = valid_entries[i + b]
                pred_result = pred_results[b]
                # 분류: 낮은 threshold로 매칭 (COCO eval처럼 score 전체를 봄)
                det_info = _classify_detection(
                    pred_result, coco_api, img_id,
                    score_thr=match_score_thr, iou_thr=iou_thr)
                summary[det_info['label']] = summary.get(det_info['label'], 0) + 1
                for which in ('top1', 'top2'):
                    _save_figure(orig_img, moe_metrics, img_id, epoch, save_dir, which,
                                 batch_idx=b, coco_api=coco_api, use_overlay=True,
                                 pred_result=pred_result, det_info=det_info,
                                 vis_score_thr=vis_score_thr)
                    _save_figure(orig_img, moe_metrics, img_id, epoch, save_dir, which,
                                 batch_idx=b, coco_api=coco_api, use_overlay=False,
                                 pred_result=pred_result, det_info=det_info,
                                 vis_score_thr=vis_score_thr)

            i += moe_group

    disable_capture(model)
    if was_training:
        model.train()

    # torchrun 환경에서 rank 0만 출력 (모든 rank가 같은 메시지를 찍지 않도록)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f'[vis] routing visualization saved → {save_dir}')
        print(f'[vis] detection summary: correct={summary["correct"]} '
              f'partial={summary["partial"]} incorrect={summary["incorrect"]} '
              f'unknown={summary["unknown"]}')


# ── 내부 헬퍼 ────────────────────────────────────────────────────────

def _load_original_pil(dataset_val, idx):
    """transforms 없이 원본 PIL 이미지 로드."""
    img_id   = dataset_val.ids[idx]
    filename = dataset_val.coco.loadImgs(img_id)[0]['file_name']
    path     = os.path.join(dataset_val.root, filename)
    return Image.open(path).convert('RGB')


def _draw_bboxes(ax, coco_api, img_id, matched_gt_indices=None):
    """COCO GT bbox 그리기.

    matched_gt_indices: set of GT annotation indices that were matched to a prediction.
        None → 구분 없이 모두 yellow.
        set  → matched=limegreen, missed=red.
    """
    if coco_api is None:
        return
    ann_ids = coco_api.getAnnIds(imgIds=img_id)
    anns    = coco_api.loadAnns(ann_ids)
    cats    = {c['id']: c['name'] for c in coco_api.loadCats(coco_api.getCatIds())}
    for i, ann in enumerate(anns):
        x, y, w, h = ann['bbox']
        cat_name    = cats.get(ann['category_id'], str(ann['category_id']))
        if matched_gt_indices is None:
            color = 'yellow'
        else:
            color = 'limegreen' if i in matched_gt_indices else 'red'
        rect = plt.Rectangle((x, y), w, h,
                              linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 2, cat_name, fontsize=5, color=color,
                verticalalignment='bottom',
                bbox=dict(facecolor='black', alpha=0.4, pad=1, linewidth=0))


def _draw_predictions(ax, pred_result, score_thr=0.3, coco_api=None, det_info=None):
    """예측 bbox 그리기.

    - 매칭된 예측 (det_info['matched_pred_boxes']): 주황색 실선 — score 관계없이 항상 표시
    - 그 외 score >= score_thr 예측: lime 점선
    """
    if pred_result is None:
        return
    cats = {}
    if coco_api is not None:
        cats = {i: c['name'] for i, c in enumerate(
            coco_api.loadCats(coco_api.getCatIds()))}

    # 1) 매칭된 예측 박스: 주황 실선 (GT와 실제로 매칭된 것, score 무관하게 표시)
    if det_info and det_info.get('matched_pred_boxes'):
        for box, score in zip(det_info['matched_pred_boxes'], det_info['matched_pred_scores']):
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2.0, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, f'matched:{score:.2f}', fontsize=5, color='orange',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1, linewidth=0))

    # 2) score >= vis_score_thr 예측: lime 점선 (매칭 여부 무관하게 표시)
    scores = pred_result['scores']
    labels = pred_result['labels']
    boxes  = pred_result['boxes']
    keep   = scores >= score_thr
    for score, lbl, box in zip(scores[keep], labels[keep], boxes[keep]):
        x1, y1, x2, y2 = box.tolist()
        cat_name = cats.get(lbl.item(), f'cls{lbl.item()}')
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=1.2, edgecolor='lime', facecolor='none',
                              linestyle='--')
        ax.add_patch(rect)
        ax.text(x2, y2 + 2, f'{cat_name}:{score:.2f}', fontsize=5, color='lime',
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.4, pad=1, linewidth=0))


def _classify_detection(pred_result, coco_api, img_id, score_thr=0.3, iou_thr=0.5):
    """이미지 1장에 대한 detection 품질 평가.

    Returns:
        dict with:
            label:             'correct' | 'partial' | 'incorrect' | 'unknown'
            n_gt:              GT 객체 수
            n_matched:         matched GT 수
            recall:            n_matched / n_gt
            matched_gt_indices: GT annotation 인덱스 중 matched된 것의 set
    """
    from util.box_ops import box_iou as box_iou_fn

    default = {'label': 'unknown', 'n_gt': 0, 'n_matched': 0,
               'recall': 0.0, 'matched_gt_indices': set(),
               'matched_pred_boxes': [], 'matched_pred_scores': []}
    if coco_api is None or pred_result is None:
        return default

    ann_ids = coco_api.getAnnIds(imgIds=img_id)
    anns    = coco_api.loadAnns(ann_ids)
    if not anns:
        return {**default, 'label': 'correct', 'recall': 1.0}

    # GT boxes: xywh → xyxy pixel
    gt_boxes = torch.tensor(
        [[a['bbox'][0], a['bbox'][1],
          a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns],
        dtype=torch.float32)

    # 예측 필터
    scores = pred_result['scores'].cpu()
    boxes  = pred_result['boxes'].cpu()
    keep   = scores >= score_thr
    boxes  = boxes[keep]

    if len(boxes) == 0:
        return {**default, 'label': 'incorrect', 'n_gt': len(anns)}

    # IoU (N_pred, N_gt)
    iou_mat, _ = box_iou_fn(boxes, gt_boxes)

    # GT별로 최대 IoU >= iou_thr인 예측을 greedy 매칭
    matched_gt_indices  = set()
    matched_pred_boxes  = []   # 실제 GT와 매칭된 예측 박스 (score 무관, 주황색 표시용)
    matched_pred_scores = []

    # score 내림차순 정렬 후 순차 매칭 (COCO eval과 동일한 greedy 방식)
    filtered_scores = pred_result['scores'].cpu()[keep]
    order = filtered_scores.argsort(descending=True)
    used_gt = set()
    for pred_idx in order:
        iou_for_pred = iou_mat[pred_idx]          # (N_gt,)
        best_gt = iou_for_pred.argmax().item()
        if iou_for_pred[best_gt].item() >= iou_thr and best_gt not in used_gt:
            used_gt.add(best_gt)
            matched_gt_indices.add(best_gt)
            matched_pred_boxes.append(boxes[pred_idx])
            matched_pred_scores.append(filtered_scores[pred_idx].item())

    n_matched = len(matched_gt_indices)
    n_gt      = len(anns)
    recall    = n_matched / n_gt

    if n_matched == n_gt:
        label = 'correct'
    elif n_matched == 0:
        label = 'incorrect'
    else:
        label = 'partial'

    return {'label': label, 'n_gt': n_gt, 'n_matched': n_matched,
            'recall': recall, 'matched_gt_indices': matched_gt_indices,
            'matched_pred_boxes': matched_pred_boxes,
            'matched_pred_scores': matched_pred_scores}


def _make_expert_rgb(rm, E):
    """routing map (H,W) → expert 색상 RGB numpy (H,W,3)."""
    H, W = rm.shape
    rgb = np.ones((H, W, 3)) * 0.85
    for e in range(E):
        mask_e = (rm == e)
        if mask_e.any():
            rgb[mask_e] = EXPERT_COLORS[e % 10]
    return rgb


def _valid_bounds(rm):
    """rm != -1 인 유효 영역의 (h_end, w_end) 반환."""
    valid_mask = (rm != -1)
    valid_rows = np.where(valid_mask.any(axis=1))[0]
    valid_cols = np.where(valid_mask.any(axis=0))[0]
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return rm.shape[0], rm.shape[1]
    return int(valid_rows[-1]) + 1, int(valid_cols[-1]) + 1


def _overlay_on_image(orig_img, expert_rgb, rm, alpha=0.35):
    """routing map의 유효 영역(rm != -1)을 원본 이미지 크기로 업스케일 후 오버레이.

    원본 이미지가 주인공(1-alpha), expert 색이 반투명 틴트(alpha).
    반환: blended numpy (H_orig, W_orig, 3), h_end, w_end (feature map 좌표)
    """
    orig_w, orig_h = orig_img.size          # PIL: (W, H)
    orig_np = np.array(orig_img) / 255.0    # (H_orig, W_orig, 3)

    h_end, w_end = _valid_bounds(rm)

    # 유효 영역 expert_rgb를 원본 해상도로 업스케일 (nearest → 블록 경계 선명하게)
    expert_valid = expert_rgb[:h_end, :w_end]   # (h_end, w_end, 3)
    expert_pil   = Image.fromarray((expert_valid * 255).astype(np.uint8))
    expert_up    = np.array(expert_pil.resize((orig_w, orig_h), Image.NEAREST)) / 255.0

    blended = (1 - alpha) * orig_np + alpha * expert_up
    return blended, h_end, w_end


def _save_figure(orig_img, moe_metrics, img_id, epoch, save_dir, which,
                 batch_idx=0, coco_api=None, use_overlay=True,
                 pred_result=None, det_info=None, vis_score_thr=0.3):
    """Figure 한 장 생성 및 저장.

    use_overlay=True  → 원본 이미지 위에 alpha=0.60 반투명 expert 색상 오버레이
    use_overlay=False → 순수 expert 컬러맵 (기존 스타일)
    pred_result       → postprocessor 출력 (scores, labels, boxes). None이면 생략.
    det_info          → _classify_detection 결과 dict. None이면 분류 없이 루트에 저장.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    map_key = f'route_maps_{which}'

    valid_layers = [
        (i, m) for i, m in enumerate(moe_metrics)
        if m is not None and map_key in m
    ]
    if not valid_layers:
        return

    num_layers = len(valid_layers)
    num_scales = len(valid_layers[0][1][map_key])
    num_cols   = 1 + num_scales

    fig, axes = plt.subplots(
        num_layers, num_cols,
        figsize=(num_cols * 3, num_layers * 3 + 0.5),
        squeeze=False)

    num_experts = None

    for row, (layer_idx, metrics) in enumerate(valid_layers):
        route_maps    = metrics[map_key]
        per_assign_key = f'overflow_maps_{which}'
        overflow_maps = metrics.get(per_assign_key,
                        metrics.get('overflow_maps', [None] * num_scales))
        tpe           = metrics.get('tokens_per_expert', None)
        overflow_rate = metrics.get('overflow_rate', None)
        load_cv       = metrics.get('load_cv', None)

        if num_experts is None and tpe is not None:
            num_experts = int(tpe.shape[0])

        # 원본 이미지 컬럼 — GT bbox (matched=초록/missed=빨강) + 예측 bbox (lime 점선)
        matched_indices = det_info['matched_gt_indices'] if det_info else None
        axes[row, 0].imshow(orig_img)
        _draw_bboxes(axes[row, 0], coco_api, img_id, matched_gt_indices=matched_indices)
        _draw_predictions(axes[row, 0], pred_result, score_thr=vis_score_thr,
                          coco_api=coco_api, det_info=det_info)
        axes[row, 0].axis('off')
        actual_layer_id = metrics.get('layer_id', layer_idx)
        axes[row, 0].set_ylabel(f'Enc Layer {actual_layer_id}', fontsize=8,
                                rotation=90, labelpad=4, va='center')
        if row == 0:
            col_title = 'GT(green=hit,red=miss) + pred(lime--)'
            axes[row, 0].set_title(col_title, fontsize=6)

        # scale별 컬럼 — expert 색상을 원본 이미지 위에 오버레이
        for col, lvl in enumerate(range(num_scales)):
            ax  = axes[row, col + 1]
            rm  = route_maps[lvl][batch_idx].numpy()
            H, W = rm.shape
            E   = num_experts if num_experts else 8

            expert_rgb = _make_expert_rgb(rm, E)

            if use_overlay:
                blended, h_end, w_end = _overlay_on_image(
                    orig_img, expert_rgb, rm, alpha=0.60)
                ax.imshow(blended, interpolation='nearest', aspect='auto')
            else:
                ax.imshow(expert_rgb, interpolation='nearest', aspect='auto')
                h_end, w_end = _valid_bounds(rm)

            # 완전 drop된 토큰 — 빨간 X
            if lvl < len(overflow_maps) and overflow_maps[lvl] is not None:
                om = overflow_maps[lvl][batch_idx].numpy()
                ys, xs = om.nonzero()
                if len(ys) > 0:
                    if use_overlay:
                        orig_w, orig_h = orig_img.size
                        xs_sc = (xs + 0.5) * orig_w / w_end
                        ys_sc = (ys + 0.5) * orig_h / h_end
                    else:
                        xs_sc = xs.astype(float)
                        ys_sc = ys.astype(float)
                    ax.scatter(xs_sc, ys_sc, marker='x', c='red',
                               s=30, linewidths=1.0, alpha=0.9)

            ax.axis('off')
            if row == 0:
                ax.set_title(f'scale{lvl} ({W}×{H})', fontsize=7)

        # 오른쪽 통계 텍스트
        stat_parts = []
        if overflow_rate is not None:
            stat_parts.append(f'overflow(total)={overflow_rate.item() * 100:.1f}%')
        ovf_top1 = metrics.get('overflow_rate_top1', None)
        ovf_top2 = metrics.get('overflow_rate_top2', None)
        if ovf_top1 is not None and ovf_top2 is not None:
            stat_parts.append(f'  top1={ovf_top1.item() * 100:.1f}%  top2={ovf_top2.item() * 100:.1f}%')
        if load_cv is not None:
            stat_parts.append(f'load_cv={load_cv.item():.3f}')
        lcv_top1 = metrics.get('load_cv_top1', None)
        lcv_top2 = metrics.get('load_cv_top2', None)
        if lcv_top1 is not None and lcv_top2 is not None:
            stat_parts.append(f'  top1={lcv_top1.item():.3f}  top2={lcv_top2.item():.3f}')
        tpe_top1 = metrics.get('tokens_per_expert_top1', None)
        tpe_top2 = metrics.get('tokens_per_expert_top2', None)
        if tpe is not None:
            tpe_str = '  '.join([f'E{e}:{int(tpe[e])}' for e in range(len(tpe))])
            stat_parts.append(f'[total] {tpe_str}')
        if tpe_top1 is not None:
            tpe1_str = '  '.join([f'E{e}:{int(tpe_top1[e])}' for e in range(len(tpe_top1))])
            stat_parts.append(f'[top1]  {tpe1_str}')
        if tpe_top2 is not None:
            tpe2_str = '  '.join([f'E{e}:{int(tpe_top2[e])}' for e in range(len(tpe_top2))])
            stat_parts.append(f'[top2]  {tpe2_str}')
        if stat_parts:
            axes[row, -1].text(
                1.03, 0.5, '\n'.join(stat_parts),
                transform=axes[row, -1].transAxes,
                fontsize=6, va='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 범례: expert 색상
    if num_experts:
        patches = [
            mpatches.Patch(color=EXPERT_COLORS[e % 10], label=f'E{e}')
            for e in range(num_experts)
        ]
        fig.legend(handles=patches, loc='lower center',
                   ncol=num_experts, fontsize=7, bbox_to_anchor=(0.5, 0.0))

    style = 'overlay' if use_overlay else 'pure'

    # 제목: detection 결과 포함
    if det_info and det_info['label'] != 'unknown':
        det_str = (f"{det_info['label'].upper()}  "
                   f"GT:{det_info['n_gt']}  hit:{det_info['n_matched']}  "
                   f"recall:{det_info['recall']:.2f}")
    else:
        det_str = ''
    title = f'Epoch {epoch:04d} | img_id={img_id} | {which.upper()} | {style}'
    if det_str:
        title += f'\n{det_str}'
    fig.suptitle(title, fontsize=9)

    # 저장 경로: det_info에 따라 서브폴더 분류
    split = det_info['label'] if det_info else None
    if split and split != 'unknown':
        actual_save_dir = save_dir / split
    else:
        actual_save_dir = save_dir
    actual_save_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = actual_save_dir / f'img_{img_id:06d}_{which}_{style}.png'
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── 독립 실행 모드 ────────────────────────────────────────────────────
#
# [단일 GPU]
#   python tools/visualize_routing.py -c ... --checkpoint ... ...
#
# [멀티 GPU / Expert Parallelism - 학습과 동일한 방식]
#   torchrun --nproc_per_node=2 tools/visualize_routing.py -c ... --checkpoint ... ...
#
# torchrun 환경에서:
#   1. dist.init_process_group: GPU간 통신 채널 열기 (all-to-all 가능하게 됨)
#   2. init_expert_parallel:    EP 상태 초기화 (_ep_world_size = 실제 GPU 수로 설정)
#   3. scatter_expert_state_dict: 전체 8 expert → 각 GPU에 4 expert씩 분배
#   4. 모든 GPU가 같은 이미지를 forward → all-to-all 정상 수행
#   5. 그림 저장은 rank 0만 (_save_figure에서 rank != 0이면 즉시 return)

if __name__ == '__main__':
    # tools/ 디렉토리에서 실행하므로 프로젝트 루트를 sys.path에 추가
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from util.slconfig import SLConfig
    from models.registry import MODULE_BUILD_FUNCS
    from datasets import build_dataset

    parser = argparse.ArgumentParser(description='MoE routing 시각화 (단독/멀티GPU 실행)')
    parser.add_argument('-c', '--config_file', required=True,
                        help='학습에 사용한 config 파일')
    parser.add_argument('--checkpoint', required=True,
                        help='시각화할 체크포인트 (checkpoint_best_regular.pth 등)')
    parser.add_argument('--coco_path', required=True,
                        help='데이터셋 루트 경로')
    parser.add_argument('--image_ids', type=int, nargs='+', required=True,
                        help='val image ID 목록 (예: 199 1006 6462 7714  5 8 33 52  429 449 458 573)')
    parser.add_argument('--output_dir', default='output/vis',
                        help='그림 저장 디렉토리')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset_file', default='coco',
                        help='데이터셋 종류 (default: coco)')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--match_score_thr', type=float, default=0.05,
                        help='GT 매칭용 score threshold — 낮게 설정해 COCO eval처럼 대부분 예측을 봄 (default: 0.05)')
    parser.add_argument('--vis_score_thr', type=float, default=0.3,
                        help='화면에 표시할 예측 박스 threshold — 높게 설정해 시각화 깔끔하게 (default: 0.3)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='GT 매칭 IoU threshold (default: 0.5)')
    args_vis = parser.parse_args()

    # ── 1. 분산 환경 감지 및 초기화 ──────────────────────────────────
    # torchrun은 환경 변수 WORLD_SIZE / LOCAL_RANK / RANK를 자동으로 설정함.
    # python으로 직접 실행하면 이 변수들이 없으므로 단일 GPU 모드로 동작.
    is_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1

    if is_distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        # GPU간 통신 채널 열기 (all-to-all이 이 그룹 위에서 동작)
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()

        # EP 상태 초기화: _ep_world_size를 실제 GPU 수로 설정
        # → DINOMoEBlock이 local_num_experts = num_experts // world_size 를 인식
        from models.moe.expert_parallel import init_expert_parallel
        init_expert_parallel()
    else:
        local_rank = 0
        rank = 0

    device = torch.device(f'cuda:{local_rank}')

    # ── 2. config 로드 ────────────────────────────────────────────────
    cfg = SLConfig.fromfile(args_vis.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        setattr(args_vis, k, v)
    if not hasattr(args_vis, 'fix_size'):
        args_vis.fix_size = False

    # ── 3. 모델 빌드 ──────────────────────────────────────────────────
    # is_distributed=True 이면 init_expert_parallel()로 _ep_world_size가 설정되어
    # DINOMoEBlock이 local_num_experts = num_experts // world_size 개만 생성함.
    # is_distributed=False 이면 _ep_world_size=1 → 8 expert 전부 단일 GPU에 올림.
    build_func = MODULE_BUILD_FUNCS.get(args_vis.modelname)
    model, _, postprocessors = build_func(args_vis)
    model.to(device)

    # ── 4. 체크포인트 로드 ────────────────────────────────────────────
    ckpt = torch.load(args_vis.checkpoint, map_location='cpu')
    model_sd = ckpt['model']

    use_ep = getattr(args_vis, 'moe_expert_parallel', False) and is_distributed
    if use_ep:
        # gather_expert_state_dict로 저장된 full checkpoint(8 expert)를
        # 각 GPU의 local expert 슬라이스로 분배.
        # GPU 0: expert 0~3, GPU 1: expert 4~7 (2GPU 8expert 예시)
        from models.moe.expert_parallel import scatter_expert_state_dict
        model_sd = scatter_expert_state_dict(model_sd, model)

    model.load_state_dict(model_sd)
    epoch = ckpt.get('epoch', 0)
    if rank == 0:
        print(f'[vis] loaded checkpoint: epoch={epoch}, EP={use_ep}, world_size={dist.get_world_size() if is_distributed else 1}')

    # ── 5. val 데이터셋 빌드 ──────────────────────────────────────────
    # 모든 rank가 동일한 image_ids를 로드하여 forward.
    # all-to-all은 모든 rank가 같은 이미지를 처리할 때 정상 동작.
    dataset_val = build_dataset(image_set='val', args=args_vis)

    # ── 6. 시각화 실행 ────────────────────────────────────────────────
    run_routing_visualization(
        model, dataset_val, args_vis.image_ids,
        epoch, args_vis.output_dir, device, tag='standalone',
        postprocessors=postprocessors,
        match_score_thr=args_vis.match_score_thr,
        vis_score_thr=args_vis.vis_score_thr,
        iou_thr=args_vis.iou_threshold)

    # ── 7. 분산 종료 ──────────────────────────────────────────────────
    if is_distributed:
        dist.barrier()          # 모든 rank가 시각화 완료할 때까지 대기
        dist.destroy_process_group()
