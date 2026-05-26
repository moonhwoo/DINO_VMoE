"""
soft_labels 검증 시각화 스크립트

각 scale(stride 8, 16, 32, 64)에서 GT 박스가 토큰 그리드 위에
올바르게 올라가는지 시각적으로 확인한다.

사용법:
    python tools/verify_soft_labels.py
    python tools/verify_soft_labels.py --img_ids 9 11 --out_dir outputs/soft_label_check
"""

import argparse
import json
import math
import os

import matplotlib  # noqa: E402 (os 먼저 필요)
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 한글 폰트 설정 (NotoSansCJK 사용)
_KO_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_KO_FONT_PATH):
    fm.fontManager.addfont(_KO_FONT_PATH)
    _prop = fm.FontProperties(fname=_KO_FONT_PATH)
    matplotlib.rcParams['font.family'] = _prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────
# 1. _compute_soft_labels 로직 (dino_moe_block.py와 동일)
# ─────────────────────────────────────────────────────────────

def compute_soft_labels(gt_boxes_list, gt_labels_list, spatial_shapes, num_classes,
                        valid_ratios=None, device='cpu'):
    """
    Args:
        gt_boxes_list  : list[Tensor(n_i, 4)]  cxcywh 정규화 좌표 (원본 이미지 기준)
        gt_labels_list : list[Tensor(n_i,)]    클래스 인덱스
        spatial_shapes : Tensor(n_levels, 2)   각 scale의 (H, W)  (padded 이미지 기준)
        num_classes    : int
        valid_ratios   : Tensor(B, n_levels, 2) or None
                         [b, l, 0] = W_valid/W_pad, [b, l, 1] = H_valid/H_pad
                         None이면 valid_ratio=1.0으로 처리 (단일 이미지 테스트 등)
        device         : str

    Returns:
        soft_labels : (B, N_total, C)
        fg_mask     : (B, N_total)   True = foreground 토큰
        token_xy    : list[(N_l, 2)] 각 scale의 토큰 중심 (x, y) padded 기준 정규화 좌표
    """
    spatial_shapes = spatial_shapes.to(device)
    B = len(gt_boxes_list)
    C = num_classes

    token_x_list, token_y_list, token_xy = [], [], []
    for l in range(spatial_shapes.shape[0]):
        H_l = int(spatial_shapes[l, 0])
        W_l = int(spatial_shapes[l, 1])
        ys = (torch.arange(H_l, device=device).float() + 0.5) / H_l
        xs = (torch.arange(W_l, device=device).float() + 0.5) / W_l
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        token_x_list.append(xx.reshape(-1))
        token_y_list.append(yy.reshape(-1))
        token_xy.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))  # (H*W, 2)

    N_per_level = [tx.shape[0] for tx in token_x_list]
    N = sum(N_per_level)

    soft_labels = torch.zeros(B, N, C, device=device)
    fg_mask     = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b in range(B):
        boxes  = gt_boxes_list[b].to(device)
        labels = gt_labels_list[b].to(device)
        if len(boxes) == 0:
            continue

        # cxcywh → xyxy (원본 이미지 기준 정규화)
        x1_orig = boxes[:, 0] - boxes[:, 2] / 2
        y1_orig = boxes[:, 1] - boxes[:, 3] / 2
        x2_orig = boxes[:, 0] + boxes[:, 2] / 2
        y2_orig = boxes[:, 1] + boxes[:, 3] / 2

        in_box_parts = []
        for l, (tx, ty) in enumerate(zip(token_x_list, token_y_list)):
            # valid_ratios로 GT 좌표를 padded 이미지 기준으로 변환
            if valid_ratios is not None:
                vw = valid_ratios[b, l, 0]
                vh = valid_ratios[b, l, 1]
                x1 = x1_orig * vw
                y1 = y1_orig * vh
                x2 = x2_orig * vw
                y2 = y2_orig * vh
            else:
                x1, y1, x2, y2 = x1_orig, y1_orig, x2_orig, y2_orig

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

        labels_oh = F.one_hot(labels.clamp(0, C - 1), C).float()   # (n_b, C)
        token_class_counts = in_box.float() @ labels_oh              # (N, C)
        soft_labels[b] = (
            token_class_counts /
            n_boxes_per_token.clamp(min=1.0).unsqueeze(1)
        )

    return soft_labels, fg_mask, token_xy


# ─────────────────────────────────────────────────────────────
# 2. DINO 스타일 이미지 전처리 (resize only, 비율 유지)
# ─────────────────────────────────────────────────────────────

def resize_keep_ratio(img, max_size=512):
    """PIL Image를 비율 유지하면서 최장변 max_size 이하로 resize."""
    w, h = img.size
    scale = min(max_size / max(h, w), 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR), scale


def get_spatial_shapes(img_h, img_w, strides=(8, 16, 32, 64)):
    """이미지 크기 → 각 stride별 feature map (H, W) 계산."""
    shapes = []
    for s in strides:
        H = math.ceil(img_h / s)
        W = math.ceil(img_w / s)
        shapes.append((H, W))
    return shapes


# ─────────────────────────────────────────────────────────────
# 3. 시각화
# ─────────────────────────────────────────────────────────────

# 클래스별 색상 (RGBA)
CLASS_COLORS = [
    (1.0, 0.2, 0.2, 0.6),   # Gun   - 빨강
    (0.2, 0.6, 1.0, 0.6),   # Knife - 파랑
    (0.2, 0.9, 0.2, 0.6),   # Wrench - 초록
    (1.0, 0.7, 0.1, 0.6),   # Pliers - 주황
    (0.8, 0.2, 0.9, 0.6),   # Scissors - 보라
]
CLASS_NAMES = ['Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors']
BOX_EDGE_COLORS = ['red', 'blue', 'green', 'orange', 'purple']


def draw_gt_boxes_on_image(ax, img_np, boxes_xyxy_pixel, labels, title='원본 이미지'):
    """원본 이미지 위에 GT 박스를 그린다 (픽셀 좌표)."""
    ax.imshow(img_np)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    h, w = img_np.shape[:2]
    for box, label in zip(boxes_xyxy_pixel, labels):
        x1, y1, x2, y2 = box
        color = BOX_EDGE_COLORS[label % len(BOX_EDGE_COLORS)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, CLASS_NAMES[label], fontsize=7,
                color=color, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))


def draw_scale_fg_mask(ax, soft_labels_l, fg_mask_l, H_l, W_l,
                       boxes_norm, labels, scale_idx, img_np, vw=1.0, vh=1.0, stride=None):
    """
    한 scale에서 foreground 토큰을 컬러로 시각화.

    vw, vh: valid_ratios (img_w/pad_size, img_h/pad_size).
            패딩 없으면 1.0. 이 값으로 GT 박스 좌표 변환과
            배경 이미지 배치 영역을 결정한다.

    - 배경: 회색 캔버스(패딩 영역) + 실제 이미지는 valid 영역에만 배치
    - 점선: valid_ratios 적용한 padded feature map 기준 GT 박스 경계
    - 컬러 오버레이: fg 토큰의 dominant class 색상
    """
    # 배경: 회색 캔버스 위에 이미지를 valid 영역에만 배치
    valid_h = max(1, int(round(vh * H_l)))
    valid_w = max(1, int(round(vw * W_l)))
    canvas = np.full((H_l, W_l, 3), 180, dtype=np.uint8)  # 회색 = 패딩 영역
    img_valid = np.array(Image.fromarray(img_np).resize((valid_w, valid_h), Image.BILINEAR))
    canvas[:valid_h, :valid_w] = img_valid
    ax.imshow(canvas)

    # valid 영역 경계선 (흰색 점선)
    border = patches.Rectangle(
        (-0.5, -0.5), valid_w, valid_h,
        linewidth=1, edgecolor='white', facecolor='none', linestyle=':')
    ax.add_patch(border)

    # foreground 토큰 오버레이
    sl = soft_labels_l.reshape(H_l, W_l, -1).numpy()   # (H, W, C)
    fg = fg_mask_l.reshape(H_l, W_l).numpy()            # (H, W)

    overlay = np.zeros((H_l, W_l, 4), dtype=np.float32)
    fg_y, fg_x = np.where(fg)
    for y, x in zip(fg_y, fg_x):
        c = sl[y, x].argmax()
        overlay[y, x] = CLASS_COLORS[c % len(CLASS_COLORS)]

    ax.imshow(overlay, origin='upper', extent=(-0.5, W_l - 0.5, H_l - 0.5, -0.5))

    # GT 박스 경계: boxes_norm(resized 이미지 기준 정규화) → padded feature map 셀 좌표
    # compute_soft_labels와 동일하게 valid_ratios 곱해서 변환
    for box, label in zip(boxes_norm, labels):
        cx, cy, bw, bh = box
        x1n, y1n = cx - bw / 2, cy - bh / 2
        x2n, y2n = cx + bw / 2, cy + bh / 2
        x1c = x1n * vw * W_l - 0.5
        y1c = y1n * vh * H_l - 0.5
        x2c = x2n * vw * W_l - 0.5
        y2c = y2n * vh * H_l - 0.5
        color = BOX_EDGE_COLORS[label % len(BOX_EDGE_COLORS)]
        rect = patches.Rectangle(
            (x1c, y1c), x2c - x1c, y2c - y1c,
            linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # 셀 경계 gridline — 실제 feature map 셀 구조를 명확히 보여줌
    for xi in np.arange(0.5, W_l, 1):
        ax.axvline(xi, color='white', lw=0.3, alpha=0.4, zorder=5)
    for yi in np.arange(0.5, H_l, 1):
        ax.axhline(yi, color='white', lw=0.3, alpha=0.4, zorder=5)

    ax.set_xlim(-0.5, W_l - 0.5)
    ax.set_ylim(H_l - 0.5, -0.5)

    n_fg = int(fg.sum())
    n_total = H_l * W_l
    stride_str = f'stride={stride}  ' if stride is not None else ''
    ax.set_title(
        f'Scale {scale_idx+1}  {stride_str}\n'
        f'{H_l}행×{W_l}열  FG: {n_fg}/{n_total} ({100*n_fg/n_total:.1f}%)',
        fontsize=8)
    ax.axis('off')


def visualize_one_image(img_id, ann_data, img_dir, out_dir, max_size=512, sim_padding=False):
    """이미지 한 장에 대해 전체 scale 시각화."""
    categories = {cat['id']: i for i, cat in enumerate(ann_data['categories'])}
    num_classes = len(ann_data['categories'])

    # 이미지 정보 로드
    img_info = next(i for i in ann_data['images'] if i['id'] == img_id)
    anns = [a for a in ann_data['annotations'] if a['image_id'] == img_id]

    img_path = os.path.join(img_dir, img_info['file_name'])
    img_orig = Image.open(img_path).convert('RGB')

    # ── 전처리: resize ──
    img_resized, scale = resize_keep_ratio(img_orig, max_size=max_size)
    img_np = np.array(img_resized)
    img_h, img_w = img_np.shape[:2]

    print(f'[이미지 {img_id}] {img_info["file_name"]}')
    print(f'  원본 크기: {img_info["width"]}×{img_info["height"]}')
    print(f'  resize 후: {img_w}×{img_h}  (scale={scale:.3f})')

    # ── GT 박스 변환 ──
    # COCO annotation: bbox = [x, y, w, h] 픽셀 절대 좌표
    # → 정규화 cxcywh (dino_moe_block.py 입력 형식)
    boxes_norm = []
    labels = []
    boxes_pixel_xyxy = []
    for a in anns:
        x, y, bw, bh = a['bbox']
        # 원본 → resize 적용
        x, y, bw, bh = x * scale, y * scale, bw * scale, bh * scale
        # 픽셀 xyxy (시각화용)
        boxes_pixel_xyxy.append([x, y, x + bw, y + bh])
        # 정규화 cxcywh (soft_labels 입력용)
        cx = (x + bw / 2) / img_w
        cy = (y + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h
        boxes_norm.append([cx, cy, nw, nh])
        cat_id = a['category_id']
        labels.append(categories[cat_id])

    gt_boxes  = torch.tensor(boxes_norm, dtype=torch.float32)   # (n, 4) cxcywh norm
    gt_labels = torch.tensor(labels,    dtype=torch.long)        # (n,)

    print(f'  GT 박스 수: {len(anns)}')
    for i, (box, label) in enumerate(zip(boxes_norm, labels)):
        print(f'    박스{i}: class={CLASS_NAMES[label]}, cx={box[0]:.3f}, cy={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}')

    # ── spatial_shapes 계산 ──
    strides = (8, 16, 32, 64)
    shapes = get_spatial_shapes(img_h, img_w, strides)
    spatial_shapes = torch.tensor(shapes, dtype=torch.long)
    print(f'  spatial_shapes (stride {strides}):')
    for s, (H, W) in zip(strides, shapes):
        print(f'    stride {s:2d} → {H}×{W} = {H*W} 토큰')

    # ── soft_labels 계산 ──
    # sim_padding: 이미지가 max_size×max_size 배치에 패딩된 상황을 시뮬레이션
    # valid_ratios = img_w/max_size, img_h/max_size (패딩 없으면 1.0)
    valid_ratios_sim = None
    if sim_padding:
        vw = img_w / max_size
        vh = img_h / max_size
        n_levels = len(shapes)
        valid_ratios_sim = torch.tensor(
            [[[vw, vh]] * n_levels], dtype=torch.float32)  # (1, L, 2)
        # spatial_shapes도 패딩 기준으로 재계산
        shapes_padded = get_spatial_shapes(max_size, max_size)
        spatial_shapes_padded = torch.tensor(shapes_padded, dtype=torch.long)
        print(f'  [sim_padding] valid_ratios: w={vw:.3f}, h={vh:.3f}')
        print(f'  spatial_shapes (padded {max_size}×{max_size}):')
        for s, (H, W) in zip((8, 16, 32, 64), shapes_padded):
            print(f'    stride {s:2d} → {H}×{W} = {H*W} 토큰')
        soft_labels, fg_mask, token_xy_per_scale = compute_soft_labels(
            [gt_boxes], [gt_labels], spatial_shapes_padded, num_classes,
            valid_ratios=valid_ratios_sim)
        shapes = shapes_padded  # 시각화도 padded shapes 기준
        spatial_shapes = spatial_shapes_padded
    else:
        soft_labels, fg_mask, token_xy_per_scale = compute_soft_labels(
            [gt_boxes], [gt_labels], spatial_shapes, num_classes)

    # soft_labels: (1, N_total, C), fg_mask: (1, N_total)
    soft_labels = soft_labels[0]   # (N_total, C)
    fg_mask     = fg_mask[0]       # (N_total,)

    # scale별로 분리
    scale_sizes = [H * W for H, W in shapes]
    sl_per_scale = soft_labels.split(scale_sizes, dim=0)  # list[(H*W, C)]
    fm_per_scale = fg_mask.split(scale_sizes, dim=0)      # list[(H*W,)]

    # valid_ratios: 시각화에서 GT 박스 좌표 변환에 사용
    if sim_padding:
        vw_vis = valid_ratios_sim[0, 0, 0].item()
        vh_vis = valid_ratios_sim[0, 0, 1].item()
        pad_info = f' → 패딩 {max_size}×{max_size} 기준 (vw={vw_vis:.3f}, vh={vh_vis:.3f})'
    else:
        vw_vis, vh_vis = 1.0, 1.0
        pad_info = ''

    # ── 시각화 ──
    # subplot 너비를 각 scale의 W_l에 비례하게 설정
    # → Scale 1(W=64)이 Scale 4(W=8)보다 8배 넓게 표시되어 해상도 차이가 명확히 보임
    n_scales = len(shapes)
    finest_W = shapes[0][1]
    width_ratios = [finest_W] + [W for _, W in shapes]
    fig_w = max(14, sum(width_ratios) / finest_W * 3.5)
    fig_h = 5.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(1, n_scales + 1, figure=fig,
                           width_ratios=width_ratios,
                           left=0.01, right=0.99, top=0.86, bottom=0.10, wspace=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(n_scales + 1)]

    fig.suptitle(
        f'Soft Label 검증: {img_info["file_name"]} ({img_w}×{img_h}){pad_info}\n'
        f'점선=GT박스경계  컬러=FG토큰  흰점선=valid경계  subplot너비∝feature map크기',
        fontsize=10)

    # 첫 번째 열: 원본 이미지 + GT 박스
    draw_gt_boxes_on_image(
        axes[0], img_np, boxes_pixel_xyxy, labels,
        title=f'원본 ({img_w}×{img_h})')

    # 나머지: scale별 foreground 마스크
    for i, (H_l, W_l) in enumerate(shapes):
        draw_scale_fg_mask(
            axes[i + 1],
            sl_per_scale[i].detach(),
            fm_per_scale[i].detach(),
            H_l, W_l,
            boxes_norm, labels,
            scale_idx=i,
            img_np=img_np,
            vw=vw_vis,
            vh=vh_vis,
            stride=strides[i])

    # 범례
    legend_handles = [
        patches.Patch(facecolor=CLASS_COLORS[i][:3], label=CLASS_NAMES[i])
        for i in range(num_classes)
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=num_classes, fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'soft_label_img{img_id}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  → 저장: {save_path}\n')

    # ── 수치 검증 출력 ──
    print('  [수치 검증] scale별 foreground 비율')
    for i, (H_l, W_l) in enumerate(shapes):
        fg = fm_per_scale[i]
        sl = sl_per_scale[i]
        n_fg = int(fg.sum().item())
        n_total = H_l * W_l
        print(f'    Scale {i+1} (stride={strides[i]:2d}, {H_l}×{W_l}): '
              f'FG={n_fg}/{n_total} ({100*n_fg/n_total:.1f}%)')
        if n_fg > 0:
            dominant = sl[fg].argmax(dim=1)
            for c in range(num_classes):
                cnt = (dominant == c).sum().item()
                if cnt > 0:
                    print(f'      {CLASS_NAMES[c]}: {cnt}개 토큰')
    print()


# ─────────────────────────────────────────────────────────────
# 4. 배치 검증 (여러 이미지를 실제 배치처럼 처리)
# ─────────────────────────────────────────────────────────────

def visualize_batch(img_ids, ann_data, img_dir, out_dir, max_size=512):
    """
    여러 이미지를 실제 배치처럼 처리하여 패딩 포함 soft_labels 검증.

    1. 각 이미지를 max_size 기준 resize (비율 유지)
    2. 배치 내 최대 H, W를 패딩 크기로 결정
    3. 이미지별 valid_ratios 계산 후 compute_soft_labels 한 번 호출
    4. 4행(이미지) × 5열(원본+4scale) 그리드 시각화
    """
    categories = {cat['id']: i for i, cat in enumerate(ann_data['categories'])}
    num_classes = len(ann_data['categories'])
    strides = (8, 16, 32, 64)

    # ── 각 이미지 로드 및 resize ──
    valid_ids = {ii['id'] for ii in ann_data['images']}
    for img_id in img_ids:
        if img_id not in valid_ids:
            sample = sorted(valid_ids)[:20]
            raise ValueError(
                f'이미지 ID {img_id}가 데이터셋에 없습니다.\n'
                f'사용 가능한 ID 예시 (앞 20개): {sample}')

    batch_data = []
    for img_id in img_ids:
        img_info = next(ii for ii in ann_data['images'] if ii['id'] == img_id)
        anns = [a for a in ann_data['annotations'] if a['image_id'] == img_id]
        img_orig = Image.open(os.path.join(img_dir, img_info['file_name'])).convert('RGB')
        img_resized, scale = resize_keep_ratio(img_orig, max_size=max_size)
        img_np = np.array(img_resized)
        img_h, img_w = img_np.shape[:2]

        boxes_norm, labels, boxes_pixel = [], [], []
        for a in anns:
            x, y, bw, bh = a['bbox']
            x, y, bw, bh = x * scale, y * scale, bw * scale, bh * scale
            boxes_pixel.append([x, y, x + bw, y + bh])
            boxes_norm.append([(x + bw / 2) / img_w, (y + bh / 2) / img_h,
                               bw / img_w, bh / img_h])
            labels.append(categories[a['category_id']])

        print(f'  img{img_id}: {img_info["file_name"]}  resize={img_w}×{img_h}  GT={len(anns)}')
        batch_data.append(dict(img_id=img_id, img_info=img_info,
                               img_np=img_np, img_h=img_h, img_w=img_w,
                               boxes_norm=boxes_norm, labels=labels,
                               boxes_pixel=boxes_pixel))

    # ── 배치 패딩 크기: 배치 내 max H, max W ──
    pad_h = max(d['img_h'] for d in batch_data)
    pad_w = max(d['img_w'] for d in batch_data)
    print(f'\n배치 패딩 크기: {pad_w}×{pad_h}')
    for d in batch_data:
        vw = d['img_w'] / pad_w
        vh = d['img_h'] / pad_h
        print(f'  img{d["img_id"]}: valid_ratio  w={vw:.3f}  h={vh:.3f}')

    # ── spatial_shapes: 패딩 크기 기준 ──
    shapes_padded = [(math.ceil(pad_h / s), math.ceil(pad_w / s)) for s in strides]
    spatial_shapes = torch.tensor(shapes_padded, dtype=torch.long)
    print(f'\nspatial_shapes (padded {pad_w}×{pad_h}):')
    for s, (H, W) in zip(strides, shapes_padded):
        print(f'  stride {s:2d} → {H}×{W} = {H*W} 토큰')

    # ── valid_ratios: (B, L, 2) ──
    n_levels = len(strides)
    valid_ratios = torch.tensor(
        [[[d['img_w'] / pad_w, d['img_h'] / pad_h]] * n_levels for d in batch_data],
        dtype=torch.float32)

    # ── compute_soft_labels: 배치 전체를 한번에 ──
    gt_boxes_list  = [torch.tensor(d['boxes_norm'], dtype=torch.float32) for d in batch_data]
    gt_labels_list = [torch.tensor(d['labels'],     dtype=torch.long)    for d in batch_data]
    soft_labels, fg_mask, _ = compute_soft_labels(
        gt_boxes_list, gt_labels_list, spatial_shapes, num_classes,
        valid_ratios=valid_ratios)

    scale_sizes = [H * W for H, W in shapes_padded]

    # ── 시각화: B행 × (1+n_scales)열 ──
    B = len(batch_data)
    n_scales = len(strides)
    finest_W = shapes_padded[0][1]
    width_ratios = [finest_W] + [W for _, W in shapes_padded]
    fig_w = max(16, sum(width_ratios) / finest_W * 3.2)
    fig_h = B * 3.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(B, n_scales + 1, figure=fig,
                           width_ratios=width_ratios,
                           left=0.01, right=0.99, top=0.96, bottom=0.04,
                           wspace=0.08, hspace=0.40)

    fig.suptitle(
        f'배치 Soft Label 검증  패딩={pad_w}×{pad_h}\n'
        f'회색=패딩영역  점선=GT박스  컬러=FG토큰  subplot너비∝feature map크기',
        fontsize=11)

    for b, d in enumerate(batch_data):
        vw = d['img_w'] / pad_w
        vh = d['img_h'] / pad_h

        sl_b = soft_labels[b]
        fm_b = fg_mask[b]
        sl_per = sl_b.split(scale_sizes, dim=0)
        fm_per = fm_b.split(scale_sizes, dim=0)

        ax0 = fig.add_subplot(gs[b, 0])
        draw_gt_boxes_on_image(
            ax0, d['img_np'], d['boxes_pixel'], d['labels'],
            title=f"img{d['img_id']}  {d['img_w']}×{d['img_h']}\n"
                  f"vw={vw:.2f}  vh={vh:.2f}")

        for i, (H_l, W_l) in enumerate(shapes_padded):
            ax = fig.add_subplot(gs[b, i + 1])
            draw_scale_fg_mask(
                ax,
                sl_per[i].detach(),
                fm_per[i].detach(),
                H_l, W_l,
                d['boxes_norm'], d['labels'],
                scale_idx=i,
                img_np=d['img_np'],
                vw=vw, vh=vh,
                stride=strides[i])

    legend_handles = [
        patches.Patch(facecolor=CLASS_COLORS[i][:3], label=CLASS_NAMES[i])
        for i in range(num_classes)
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=num_classes, fontsize=9, framealpha=0.8)

    os.makedirs(out_dir, exist_ok=True)
    ids_str = '_'.join(str(i) for i in img_ids)
    save_path = os.path.join(out_dir, f'batch_verify_{ids_str}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'\n  → 저장: {save_path}')

    # ── 수치 검증 ──
    print('\n[배치 수치 검증]')
    for b, d in enumerate(batch_data):
        vw = d['img_w'] / pad_w
        vh = d['img_h'] / pad_h
        print(f"  img{d['img_id']} (vw={vw:.3f}, vh={vh:.3f}):")
        sl_b = soft_labels[b]
        fm_b = fg_mask[b]
        sl_per = sl_b.split(scale_sizes, dim=0)
        fm_per = fm_b.split(scale_sizes, dim=0)
        for i, (H_l, W_l) in enumerate(shapes_padded):
            n_fg = int(fm_per[i].sum())
            n_total = H_l * W_l
            print(f'    Scale {i+1} stride={strides[i]:2d} {H_l}×{W_l}: '
                  f'FG={n_fg}/{n_total} ({100*n_fg/n_total:.1f}%)')


# ─────────────────────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='soft_labels 시각화 검증')
    parser.add_argument('--coco_path', default='/home/moonhw/SIXray-D/SIXray_COCO_ROOT')
    parser.add_argument('--split', default='train2017')
    parser.add_argument('--img_ids', type=int, nargs='+', default=[9, 14],
                        help='시각화할 이미지 ID')
    parser.add_argument('--max_size', type=int, default=512,
                        help='DINO 전처리 최대 이미지 크기')
    parser.add_argument('--out_dir', default='outputs/soft_label_verify')
    parser.add_argument('--sim_padding', action='store_true',
                        help='단일 이미지 패딩 시뮬레이션: max_size×max_size 기준 valid_ratios 적용')
    parser.add_argument('--batch_verify', action='store_true',
                        help='배치 패딩 검증: img_ids를 한 배치로 처리하여 실제 패딩 동작 확인')
    args = parser.parse_args()

    ann_file = os.path.join(args.coco_path, 'annotations', f'instances_{args.split}.json')
    img_dir  = os.path.join(args.coco_path, 'images', args.split)

    print(f'어노테이션 로드: {ann_file}')
    with open(ann_file) as f:
        ann_data = json.load(f)

    if args.batch_verify:
        print(f'\n[배치 검증 모드] 이미지 {args.img_ids}를 한 배치로 처리')
        visualize_batch(args.img_ids, ann_data, img_dir, args.out_dir, args.max_size)
    else:
        for img_id in args.img_ids:
            visualize_one_image(img_id, ann_data, img_dir, args.out_dir, args.max_size,
                                sim_padding=args.sim_padding)

    print(f'\n완료! 결과 디렉토리: {args.out_dir}')


if __name__ == '__main__':
    main()
