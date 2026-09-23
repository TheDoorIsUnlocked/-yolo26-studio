#!/usr/bin/env python
"""DINO 特征异常检测核心库(PatchCore 式良品建模)。.

流程:
  良品照片 → 逐块 DINO 特征 → coreset 降采样 → 特征库 bank.npz
  待检图   → 逐块 DINO 特征 → 与特征库最近邻距离 → 异常分数 + 热力图

适用场景:缺陷样本极少(甚至为零)的注塑件外观检测(缺胶/毛边/黑点)。
只需大量良品照片建库,缺陷照片仅用于调阈值与验证召回。

模型默认 DINOv2 ViT-S/14(权重可自由下载,首次运行自动缓存)。
拿到 DINOv3 权重后可 --model dinov3_vits16 --hub-repo facebookresearch/dinov3
切换(需按 HF 要求接受许可),用法与特征维度处理完全一致。
"""

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


_HF_NAME_MAP = {  # torch.hub 名 → HuggingFace 权重名
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
}


def _norm_device(device: str) -> str:
    """'0'/'cuda:0'/'cuda' → 'cuda';其余按 'cpu' 处理(GUI 传 ultralytics 风格)。."""
    d = str(device)
    return "cpu" if d.startswith("cpu") else "cuda"


def load_backbone_model(
    model_name: str = "dinov3_vits16", weights_path: str = "", repo_dir: str = "", device: str = "cuda"
):
    """加载 dinov3 骨干**本体**,返回 (model, patch_size, device)。.

    与 load_model 的区别:返回模型对象本身(导出 ONNX 需要包 wrapper), 而非特征提取闭包。两者共用同一套权重加载路径,特征结果一致。
    """
    device = _norm_device(device)
    if not repo_dir:
        cand = Path(__file__).resolve().parent.parent / "dinov3-main"
        repo_dir = str(cand) if cand.is_dir() else ""
    if not repo_dir or not weights_path:
        raise RuntimeError("dinov3 需要 --repo-dir(dinov3 仓库代码目录)与 --weights(本地 .pth),未提供")
    # 占位绕过 dinov3 的分割 eval 分支:其 ms_deform_attn.py 用了部分环境
    # torch 不支持的 torch.amp 装饰器签名;建库/推理/导出只用 ViT backbone,
    # 用不到分割算子,预置空模块让 import 直接命中占位。
    stub_name = "dinov3.eval.segmentation.models.utils.ms_deform_attn"
    if stub_name not in sys.modules:
        import types

        stub = types.ModuleType(stub_name)

        class MSDeformAttn:  # 占位类,仅满足 from ... import MSDeformAttn
            pass

        stub.MSDeformAttn = MSDeformAttn
        sys.modules[stub_name] = stub
    # 直接按名字从仓库取 backbone 构造函数,不走 torch.hub.load:
    # hubconf.py 会连带引入 torchmetrics 等额外依赖(Windows 环境常缺失),
    # 且会把本地权重复制缓存到 ~/.cache/torch/hub(Windows 上通常是 C 盘)。
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from dinov3.hub import backbones as _dino_backbones
    except Exception as e:
        raise RuntimeError(f"无法从 {repo_dir} 导入 dinov3 仓库代码: {e}")
    builder = getattr(_dino_backbones, model_name, None)
    if builder is None:
        raise RuntimeError(
            f"dinov3 不支持的骨干名 '{model_name}'(可用: dinov3_vits16 / dinov3_vits16plus / dinov3_vitb16 ...)"
        )
    m = builder(pretrained=False)
    state = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval().to(device)
    return m, int(m.patch_size), device


def load_model(
    model_name: str = "dinov2_vits14",
    hub_repo: str = "facebookresearch/dinov2",
    device: str = "cuda",
    hf_endpoint: str = "https://hf-mirror.com",
    weights_path: str = "",
    repo_dir: str = "",
):
    """加载 DINO backbone,返回 (run, patch_size, device)。.

    run(x) 接收预处理张量 (1,3,S,S),返回 patch tokens (1,P,D)。
    - dinov2:优先 torch.hub(GitHub),失败自动落 HuggingFace 镜像。
    - dinov3(权重 .pth 已下载到本地):weights_path=本地权重路径,
    repo_dir=仓库代码目录(默认自动找 anomaly 同级的 dinov3-main)。 权重已缓存/本地时离线可用。
    """
    device = _norm_device(device)

    if model_name.startswith("dinov3"):
        m, patch_size, device = load_backbone_model(model_name, weights_path, repo_dir, device)

        def run(x, _m=m):
            return _m.forward_features(x)["x_norm_patchtokens"]

        return run, patch_size, device

    try:
        m = torch.hub.load(hub_repo, model_name)
        m.eval().to(device)
        patch_size = m.patch_size

        def run(x, _m=m):
            return _m.forward_features(x)["x_norm_patchtokens"]

        return run, patch_size, device
    except Exception as e:
        print(f"torch.hub 加载失败({e.__class__.__name__}: {e}),改用 HuggingFace 镜像 {hf_endpoint}")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    try:  # huggingface_hub 若已导入,常量在 import 时固化,需同步改
        import huggingface_hub.constants as _hfc

        _hfc.ENDPOINT = hf_endpoint
    except Exception:
        pass
    from transformers import AutoModel

    hf_name = _HF_NAME_MAP.get(model_name, model_name)
    try:
        m = AutoModel.from_pretrained(hf_name)
    except Exception:
        m = AutoModel.from_pretrained(hf_name, local_files_only=True)
    m.eval().to(device)
    patch_size = int(m.config.patch_size)

    def run(x, _m=m):
        # dinov2 无 register token:last_hidden_state 的第 0 位是 CLS
        return _m(x).last_hidden_state[:, 1:, :]

    return run, patch_size, device


def _preprocess(img_bgr: np.ndarray, img_size: int, device: str) -> torch.Tensor:
    """BGR uint8 → 归一化 float 张量(1,3,S,S)。整图 resize 到 S×S, 良品与待检图用同一变换,轻微形变不影响块级比对。.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        x[c] = (x[c] - m) / s
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def extract_patch_features(model, img_bgr: np.ndarray, patch_size: int, device: str = "cuda") -> tuple:
    """返回 (patch 特征 (P,D) float32 numpy, 网格 (gh, gw))。."""
    device = _norm_device(device)
    img_size = patch_size * 37  # dinov2:518 / dinov3(vits16):592
    x = _preprocess(img_bgr, img_size, device)
    tokens = model(x)[0]  # (gh*gw, D)
    tokens.shape[-1]
    gh = gw = int(tokens.shape[0] ** 0.5)
    feats = F.normalize(tokens, dim=-1)  # L2 归一化:余弦距离可比且稳定
    return (feats.cpu().numpy().astype(np.float32), (gh, gw))


def _greedy_coreset(feats: np.ndarray, budget: int, device: str, seed: int = 0) -> np.ndarray:
    """贪心 k-center coreset:选出的点尽量覆盖整个特征空间 (PatchCore 同款思想)。候选量大时先随机下采样到 8*budget 再贪心。.
    """
    rng = np.random.default_rng(seed)
    n = len(feats)
    if n <= budget:
        return np.arange(n)
    cand = rng.choice(n, size=min(n, budget * 8), replace=False)
    x = torch.from_numpy(feats[cand]).to(device)
    sel = [int(rng.integers(len(cand)))]
    dmin = torch.cdist(x[sel[-1] : sel[-1] + 1], x)[0]
    while len(sel) < budget:
        idx = int(torch.argmax(dmin).item())
        sel.append(idx)
        dmin = torch.minimum(dmin, torch.cdist(x[idx : idx + 1], x)[0])
    return cand[np.array(sel)]


def build_bank(
    model,
    patch_size: int,
    image_paths: list,
    out_path: str,
    device: str = "cuda",
    bank_size: int = 16384,
    log=print,
    model_name: str = "",
) -> dict:
    device = _norm_device(device)
    """遍历良品照片提取逐块特征,coreset 降采样后保存特征库。

    返回统计信息 {images, patches_raw, patches_bank, out}。"""
    all_feats = []
    for i, p in enumerate(image_paths):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)  # 兼容中文路径
        if img is None:
            log(f"skip unreadable: {p}")
            continue
        f, _ = extract_patch_features(model, img, patch_size, device)
        all_feats.append(f)
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(image_paths)} images")
    if not all_feats:
        raise RuntimeError("no readable images for bank building")
    raw = np.concatenate(all_feats, axis=0)
    log(f"total patches: {raw.shape[0]}, running coreset (budget={bank_size})")
    keep = _greedy_coreset(raw, bank_size, device)
    bank = raw[keep]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feats=bank.astype(np.float16),
        meta=json.dumps(
            {
                "model": model_name or "dino",
                "patch_size": patch_size,
                "n_images": len(all_feats),
                "n_patches_raw": int(raw.shape[0]),
            }
        ),
    )
    # 同步输出 .fbin(服务端 C++ 加载格式: int32 M,D + float32 M*D)
    fbin = out.with_suffix(".fbin")
    fb = bank.astype(np.float32)
    with open(fbin, "wb") as f:
        f.write(np.int32(bank.shape[0]).tobytes())
        f.write(np.int32(bank.shape[1]).tobytes())
        f.write(np.ascontiguousarray(fb).tobytes())
    log(f"feature bank fbin saved: {fbin}")
    log(f"bank saved: {out} ({bank.shape[0]} vectors)")
    return {
        "images": len(all_feats),
        "patches_raw": int(raw.shape[0]),
        "patches_bank": int(bank.shape[0]),
        "out": str(out),
    }


def load_bank(path: str) -> tuple:
    """返回 (bank (M,D) float32, meta dict)。."""
    z = np.load(path, allow_pickle=False)
    return z["feats"].astype(np.float32), json.loads(str(z["meta"]))


@torch.no_grad()
def score_image(
    model, patch_size: int, bank_t: torch.Tensor, img_bgr: np.ndarray, device: str = "cuda", top_pct: float = 0.01
) -> tuple:
    """单图异常评分。返回 (image_score, patch 距离图 (gh,gw), 网格)。.

    image_score = 最异常的 top_pct 比例块的距离均值(比单块 max 稳定)。
    """
    feats, (gh, gw) = extract_patch_features(model, img_bgr, patch_size, device)
    x = torch.from_numpy(feats).to(device)
    dmin = torch.full((x.shape[0],), float("inf"), device=device)
    chunk = 4096
    for s in range(0, bank_t.shape[0], chunk):
        d = torch.cdist(x, bank_t[s : s + chunk])  # (P, m) 余弦距离(L2 归一化)
        dmin = torch.minimum(dmin, d.min(dim=1).values)
    k = max(1, round(x.shape[0] * top_pct))
    image_score = float(dmin.topk(k).values.mean().item())
    return image_score, dmin.reshape(gh, gw).cpu().numpy(), (gh, gw)


def make_heatmap(img_bgr: np.ndarray, dist_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """原始图 + 异常热力图叠加(BGR,与输入同尺寸)。."""
    m = dist_map - dist_map.min()
    rng = m.max() - m.min()
    if rng > 1e-9:
        m /= rng
    h, w = img_bgr.shape[:2]
    heat = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)


def list_images(d: str):
    """目录下的图片文件(递归,常见格式)。."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    p = Path(d)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)


def _auroc(labels, scores) -> float:
    """AUROC(Mann-Whitney U 秩公式)。优先用 sklearn,缺失时退回自实现, 避免验证环节因环境缺 sklearn 而整体失败。并列分数取平均秩。.
    """
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        pass
    y = np.asarray(labels).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s_sorted), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = ranks[y[order] == 1]
    return float((pos.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize_scores(rows: list) -> list:
    """根据 [(path, label 1=NG, score), ...] 生成指标说明行 (AUROC/Youden 阈值/零误报阈值/分数分布)。.
    """
    lines = [
        (f"images: {len(rows)} (ng={int(sum(r[1] for r in rows))}, good={int(sum(1 for r in rows if r[1] == 0))})")
    ]
    labels = np.array([r[1] for r in rows])
    scores = np.array([r[2] for r in rows])
    n_ng = int(labels.sum())
    n_good = int((labels == 0).sum())
    if n_ng == 0 or n_good == 0:
        lines.append("提示: 只有单一类别,无法算 AUROC;以下仅列分数分布。")
    else:
        lines.append(f"AUROC: {_auroc(labels, scores):.4f}")
        best = (0.0, 0.0)  # (Youden J, threshold)
        for th in np.unique(scores):
            pred = scores >= th
            j = pred[labels == 1].mean() - pred[labels == 0].mean()
            if j > best[0]:
                best = (j, float(th))
        th = best[1]
        pred = scores >= th
        lines.append(
            f"建议阈值: {th:.4f} (Youden 最优) — "
            f"漏检={int(((labels == 1) & ~pred).sum())} "
            f"误报={int(((labels == 0) & pred).sum())} "
            f"正确NG={int(((labels == 1) & pred).sum())} "
            f"正确OK={int(((labels == 0) & ~pred).sum())}"
        )
        if n_good:
            th0 = float(scores[labels == 0].max()) * 1.05 + 1e-9
            lines.append(
                f"零误报阈值: {th0:.4f} (良品最高分×1.05), 该阈值下漏检={int((scores[labels == 1] < th0).sum())}"
            )
    qs = np.percentile(scores, [0, 25, 50, 75, 100])
    lines.append(f"分数分布 min/25/50/75/max: {qs[0]:.4f}/{qs[1]:.4f}/{qs[2]:.4f}/{qs[3]:.4f}/{qs[4]:.4f}")
    return lines


def validate_dataset(
    model,
    patch_size: int,
    bank_t: torch.Tensor,
    items: list,
    out_dir: str,
    device: str = "cuda",
    top_k_heatmap: int = 20,
    log=print,
) -> list:
    device = _norm_device(device)
    """对 [(path, label), ...] 全量评分并输出 CSV/summary/热力图。

    返回 summary 文本行(GUI 直接显示,与 summary.txt 一致)。"""
    out = Path(out_dir)
    (out / "heatmap").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (p, label) in enumerate(items):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log(f"skip unreadable: {p}")
            continue
        score, dmap, _ = score_image(model, patch_size, bank_t, img, device)
        rows.append((str(p), label, score))
        log(f"[{i + 1}/{len(items)}] {'NG ' if label else 'OK '} score={score:.4f} {p}")
    if not rows:
        raise RuntimeError("no readable images")
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label_ng", "score"])
        w.writerows(rows)
    lines = summarize_scores(rows)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    rank = sorted(rows, key=lambda r: -r[2])
    ng_rows = [r for r in rank if r[1] == 1][:top_k_heatmap]
    fp_rows = [r for r in rank if r[1] == 0][: max(0, top_k_heatmap - len(ng_rows))]
    for p, label, score in ng_rows + fp_rows:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        _, dmap, _ = score_image(model, patch_size, bank_t, img, device)
        vis = make_heatmap(img, dmap)
        stem = Path(p).stem
        tag = "ng" if label else "fp"
        cv2.imwrite(str(out / "heatmap" / f"{tag}_{score:.4f}_{stem}.jpg"), vis)
    log(f"heatmaps saved to {out / 'heatmap'}")
    return lines


# 供外部(GUI worker)归一化设备字符串:'0'/'cuda:0'/'cuda' → 'cuda'
norm_device = _norm_device
