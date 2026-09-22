#!/usr/bin/env python
"""异常检测离线验证 CLI:良品/缺陷照片 → AUROC/阈值建议/热力图。.

(核心逻辑在 dino_anomaly.validate_dataset,GUI 的异常检测页复用同一实现)

用法:
  .../python anomaly/validate.py --bank models/anomaly_bank.npz \
      --good-dir data/test_good --ng-dir data/test_ng --out out/anomaly

输出(out 目录):
  scores.csv           每张图的异常分数与标签
  summary.txt          AUROC、建议阈值(Youden 最优)、零误报阈值
  heatmap/             NG 图与高分误报图的热力图叠加(定位缺陷位置)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dino_anomaly as da


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True, help="build_bank.py 产出的 .npz")
    ap.add_argument("--good-dir", default="", help="良品验证图目录(可选)")
    ap.add_argument("--ng-dir", default="", help="缺陷图目录(可选)")
    ap.add_argument("--out", default="out/anomaly")
    ap.add_argument("--model", default="dinov3_vits16")
    ap.add_argument("--hub-repo", default="facebookresearch/dinov2", help="仅 dinov2 走 torch.hub 时使用")
    ap.add_argument("--weights", default="", help="dinov3 本地权重 .pth(模型名以 dinov3 开头时必填)")
    ap.add_argument("--repo-dir", default="", help="dinov3 仓库代码目录(默认自动找 anomaly 同级的 dinov3-main)")
    ap.add_argument("--top-k-heatmap", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    items = []
    if args.good_dir:
        items += [(p, 0) for p in da.list_images(args.good_dir)]
    if args.ng_dir:
        items += [(p, 1) for p in da.list_images(args.ng_dir)]
    if not items:
        print("error: 需要 --good-dir 和/或 --ng-dir")
        return 2

    bank, meta = da.load_bank(args.bank)
    print(f"bank: {bank.shape[0]} vectors ({meta})")
    # 归一化设备字符串(GUI 会传 '0'/'cuda:0',直接 .to('0') 会抛异常)
    device = da.norm_device(args.device)
    model, patch_size, device = da.load_model(
        args.model, args.hub_repo, device, weights_path=args.weights, repo_dir=args.repo_dir
    )
    bank_t = torch.from_numpy(bank).to(device)
    lines = da.validate_dataset(model, patch_size, bank_t, items, args.out, device, args.top_k_heatmap)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
