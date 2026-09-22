# -*- coding: utf-8 -*-
"""
YOLO 命令行训练脚本（不依赖界面，底层调用 ultralytics 官方 `yolo` CLI）

之所以包一层：本机 `yolo.exe` 装在 conda base（torch 2.5.1），而 Ultralytics Studio
跑在 conda `yolo` 环境（torch 2.9.1），直接敲 `yolo` 可能用到错误的解释器/版本。
本脚本保证「当前 python 就是真正跑训练的那个 python」。

用法示例：
    python train_yolo.py --data 3631.yaml --model weights/yolo26n.pt --epochs 100
    python train_yolo.py --data 3631.yaml --model yolo26n.pt --batch 2 --imgsz 640 --epochs 200
    python train_yolo.py --data 3631.yaml --model runs/detect/train/weights/last.pt --resume
    python train_yolo.py --data 3631.yaml --model yolo26n-seg.pt --task segment

其余 ultralytics 原生参数可用 `--extra key=value` 透传：
    python train_yolo.py --data 3631.yaml --model yolo26n.pt --extra close_mosaic=0 mixup=0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TASKS = ["detect", "segment", "classify", "pose", "obb"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_yolo.py",
        description="YOLO 命令行训练（Ultralytics Studio 配套脚本）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default="detect", choices=TASKS, help="任务类型")
    p.add_argument("--mode", default="train", help="train / val / predict / export")
    p.add_argument("--model", required=True, help="模型文件（.pt/.yaml）")
    p.add_argument("--data", default=None, help="数据集 YAML")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", default=None,
                   help="物理 batch；-1 为自动，-0.5 为按显存百分比；留空用默认 16")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="0 / 0,1 / cpu")
    p.add_argument("--workers", type=int, default=0, help="Windows 建议 0；Linux 可 8")
    p.add_argument("--optimizer", default="auto", help="auto / SGD / Adam / AdamW")
    p.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    p.add_argument("--lrf", type=float, default=0.01, help="最终学习率系数(lr0*lrf)")
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--patience", type=int, default=100, help="早停耐心轮数，0=关")
    p.add_argument("--project", default=None,
                   help="输出根目录，留空用 ultralytics 默认（runs/<task>）")
    p.add_argument("--name", default=None, help="本次运行名，留空自动 train/train2…")
    p.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    p.add_argument("--amp", action="store_true", default=True, help="混合精度")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="关闭混合精度")
    p.add_argument("--cache", default=False, help="False / ram / disk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto-fix-data", dest="auto_fix", action="store_true", default=True,
                   help="自动修正数据集路径（train/images 与 images/train 两种布局）")
    p.add_argument("--no-auto-fix-data", dest="auto_fix", action="store_false",
                   help="关掉上面的自动修正，原样使用 --data")
    p.add_argument("--extra", nargs="*", default=[],
                   help="透传其它 ultralytics 参数，形如 key=value")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from ultralytics import YOLO
    import torch

    # Ultralytics 不会像 RF-DETR 适配器那样自动处理目录布局，
    # 这里借用同一套适配器把 yaml 里的路径修正到磁盘真实目录。
    data_yaml = args.data
    if args.auto_fix and args.data:
        try:
            from rfdetr_adapter import prepare_rfdetr_dataset
            info = prepare_rfdetr_dataset(args.data)
            data_yaml = info["data_yaml"]
            for note in info.get("notes", []):
                print("· " + str(note))
            print(f"[数据] {info['dataset_dir']}  （{info['num_classes']} 类："
                  f"{list(info['names'].values())}）")
        except Exception as e:
            print(f"[数据] 自动修正失败（{e}），改用原始配置：{args.data}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        dev = torch.cuda.get_device_properties(0)
        print(f"[显存] {dev.name}  可用 {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
    else:
        print("[显存] CUDA 不可用，将走 CPU")

    kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "project": args.project,
        "resume": args.resume,
        "amp": args.amp,
        "seed": args.seed,
    }
    if args.batch is not None:
        # batch 必须是 int 或 float（-1 自动、-0.5 表示占 50% 显存）
        coerced = _coerce(args.batch)
        if isinstance(coerced, str):
            print(f"[错误] --batch 需要数字，收到：{args.batch}")
            return 2
        kwargs["batch"] = coerced
    if args.project:
        kwargs["project"] = args.project
    if args.name:
        kwargs["name"] = args.name
    if str(args.cache).lower() in ("ram", "disk"):
        kwargs["cache"] = args.cache

    for kv in args.extra:
        if "=" not in kv:
            print(f"[错误] --extra 需要 key=value 形式，收到：{kv}")
            return 2
        k, v = kv.split("=", 1)
        kwargs[k] = _coerce(v)

    print("-" * 72)
    print(f"[配置] task={args.task}  model={args.model}  epochs={args.epochs}  "
          f"batch={kwargs.get('batch', '默认')}  imgsz={args.imgsz}  "
          f"lr0={args.lr0}  amp={args.amp}  resume={args.resume}")

    model = YOLO(args.model)
    try:
        results = getattr(model, args.mode)(**kwargs)
    except KeyboardInterrupt:
        print("\n[中断] 已停止")
        return 130
    except Exception as e:
        print(f"[错误] {type(e).__name__}: {e}")
        print("[提示] 显存不足就调小 --batch，或把 --imgsz 降到 320/416")
        return 1

    save_dir = getattr(results, "save_dir", None)
    print("-" * 72)
    print(f"[完成] 输出目录：{save_dir}")
    return 0


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


if __name__ == "__main__":
    raise SystemExit(main())
