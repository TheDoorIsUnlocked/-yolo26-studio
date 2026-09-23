"""
RF-DETR 命令行训练脚本（不依赖界面，独立进程跑，显存更省、崩了不影响界面）.

用法示例：
    python train_rfdetr.py --data 3631.yaml --variant Nano --epochs 50
    python train_rfdetr.py --data 3631.yaml --variant Nano --batch 2 --grad-accum 8 --epochs 100
    python train_rfdetr.py --data 3631.yaml --variant Nano --epochs 50 --export --opset 17

全部参数见 --help。数据集支持 train/images 与 images/train 两种布局，自动识别。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MB = 1024**2

VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}

# 各变体的默认分辨率（取自 rfdetr/config.py 各 *Config 的 resolution）
DEFAULT_RESOLUTION = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "base": 560,
    "large": 560,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_rfdetr.py",
        description="RF-DETR 命令行训练 / 导出（Ultralytics Studio 配套脚本）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_data = p.add_argument_group("数据与模型")
    g_data.add_argument("--data", required=True, help="数据集 YAML（Ultralytics 格式）")
    g_data.add_argument("--variant", default="nano", choices=list(VARIANTS), help="模型变体")
    g_data.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="输入分辨率，必须是 patch_size*num_windows 的倍数；留空用变体默认值",
    )
    g_data.add_argument("--resume", default=None, help="从已有 checkpoint（.pth）继续训练")

    g_train = p.add_argument_group("训练超参")
    g_train.add_argument("--epochs", type=int, default=50)
    g_train.add_argument("--batch", default="auto", help="物理 batch，auto 让 RF-DETR 自己探测显存")
    g_train.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--lr-encoder", type=float, default=1.5e-4, help="backbone 学习率，一般不用改")
    g_train.add_argument("--weight-decay", type=float, default=1e-4)
    g_train.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    g_train.add_argument("--num-workers", type=int, default=0, help="Windows 下保持 0；Linux 可设 4~8")
    g_train.add_argument("--no-ema", action="store_true", help="关闭 EMA（省显存，4GB 卡建议关）")
    g_train.add_argument("--early-stop", action="store_true", help="启用早停")
    g_train.add_argument("--patience", type=int, default=10, help="早停耐心轮数")
    g_train.add_argument("--run-test", action="store_true", help="训练结束后用 test 集再评估一次")

    g_out = p.add_argument_group("输出")
    g_out.add_argument(
        "--output", default=str(HERE / "rfdetr_output"), help="输出根目录，每次训练在其下自动递增建 trainN 子目录"
    )
    g_out.add_argument("--run-name", default=None, help="指定子目录名，不给则自动 trainN")
    g_out.add_argument("--export", action="store_true", help="训练完自动导出 ONNX")
    g_out.add_argument("--opset", type=int, default=17)
    return p


def next_run_dir(root: Path, run_name: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        d = root / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    n = 1
    while (root / f"train{n}").exists():
        n += 1
    d = root / f"train{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gpu_report():
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        print("[显存] CUDA 不可用，将走 CPU")
        return
    free, total = torch.cuda.mem_get_info()
    dev = torch.cuda.get_device_properties(0)
    print(f"[显存] 显卡 {dev.name}  可用 {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
    print("[显存] 提示：Windows 桌面本身通常已占约 1GB，剩余才是训练可用额度")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from rfdetr_adapter import RFDETRDatasetError, ensure_rf_home, prepare_rfdetr_dataset
    except Exception as e:
        print(f"[错误] 无法导入 rfdetr_adapter.py：{e}")
        return 2

    gpu_report()
    print("-" * 72)

    # 1) 数据集适配：自动识别 train/images 或 images/train，并生成 data.yaml
    try:
        info = prepare_rfdetr_dataset(args.data)
    except RFDETRDatasetError as e:
        print(f"[错误] {e}")
        return 1
    for note in info.get("notes", []):
        print("· " + str(note))
    print(f"[数据] {info['dataset_dir']}  （{info['num_classes']} 类：{list(info['names'].values())}）")
    print(f"[数据] RF-DETR 使用：{info['data_yaml']}")

    ensure_rf_home()  # 权重缓存重定向到项目所在盘，避免写 C 盘

    # 2) 输出目录自增
    run_dir = next_run_dir(Path(args.output), args.run_name)
    print(f"[输出] {run_dir}")

    # 3) 建模型
    import rfdetr

    cls_name = VARIANTS[args.variant.lower()]
    model = getattr(rfdetr, cls_name)()

    batch = "auto" if str(args.batch).lower() == "auto" else int(args.batch)
    res = args.resolution or DEFAULT_RESOLUTION[args.variant.lower()]

    print("-" * 72)
    print(
        f"[配置] 变体={cls_name}  分辨率={res}  epochs={args.epochs}  "
        f"batch={batch}  grad_accum={args.grad_accum}  lr={args.lr}  "
        f"EMA={'关' if args.no_ema else '开'}"
    )

    train_kwargs = {
        "dataset_file": "yolo",
        "dataset_dir": info["dataset_dir"],
        "epochs": args.epochs,
        "batch_size": batch,
        "grad_accum_steps": args.grad_accum,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "num_workers": args.num_workers,
        "output_dir": str(run_dir),
        "resolution": res,
        "use_ema": not args.no_ema,
        "early_stopping": args.early_stop,
        "early_stopping_patience": args.patience,
        "run_test": args.run_test,
        "tensorboard": False,
        "wandb": False,
    }
    if args.resume:
        train_kwargs["resume"] = args.resume

    try:
        model.train(**train_kwargs)
    except KeyboardInterrupt:
        print("\n[中断] 已停止训练")
        return 130
    except Exception as e:
        print(f"[错误] 训练失败：{type(e).__name__}: {e}")
        print("[提示] 若是显存不足（OOM / 进程无堆栈闪退），把 --batch 降到 2，并把 --grad-accum 加倍")
        return 1

    best = run_dir / "checkpoint_best_regular.pth"
    print("-" * 72)
    if best.is_file():
        print(f"[完成] 最佳权重：{best}")
    else:
        print(f"[完成] 产物目录：{run_dir}")

    if args.export:
        try:
            print(f"[导出] 正在导出 ONNX（opset={args.opset}）…")
            out = model.export(output_dir=str(run_dir), opset_version=args.opset, verbose=False)
            print(f"[导出] 完成：{out}")
        except Exception as e:
            print(f"[导出] 失败：{type(e).__name__}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
