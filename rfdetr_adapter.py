"""RF-DETR 数据集适配器。.

RF-DETR 的 YOLO 数据集读取规则（见 rfdetr/datasets/yolo.py）：

    data.yaml 里声明 train / val 为「图片目录」的绝对/相对路径；
    labels 目录 = 把该路径中**任意位置的 images 段换成 labels** 得到的目录。
    因此 RF-DETR 原生同时兼容两种布局：

        <根>/train/images  + <根>/train/labels      （split-first）
        <根>/images/train  + <根>/labels/train      （images-first）

本项目里不同数据集的 yaml 写法不统一（有的写 train/images，有的写
images/train），且用户原始 yaml 可能与磁盘实际布局不一致。本模块负责：

    1. 解析用户 yaml 的 train/val 声明；
    2. 以**磁盘上真实存在的目录**为准，自动在两种顺序间选择；
    3. 生成一份 data.yaml（所有路径写绝对路径）供 RF-DETR 训练使用。

**不复制、不移动、不修改任何图片和标注**，也**不改动用户原始 yaml**。
这样即便 yaml 写成 images/train 而磁盘是 train/images（或反过来），
也能自动适配，不会再误报「反向布局」错误（历史上 3631 就踩过这个坑）。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

__all__ = [
    "RFDETRDatasetError",
    "prepare_rfdetr_dataset",
    "rfdetr_available",
    "rfdetr_missing_reason",
    "suggest_rf_batch",
]


def suggest_rf_batch() -> int:
    """返回 RF-DETR 训练的安全物理 batch（**具体整数，绝不返回 'auto'**）。.

    为什么要自己算、不直接传 'auto'：

    RF-DETR 收到 ``batch_size="auto"`` 时会跑自动 batch 探测 （``rfdetr/training/auto_batch.py`` 的 ``resolve_auto_batch_config``）。
    该探测会额外建一份 shadow 参数副本 + 跑 AdamW 的 ``step()``（再占 2× 参数）， 并用合成 batch 从大到小试到 OOM。在显存本就紧张的卡上（如 4GB 笔记本卡， Windows 桌面 +
    YOLO 推理模型常驻后只剩约 2.8GB 可用），探测本身就会把显存顶爆， 连 ``torch.cuda.empty_cache()`` 都会抛 ``CUDA error: out of memory``， 进程被硬崩（GUI
    里表现为无 traceback 的 "Unhandled Python exception"）。

    因此这里按**当前可用显存**保守给出一个具体整数，交给 RF-DETR 使用—— 只要 batch 是具体整数，RF-DETR 就会**跳过**自动探测分支，从根本上避坑。

    取值依据（实测峰值，含 Windows 桌面 ~1GB + YOLO 推理模型常驻）：
        batch 2 → 约 1.50 GB（安全）
        batch 4 → 约 2.47 GB（4GB 卡上逼近上限，余量不足易崩）
    故 4GB 级别卡封顶 2；更大显存按空闲量放宽。
    """
    try:
        import torch
    except Exception:
        return 2
    if not torch.cuda.is_available():
        return 2  # CPU 下用一个较小值，避免内存撑爆
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return 2
    free_gb = free / 1024**3
    total_gb = total / 1024**3

    # 4GB / 6GB 笔记本卡：Windows 桌面常驻吃 ~1GB，峰值随 batch 急剧上升，封顶 2
    if total_gb <= 6.0:
        return 2 if free_gb >= 1.7 else 1

    # 更大显存的卡：按训练启动时实际空闲量给，留 1.5GB 余量给探测外开销
    if free_gb >= 6.0:
        return 8
    if free_gb >= 3.5:
        return 4
    if free_gb >= 1.9:
        return 2
    return 1


class RFDETRDatasetError(Exception):
    """数据集不符合 RF-DETR 要求时抛出，消息面向终端用户，可直接显示到界面日志。."""


# --------------------------------------------------------------------------
# 运行时可用性探测（可选依赖，缺失不应影响其余功能）
# --------------------------------------------------------------------------
def rfdetr_available() -> bool:
    """Rfdetr 是否已安装。.

    刻意**不真正 import**：实测 `import rfdetr` 耗时约 22 秒（会拉起 transformers / pytorch_lightning / supervision 等重型依赖），若在 UI
    主线程调用会直接卡死界面。这里只查元数据，导入动作全部放到子线程。
    """
    try:
        from importlib.util import find_spec

        return find_spec("rfdetr") is not None
    except Exception:
        return False


def ensure_rf_home(prefer: str | None = None) -> str:
    """把 RF-DETR 权重缓存目录落到**项目所在盘**，返回最终目录。.

    背景（重要）：
        RF-DETR 首次实例化模型会自动下载预训练权重，默认缓存到
        ``~/.roboflow/models``（即 C 盘用户目录），单个 Nano 权重就有
        349MB。开发机 C 盘空间紧张，所以这里改成跟随项目目录存放；
        项目在 E 盘时权重就落在 E 盘。

    优先级：已存在的 RF_HOME 环境变量 > prefer 参数 > <项目根>/.rfdetr_models
    """
    env = os.environ.get("RF_HOME") or os.environ.get("ROBOFLOW_HOME")
    if env:
        return env
    target = prefer or str(Path(__file__).resolve().parent / ".rfdetr_models")
    os.makedirs(target, exist_ok=True)
    os.environ["RF_HOME"] = target
    return target


def rfdetr_missing_reason() -> str:
    """返回给人看的安装提示。."""
    return (
        "未检测到 rfdetr。RF-DETR 为可选功能，其余功能不受影响。\n"
        "安装命令（务必带约束文件，防止 torch 被换成 CPU 版）：\n"
        "    pip install -c constraints-torch.txt -r requirements-rfdetr.txt"
    )


# --------------------------------------------------------------------------
# 数据集适配
# --------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise RFDETRDatasetError(f"数据集配置文件不存在：{path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RFDETRDatasetError(f"无法解析 {path.name}：{e}")
    if not isinstance(data, dict):
        raise RFDETRDatasetError(f"{path.name} 内容不是合法的 YAML 映射（dict）。")
    return data


def _resolve(base: Path, value) -> Path | None:
    """把 yaml 里的路径值解析成绝对路径；空值返回 None。."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _normalize_names(names) -> dict:
    """把 names 统一成 {int_id: str_name}。支持 dict 与 list 两种写法。."""
    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                raise RFDETRDatasetError(f"names 的键必须是整数类别 ID，收到：{k!r}")
        return out
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise RFDETRDatasetError("data yaml 缺少 names（类别名）字段，无法确定类别。")


def _resolve_split_dir(root: Path, raw, want: str) -> Path | None:
    """解析某个 split 的 images / labels 目录，兼容两种常见 YOLO 写法：.

    - split-first : <root>/train/images   （RF-DETR 原生期望）
    - images-first: <root>/images/train   （部分 Roboflow / 旧导出用这种）

    以「磁盘上真实存在的目录」为准，返回绝对路径；都不存在返回 None。 want 为 "images" 或 "labels"。
    """
    if raw is None:
        return None
    raw_p = Path(str(raw).strip())
    if not str(raw_p):
        return None
    split_tokens = {"train", "val", "valid", "test"}
    parts = [p.lower() for p in raw_p.parts]
    split = next((t for t in parts if t in split_tokens), None)

    cands: list[Path] = []
    if split:
        # 两种顺序都试，再补上 yaml 可能已经写全的相对路径
        cands += [root / split / want, root / want / split, root / raw_p]
    else:
        # 只给了 split 名（如 "train"），补上 want
        cands += [root / raw_p / want, root / want / raw_p]

    seen = set()
    for c in cands:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_dir():
            return c
    return None


def _rf_swap_labels(images_dir: Path) -> Path:
    """复刻 RF-DETR 的 labels 推导规则：把路径里**任意位置**的 'images' 段换成 'labels'。.

    RF-DETR 用这个规则从 data.yaml 的 train/val（图片目录）反推 labels 目录， 因此生成的 data.yaml 里 labels 必须落在这个位置，训练才能找到标注。
    """
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def prepare_rfdetr_dataset(yaml_path: str, output_root: Path | None = None) -> dict:
    """把选中的 YOLO yaml 适配成 RF-DETR 可直接训练的数据集。.

    在**数据集根目录**生成 `data.yaml`（所有路径为绝对路径）。

    参数
    ----
    yaml_path : 用户选中的数据集 yaml（如 4940_has_labled.yaml） output_root : data.yaml 写入位置，默认写到数据集根目录

    返回
    ----
    dict: {
        dataset_dir, data_yaml, splits:{"train":..,"val":..,"test":..},
        num_classes, names, generated: bool
    }

    抛出 RFDETRDatasetError：数据集不满足要求（消息可直接显示给用户）
    """
    yaml_path = Path(yaml_path).resolve()
    cfg = _load_yaml(yaml_path)
    yaml_dir = yaml_path.parent

    # 数据集根：优先用 yaml 里的 path（相对 yaml 所在目录解析）
    root = _resolve(yaml_dir, cfg.get("path")) or yaml_dir
    if not root.is_dir():
        raise RFDETRDatasetError(f"数据集根目录不存在：{root}（来自 yaml 的 path 字段）")

    names = _normalize_names(cfg.get("names"))

    notes: list[str] = []

    def _resolve_split(name: str):
        raw = cfg.get(name)
        if name == "val" and raw is None:
            raw = cfg.get("valid")  # 兼容 valid 写法
        imgs = _resolve_split_dir(root, raw, "images")
        return raw, imgs

    # ---- train ----
    train_raw, train_imgs = _resolve_split("train")
    if train_imgs is None:
        raise RFDETRDatasetError(
            f"找不到训练集图片目录。已尝试 <根>/train/images 与 <根>/images/train 两种布局均未命中。\n"
            f"请检查 {yaml_path.name} 的 train 字段（当前值：{train_raw!r}）是否指向真实存在的目录。"
        )
    # RF-DETR 通过 swap(images->labels) 推导 labels，必须落在该位置
    train_labels = _rf_swap_labels(train_imgs)
    if not train_labels.is_dir():
        raise RFDETRDatasetError(
            f"找不到训练标签目录。\n"
            f"  图片目录：{train_imgs}\n"
            f"  RF-DETR 需要的标签目录：{train_labels}（与 images 同级的 labels）\n"
            f"请确认标注文件（.txt）放在该 labels 目录下。"
        )

    # ---- val（缺失则退化为用 train） ----
    _val_raw, val_imgs = _resolve_split("val")
    if val_imgs is None:
        val_imgs = train_imgs
        notes.append("未找到独立 val 集，已用 train 集作为验证集。")
    val_labels = _rf_swap_labels(val_imgs)
    if not val_labels.is_dir():
        if val_imgs is not train_imgs:
            raise RFDETRDatasetError(
                f"找不到验证标签目录。\n  图片目录：{val_imgs}\n  RF-DETR 需要的标签目录：{val_labels}\n"
            )
        val_labels = train_labels

    # ---- 自动适配提示：声明路径与磁盘实际不一致时记录到日志 ----
    try:
        declared = Path(str(train_raw)).as_posix().strip("/")
        actual = train_imgs.relative_to(root).as_posix()
        if declared and declared != actual:
            notes.append(f"训练集路径已自动适配：声明 {train_raw!r} → 实际使用 {actual}")
    except Exception:
        pass

    # 生成 data.yaml（绝对路径，避免 CWD 解析问题）
    target_dir = Path(output_root) if output_root else root
    target_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = target_dir / "data.yaml"

    payload = {
        "path": str(root),
        "train": str(train_imgs),
        "val": str(val_imgs),
    }
    payload["names"] = names

    header = (
        "# 本文件由 rfdetr_adapter.py 自动生成，供 RF-DETR 训练使用。\n"
        f"# 源配置：{yaml_path.name}\n"
        "# 路径均为绝对路径，避免相对路径按 CWD 解析导致的找不到图片问题。\n"
        "# RF-DETR 同时兼容 train/images 与 images/train 两种写法，本适配器会选磁盘上真实存在的目录。\n"
    )
    try:
        with open(data_yaml, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        raise RFDETRDatasetError(f"写入 {data_yaml} 失败：{e}")

    return {
        "dataset_dir": str(root),
        "data_yaml": str(data_yaml),
        "splits": {
            "train": str(train_imgs),
            "val": str(val_imgs),
            "train_labels": str(train_labels),
            "val_labels": str(val_labels),
        },
        "num_classes": len(names),
        "names": names,
        "generated": True,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# 自测：python rfdetr_adapter.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    cases = ["4940_has_labled.yaml", "3631.yaml", "不存在的.yaml"]

    for c in cases:
        print(f"\n=== {c} ===")
        try:
            info = prepare_rfdetr_dataset(here / c)
            print(f"  dataset_dir : {info['dataset_dir']}")
            print(f"  data_yaml   : {info['data_yaml']}")
            print(f"  num_classes : {info['num_classes']}  names={info['names']}")
            for k, v in info["splits"].items():
                print(f"  {k:5s}       : {v}")
        except RFDETRDatasetError as e:
            print(f"  [预期内失败] {e}")
        except Exception as e:
            print(f"  [!! 非预期异常] {type(e).__name__}: {e}")

    print(f"\nrfdetr 已安装：{rfdetr_available()}")
