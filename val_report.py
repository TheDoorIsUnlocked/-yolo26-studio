"""模型验证结果整理、自动评价与报告生成（无 GUI 依赖，便于单元测试）。

对外提供三个核心能力：
1. collect_val_results(metrics, meta) —— 把 ultralytics 的 DetMetrics 对象整理成结构化字典，
   包含总体指标、逐类指标、推理耗时、混淆矩阵以及自动评价。
2. evaluate_results(scalar) —— 依据行业常用阈值对核心指标自动评级并给出改进建议。
3. build_markdown(results) / build_json(results) —— 生成“验证报告”，
   报告中详细说明每个参数的含义，并对验证结果给出综合评价。
"""

from __future__ import annotations

import json
from typing import Any

EPS = 1e-9

# ---------------------------------------------------------------------------
# 评价阈值（从高到低排列）。value 落在第一个满足“>= 阈值”的档位，
# 否则归入最低档“较差”。
# ---------------------------------------------------------------------------
_MAP_THRESHOLDS = [
    (0.75, "优秀"),
    (0.60, "良好"),
    (0.50, "中等"),
    (0.40, "一般"),
]  # 低于 0.40 -> 较差
_MAP50_THRESHOLDS = [
    (0.90, "优秀"),
    (0.75, "良好"),
    (0.60, "中等"),
    (0.50, "一般"),
]  # 低于 0.50 -> 较差
_PR_THRESHOLDS = [
    (0.90, "优秀"),
    (0.80, "良好"),
    (0.70, "中等"),
    (0.60, "一般"),
]  # 低于 0.60 -> 较差


def _grade(value: float, thresholds: list[tuple[float, str]]) -> str:
    """根据阈值表返回指标评级。"""
    for thr, grade in thresholds:
        if value >= thr:
            return grade
    return "较差"


# ---------------------------------------------------------------------------
# 参数含义详解：key -> (中文名, 含义说明)。报告“参数含义详解”一节使用。
# ---------------------------------------------------------------------------
METRIC_EXPLAIN = {
    "precision": (
        "精确率 (Precision)",
        "在所有被模型判定为“目标”的预测框中，真正确实是目标的比例（TP / (TP + FP)）。"
        "精确率越高，误报（把背景或无关物体错认成目标）越少。",
    ),
    "recall": (
        "召回率 (Recall)",
        "在所有真实存在的目标中，被模型成功检出的比例（TP / (TP + FN)）。"
        "召回率越高，漏检越少。精确率与召回率往往此消彼长。",
    ),
    "f1": (
        "F1 分数",
        "精确率与召回率的调和平均：F1 = 2·P·R / (P + R)。"
        "它更关注两者的短板，是单一数值综合衡量检测完整性的常用指标。",
    ),
    "map50": (
        "mAP@0.5",
        "IoU 交并比阈值为 0.5 时的平均精度均值（mean Average Precision）。"
        "定位要求较宽松，主要反映“模型能否大致把目标框住”。",
    ),
    "map": (
        "mAP@0.5:0.95",
        "在 IoU 0.50、0.55、…、0.95 共 10 个阈值上的平均 mAP 再取均值。"
        "这是目标检测领域最常用、最综合的精度指标，阈值越严格越能体现定位精准度。",
    ),
    "map75": (
        "mAP@0.75",
        "IoU 阈值为 0.75 时的平均精度均值，代表更严格（更精准）的定位能力，"
        "常用于评估模型在高精度要求场景下的表现。",
    ),
    "fitness": (
        "综合适应度 (Fitness)",
        "YOLO 训练阶段默认优化的目标，数值上等于 mAP@0.5:0.95，"
        "用于在不同训练轮次/模型之间做横向比较。",
    ),
}


def evaluate_results(scalar: dict[str, float]) -> dict[str, Any]:
    """对核心指标自动评级并生成改进建议。

    Args:
        scalar: 包含 precision / recall / map50 / map 等浮点指标的字典。

    Returns:
        {
            "overall_grade": str,
            "metrics": {显示名: {"value": float, "grade": str}},
            "suggestions": [str, ...],
        }
    """
    p = float(scalar.get("precision", 0.0) or 0.0)
    r = float(scalar.get("recall", 0.0) or 0.0)
    map50 = float(scalar.get("map50", 0.0) or 0.0)
    mapv = float(scalar.get("map", 0.0) or 0.0)

    grades = {
        "mAP@0.5:0.95": {"value": mapv, "grade": _grade(mapv, _MAP_THRESHOLDS)},
        "mAP@0.5": {"value": map50, "grade": _grade(map50, _MAP50_THRESHOLDS)},
        "精确率 (Precision)": {"value": p, "grade": _grade(p, _PR_THRESHOLDS)},
        "召回率 (Recall)": {"value": r, "grade": _grade(r, _PR_THRESHOLDS)},
    }
    overall = grades["mAP@0.5:0.95"]["grade"]

    suggestions: list[str] = []
    if p < 0.6 and p < r - 0.10:
        suggestions.append(
            "精确率明显偏低（误报偏多）：可尝试提高置信度阈值(conf)、"
            "增加难负样本挖掘或数据增强、并核查标注是否过松。"
        )
    if r < 0.6 and r < p - 0.10:
        suggestions.append(
            "召回率明显偏低（漏检偏多）：可尝试降低置信度阈值、"
            "扩充训练数据、检查标注完整性，或适当增大模型容量。"
        )
    if mapv < 0.50:
        suggestions.append(
            "mAP@0.5:0.95 偏低：建议扩充并清洗数据集、延长训练轮数、"
            "调优超参数、使用更大模型，并核查训练/验证集划分是否合理、是否存在数据泄漏。"
        )
    if not suggestions:
        suggestions.append(
            "各核心指标均处于良好区间，模型整体可用；如需进一步提升，"
            "可针对 mAP@0.5:0.95 最低的类别做定向优化（补充该类样本、检查该类标注质量）。"
        )

    return {"overall_grade": overall, "metrics": grades, "suggestions": suggestions}


def collect_val_results(metrics: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """把 ultralytics 的 DetMetrics 整理成结构化结果字典。

    Args:
        metrics: 由 model.val() 返回的 DetMetrics 对象（检测任务）。
        meta: 本次验证的元信息（模型路径、参数、运行环境、时间等）。

    Returns:
        结构化结果字典，字段含 meta / scalar / per_class / speed /
        confusion_matrix / save_dir / evaluation。
    """
    box = getattr(metrics, "box", metrics)
    raw_names = getattr(metrics, "names", None) or meta.get("names", {}) or {}
    names = raw_names if isinstance(raw_names, dict) else {}

    try:
        mp = float(getattr(box, "mp", 0.0) or 0.0)
        mr = float(getattr(box, "mr", 0.0) or 0.0)
    except Exception:
        mp = mr = 0.0
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    mapv = float(getattr(box, "map", 0.0) or 0.0)
    map75 = float(getattr(box, "map75", 0.0) or 0.0)
    try:
        fitness = float(getattr(box, "fitness")())
    except Exception:
        fitness = mapv
    f1 = (2.0 * mp * mr / (mp + mr + EPS)) if (mp + mr) > 0 else 0.0

    scalar = {
        "precision": mp,
        "recall": mr,
        "f1": f1,
        "map50": map50,
        "map": mapv,
        "map75": map75,
        "fitness": fitness,
    }

    per_class: list[dict[str, Any]] = []
    # 注意：真实 ultralytics 的 box.p / box.r / box.ap* / box.ap_class_index 都是 numpy 数组，
    # 不能用 `or []` 这种写法（会触发 "truth value of an array is ambiguous" 异常）。
    ap_idx_raw = getattr(box, "ap_class_index", None)
    ap_idx = list(ap_idx_raw) if ap_idx_raw is not None else []
    nt_img = getattr(metrics, "nt_per_image", None)
    nt_cls = getattr(metrics, "nt_per_class", None)
    p_arr = getattr(box, "p", None)
    r_arr = getattr(box, "r", None)
    ap50_arr = getattr(box, "ap50", None)
    ap_arr = getattr(box, "ap", None)

    def _safe_at(arr, i, default=0.0):
        if arr is None or i >= len(arr):
            return default
        try:
            return float(arr[i])
        except Exception:
            return default

    for i, c in enumerate(ap_idx):
        try:
            c_key = int(c)
        except Exception:
            c_key = c
        c_is_int = isinstance(c_key, int)
        per_class.append(
            {
                "class": names.get(c_key, str(c)) if c_is_int else str(c),
                "images": int(nt_img[c_key]) if (nt_img is not None and c_is_int) else None,
                "instances": int(nt_cls[c_key]) if (nt_cls is not None and c_is_int) else None,
                "precision": _safe_at(p_arr, i),
                "recall": _safe_at(r_arr, i),
                "map50": _safe_at(ap50_arr, i),
                "map": _safe_at(ap_arr, i),
            }
        )

    speed: dict[str, float] = {}
    sp = getattr(metrics, "speed", None)
    if isinstance(sp, dict):
        speed = {k: float(v) for k, v in sp.items()}

    confusion_matrix = None
    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is not None:
        mat = getattr(cm_obj, "matrix", None)
        if mat is not None:
            try:
                import numpy as np

                arr = np.asarray(mat, dtype=float)
                if arr.sum() > 0:
                    confusion_matrix = arr.tolist()
            except Exception:
                confusion_matrix = None

    results: dict[str, Any] = {
        "meta": meta,
        "scalar": scalar,
        "per_class": per_class,
        "speed": speed,
        "confusion_matrix": confusion_matrix,
        "save_dir": str(getattr(metrics, "save_dir", "") or ""),
        "names": names,
    }
    results["evaluation"] = evaluate_results(scalar)
    return results


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_markdown(results: dict[str, Any]) -> str:
    """生成 Markdown 格式的验证报告（含参数含义详解与综合评价）。"""
    meta = results.get("meta", {})
    scalar = results.get("scalar", {})
    ev = results.get("evaluation", {})
    grades = ev.get("metrics", {})

    lines: list[str] = []
    lines.append("# YOLO 模型验证报告")
    lines.append("")
    lines.append(f"- 生成时间：{meta.get('timestamp', '-')}")
    lines.append(f"- 模型文件：{meta.get('model_path', '-')}")
    lines.append(f"- 任务类型：{meta.get('task', '-')}")
    lines.append(f"- 类别数量：{meta.get('nc', '-')}")
    lines.append(f"- 数据配置：{meta.get('data_yaml', '-')}")
    lines.append(f"- 批次大小：{meta.get('batch', '-')}")
    lines.append(f"- 图像尺寸：{meta.get('imgsz', '-')}")
    lines.append(f"- 运行设备：{meta.get('device', '-')}（{meta.get('cuda_device', '-')}）")
    lines.append(f"- 验证结果目录：{results.get('save_dir', '-')}")
    lines.append(f"- Ultralytics：{meta.get('ultralytics_version', '-')} | PyTorch：{meta.get('torch_version', '-')}")
    lines.append("")

    # 一、总体评价指标
    lines.append("## 一、总体评价指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 评价 |")
    lines.append("| --- | --- | --- |")
    overall_rows = [
        ("精确率 Precision", scalar.get("precision"), "精确率 (Precision)"),
        ("召回率 Recall", scalar.get("recall"), "召回率 (Recall)"),
        ("F1 分数", scalar.get("f1"), None),
        ("mAP@0.5", scalar.get("map50"), "mAP@0.5"),
        ("mAP@0.5:0.95", scalar.get("map"), "mAP@0.5:0.95"),
        ("mAP@0.75", scalar.get("map75"), None),
        ("综合适应度 Fitness", scalar.get("fitness"), None),
    ]
    for name, val, key in overall_rows:
        grade = ""
        if key and key in grades:
            grade = grades[key].get("grade", "")
        lines.append(f"| {name} | {_fmt(val)} | {grade} |")
    lines.append("")

    # 二、参数含义详解
    lines.append("## 二、参数含义详解")
    lines.append("")
    for key in ("precision", "recall", "f1", "map50", "map", "map75", "fitness"):
        if key in METRIC_EXPLAIN:
            cn, desc = METRIC_EXPLAIN[key]
            lines.append(f"### {cn}")
            lines.append("")
            lines.append(desc)
            lines.append("")
    lines.append("> 评级标准：mAP@0.5:0.95 ≥ 0.75 优秀 / ≥ 0.60 良好 / ≥ 0.50 中等 / ≥ 0.40 一般 / 其余较差；")
    lines.append("> mAP@0.5 与 P/R 采用更宽松的对应阈值（详见上文含义）。")
    lines.append("")

    # 三、逐类结果
    lines.append("## 三、逐类检测结果")
    lines.append("")
    pc = results.get("per_class", [])
    if pc:
        lines.append("| 类别 | 图像数 | 实例数 | P | R | mAP@0.5 | mAP@0.5:0.95 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in pc:
            lines.append(
                f"| {row.get('class')} | {_fmt(row.get('images'), 0)} | {_fmt(row.get('instances'), 0)} | "
                f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | "
                f"{_fmt(row.get('map50'))} | {_fmt(row.get('map'))} |"
            )
    else:
        lines.append("_未统计到逐类结果（数据集可能无标注或验证集为空）。_")
    lines.append("")

    # 四、推理性能
    lines.append("## 四、推理性能（毫秒 / 张）")
    lines.append("")
    speed = results.get("speed", {})
    if speed:
        lines.append("| 阶段 | 耗时(ms) |")
        lines.append("| --- | --- |")
        for stage in ("preprocess", "inference", "postprocess", "loss"):
            if stage in speed:
                label = {
                    "preprocess": "前处理",
                    "inference": "推理",
                    "postprocess": "后处理",
                    "loss": "损失计算",
                }.get(stage, stage)
                lines.append(f"| {label} | {_fmt(speed[stage], 3)} |")
    else:
        lines.append("_无推理耗时数据。_")
    lines.append("")

    # 五、综合评价与建议
    lines.append("## 五、综合评价与建议")
    lines.append("")
    overall_grade = ev.get("overall_grade", "-")
    lines.append(f"**综合评价：{overall_grade}**（mAP@0.5:0.95 = {_fmt(scalar.get('map'))}）")
    lines.append("")
    lines.append("逐指标评价：")
    for name, g in grades.items():
        lines.append(f"- {name}：{g.get('grade', '-')}（{_fmt(g.get('value'))}）")
    lines.append("")
    lines.append("改进建议：")
    for i, s in enumerate(ev.get("suggestions", []), 1):
        lines.append(f"{i}. {s}")
    lines.append("")

    # 六、混淆矩阵（如有）
    cm = results.get("confusion_matrix")
    if cm:
        lines.append("## 六、混淆矩阵")
        lines.append("")
        lines.append("_（行=真实类别，列=预测类别；数值为样本计数。完整矩阵另存于验证结果目录。）_")
        lines.append("")
        names_list = [meta.get("names", {})]  # placeholder, not used below
        try:
            import numpy as np

            arr = np.asarray(cm, dtype=float)
            n = arr.shape[0]
            header = "| 真实 \\ 预测 | " + " | ".join(str(i) for i in range(n)) + " |"
            lines.append(header)
            lines.append("| --- " * (n + 1) + "|")
            for ri in range(n):
                cells = " | ".join(_fmt(arr[ri, ci], 1) for ci in range(n))
                lines.append(f"| 类别{ri} | {cells} |")
        except Exception:
            lines.append("_混淆矩阵无法渲染。_")
        lines.append("")

    return "\n".join(lines)


def build_json(results: dict[str, Any]) -> str:
    """生成 JSON 格式的验证结果（机器可读，含全部指标与评价）。"""
    return json.dumps(results, ensure_ascii=False, indent=2, default=str)
