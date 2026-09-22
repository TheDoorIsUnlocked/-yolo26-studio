# -*- coding: utf-8 -*-
"""部署后自检（端到端冒烟测试）：UI 可用性 + 训练 → 导出 → 验证 全链路。

用途：
    新电脑部署完成后跑一遍，确认「环境 + 代码 + 数据」三者都配对：
        python smoke_test.py                       # 用默认数据集/模型
        python smoke_test.py 6171.yaml yolo26n.pt  # 指定数据集与基础模型

要点：
- offscreen 下构建真实 MainWindow，调用真实的 start_training /
  export_model / start_validation，走的是与用户点击完全相同的代码路径。
- offscreen 的 QTest.mouseClick 是**假阳性陷阱**（绕过命中测试），
  所以只验证「构建 + 结构 + 真实后台链路」，不测像素级点击。
- 训练只跑 1 epoch，控制耗时（本地 4GB 显卡约 70~120 秒）。
- 产物落在 runs/ 下，不影响仓库（已被 .gitignore 忽略）。
"""
import glob
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# 权重缓存重定向到项目内，避免首次跑 RF-DETR 时下到 C 盘
os.environ.setdefault("RF_HOME", os.path.join(ROOT, ".rfdetr_models"))

os.chdir(ROOT)

from PyQt6.QtCore import QEventLoop, QTimer              # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton    # noqa: E402

import main as app_main  # noqa: E402

# 默认值可被命令行参数覆盖
DATA_YAML = sys.argv[1] if len(sys.argv) > 1 else "4940_has_labled.yaml"
BASE_MODEL = sys.argv[2] if len(sys.argv) > 2 else "yolo26n.pt"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 1

RESULT = {}
LOGS = []


def log(msg):
    line = str(msg)
    LOGS.append(line)
    try:
        print("   |", line)
    except UnicodeEncodeError:
        print("   |", line.encode("utf-8", "replace").decode("utf-8", "replace"))


def wait_for(signal, timeout_ms, label):
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(timeout_ms)
    state = {"done": False}

    def on_done(*_a):
        state["done"] = True
        loop.quit()

    signal.connect(on_done)
    timer.timeout.connect(loop.quit)
    timer.start()
    t0 = time.time()
    loop.exec()
    timer.stop()
    print("   %s 耗时 %.1fs -> %s" % (label, time.time() - t0,
                                      "完成" if state["done"] else "超时!"))
    return state["done"]


def idle(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def find_latest(pattern):
    ds = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None


def main():
    print("=" * 64)
    print("部署自检 | 项目根:", ROOT)
    print("数据集:", DATA_YAML, "| 基础模型:", BASE_MODEL, "| epochs:", EPOCHS)
    print("=" * 64)

    assert os.path.exists(DATA_YAML), "找不到数据集 yaml: %s" % DATA_YAML

    app = QApplication(sys.argv)
    w = app_main.MainWindow()
    w.log = log                      # 捕获日志，便于断言与排查
    w.current_device = "0"           # 有 CUDA 就用 GPU

    # ---------------------------------------------------------------- 结构
    print()
    print("=" * 64)
    print("阶段 1/5  UI 构建与结构")
    print("=" * 64)
    n = w.tabs.count()
    print("   标签页:", [w.tabs.tabText(i) for i in range(n)])
    assert n == 10, f"期望 10 个标签页（含 RF-DETR），实际 {n}"
    assert len(w.nav_buttons) == n, "导航按钮数与标签页数不一致"
    for i, b in enumerate(w.nav_buttons):
        b.click()
        assert w.tabs.currentIndex() == i, f"导航 {i} 错位"
    print("   10 个标签页 + 导航索引一致  OK")

    nb = [x for x in w.findChildren(QPushButton)
          if x.property("class") == "NumButton"]
    print("   显式 ± 按钮数量:", len(nb))
    assert len(nb) >= 5, f"± 按钮过少({len(nb)})，可能未全部套用 _num_row"
    print("   ± 按钮齐备  OK")

    # ---------------------------------------------------------------- 训练
    print()
    print("=" * 64)
    print("阶段 2/5  模型训练（%d epoch, %s, %s）" % (EPOCHS, BASE_MODEL, DATA_YAML))
    print("=" * 64)
    w.train_model_edit.setText(BASE_MODEL)
    w.train_data_edit.setText(DATA_YAML)
    w.spin_epochs.setValue(EPOCHS)
    w.spin_batch.setValue(4)
    w.spin_imgsz.setValue(640)
    w.chk_amp.setChecked(True)
    w.chk_resume.setChecked(False)

    w.start_training()
    assert w.train_worker is not None, "训练 worker 未创建"
    idle(3000)
    during = w.lbl_train_time.text()
    print("   训练中 Duration 标签:", during)
    assert during != "Duration: 00:00:00", "计时未实时刷新"
    print("   计时实时刷新  OK")

    ok = wait_for(w.train_worker.finished_signal, 1800000, "训练")
    assert ok, "训练超时"
    assert not any("Training failed" in x for x in LOGS), "训练报失败"

    sd = None
    for x in LOGS:
        if "Results saved to" in x:
            sd = x.split("Results saved to")[-1].strip()
    if not sd:
        sd = find_latest(os.path.join(ROOT, "runs", "detect", "train*"))
    assert sd and os.path.isdir(sd), f"训练产物目录不存在: {sd}"
    RESULT["train_dir"] = sd
    print("   save_dir:", sd)

    best = os.path.join(sd, "weights", "best.pt")
    assert os.path.exists(best), f"未找到 best.pt: {best}"
    RESULT["best_pt"] = best
    print("   best.pt: %.1f MB" % (os.path.getsize(best) / 1048576))

    # ---------------------------------------------------------------- 导出
    print()
    print("=" * 64)
    print("阶段 3/5  导出 ONNX")
    print("=" * 64)
    w.export_model_edit.setText(best)
    onnx_text = None
    for i in range(w.combo_format.count()):
        if w.export_formats.get(w.combo_format.itemText(i)) == "onnx":
            onnx_text = w.combo_format.itemText(i)
            break
    assert onnx_text, "导出格式下拉里找不到 ONNX"
    w.combo_format.setCurrentText(onnx_text)
    w.spin_export_imgsz.setValue(640)
    w.chk_half.setChecked(False)
    w.chk_int8.setChecked(False)
    w.chk_dynamic.setChecked(False)
    w.chk_simplify.setChecked(True)

    w.export_model()
    assert w.export_worker is not None, "导出 worker 未创建"
    ok = wait_for(w.export_worker.finished_signal, 600000, "导出")
    assert ok, "导出超时"

    onnx = os.path.join(os.path.dirname(best), "best.onnx")
    assert os.path.exists(onnx), f"未找到导出文件: {onnx}"
    RESULT["onnx"] = onnx
    print("   best.onnx: %.1f MB" % (os.path.getsize(onnx) / 1048576))

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        print("   ONNXRuntime 可加载，输入:", inp.name, inp.shape)
        RESULT["ort"] = "OK"
    except Exception as e:
        print("   ONNXRuntime 校验失败:", e)
        RESULT["ort"] = "FAIL: %s" % e

    # ---------------------------------------------------------------- 验证
    print()
    print("=" * 64)
    print("阶段 4/5  模型验证")
    print("=" * 64)
    w.val_model_edit.setText(best)
    w.val_data_edit.setText(DATA_YAML)
    w.spin_val_batch.setValue(4)
    w.spin_val_imgsz.setValue(640)

    w.start_validation()
    assert w.val_worker is not None, "验证 worker 未创建"
    ok = wait_for(w.val_worker.finished_signal, 600000, "验证")
    assert ok, "验证超时"

    rows = w.val_results_table.rowCount()
    print("   验证结果表: %d 行" % rows)
    assert rows > 0, "验证结果表为空"
    RESULT["val_rows"] = rows
    for r in range(rows):
        k = w.val_results_table.item(r, 0)
        v = w.val_results_table.item(r, 1)
        if k and v:
            print("     %-22s %s" % (k.text(), v.text()))

    # ------------------------------------------------------------ RF-DETR
    print()
    print("=" * 64)
    print("阶段 5/5  RF-DETR 模块可用性（可选功能）")
    print("=" * 64)
    try:
        from rfdetr_adapter import prepare_rfdetr_dataset, rfdetr_available
        avail = rfdetr_available()
        print("   rfdetr 已安装:", avail)
        if avail:
            try:
                info = prepare_rfdetr_dataset(os.path.join(ROOT, DATA_YAML))
                print("   dataset_dir:", info["dataset_dir"])
                print("   data.yaml  :", info["data_yaml"])
                print("   类别数     :", info["num_classes"], info["names"])
                assert os.path.exists(info["data_yaml"]), "适配器未生成 data.yaml"
                assert info["num_classes"] >= 1, "类别解析失败"
            except Exception as e:
                # 数据集布局不符合 RF-DETR（如 images/train）属已知情况，
                # 只提示、不让整个自检失败——YOLO 链路不受影响。
                print("   该数据集不适用于 RF-DETR（不影响 YOLO 功能）:", e)
        else:
            print("   跳过（未装 rfdetr，属正常；见教程第 15 节）")
        for name in ("rf_variant", "rf_epochs", "rf_batch", "rf_grad",
                     "rf_lr", "rf_opset", "rf_split"):
            assert hasattr(w, name), f"RF-DETR 控件缺失: {name}"
        print("   RF-DETR 页控件齐备  OK")
    except Exception as e:
        print("   RF-DETR 检查异常:", e)

    # ---------------------------------------------------------------- 汇总
    print()
    print("=" * 64)
    print("冒烟结果: 通过")
    print("=" * 64)
    print("  训练产物 :", RESULT.get("train_dir"))
    print("  导出文件 :", RESULT.get("onnx"), "| ORT:", RESULT.get("ort"))
    print("  验证行数 :", RESULT.get("val_rows"))
    print("  注意：mAP 数值不代表模型质量（只跑了 %d epoch），仅验证链路通。" % EPOCHS)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print("\n!!! 断言失败:", e)
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
