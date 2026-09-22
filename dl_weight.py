# -*- coding: utf-8 -*-
"""RF-DETR 预训练权重多线程下载器（支持断点续传）。

背景：
  RF-DETR 首次训练会自动下载预训练权重，默认落在
  C:\\Users\\<用户名>\\.roboflow\\models\\，而权重源站在境外
  （storage.googleapis.com），实测单线程仅约 230KB/s —— Nano 权重 349MB
  要下 25 分钟，而且默认写 C 盘。

  本脚本改用 8 线程分块下载（实测 2.5MB/s 起步），并直接落到项目目录下，
  避免占用 C 盘。

用法：
    python dl_weight.py nano                 # 下载 Nano 权重（默认）
    python dl_weight.py small                # Small
    python dl_weight.py medium               # Medium
    python dl_weight.py nano D:\\cache       # 指定缓存目录（可选第二参数）

支持的变体（与 UI「RF-DETR」页开放的一致）：
    nano / small / medium / base / large
    UI 不放开 XLarge / 2XLarge —— 它们是 PML 1.0 许可，商用需自行确认条款。
    已存在且大小正确的权重会自动跳过，不会重复下载。

下载完成后，程序会自动从该目录加载（内置 RF_HOME 重定向）；
若要在命令行单独使用，先设置环境变量：
    set RF_HOME=<缓存目录>          (CMD)
    $env:RF_HOME = "<缓存目录>"      (PowerShell)
"""
import os
import sys
import time
import threading

import requests

# URL 与文件名取自 rfdetr 官方 rfdetr/assets/model_weights.py，
# 与 UI「RF-DETR」页开放的 5 个变体一一对应（Nano/Small/Medium/Base/Large）。
URLS = {
    "nano":   "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "small":  "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "base":   "https://storage.googleapis.com/rfdetr/rf-detr-base.pth",
    "large":  "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
}
FILENAMES = {
    "nano":   "rf-detr-nano.pth",
    "small":  "rf-detr-small.pth",
    "medium": "rf-detr-medium.pth",
    "base":   "rf-detr-base.pth",
    "large":  "rf-detr-large.pth",
}
# 官方权重体积（MB），仅用于下载前的体积提示
SIZES_MB = {
    "nano": 349, "small": 408, "medium": 776, "base": 950, "large": 1100,
}
N_THREADS = 8
CHUNK = 1024 * 512

# 默认缓存目录：<项目根>/.rfdetr_models（与程序内置重定向路径保持一致）
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, ".rfdetr_models")


def md5_of(path):
    """计算文件 MD5（分块读取，避免大文件占内存）。"""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_md5(variant):
    """从已安装的 rfdetr 里取官方 MD5；取不到返回 None。

    不硬编码 MD5 —— 权重更新时硬编码值会过期，导致误判。
    """
    try:
        from rfdetr.assets.model_weights import ModelWeights
        fn = FILENAMES[variant]
        for m in ModelWeights:
            if m.value.filename == fn:
                return m.value.md5_hash
    except Exception:
        pass
    return None


def probe_size(url):
    """探测远端文件大小，同时确认链接可访问。

    用 Range 请求而不是 HEAD：实测 HEAD 对已失效的链接会返回
    Content-Length: 0（而不是 403），容易误判成「0 字节文件」。
    返回 (是否可下载, 字节数)。
    """
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"},
                         timeout=30, stream=True)
        if r.status_code not in (200, 206):
            return False, 0
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            return True, int(cr.split("/")[-1])
        return True, int(r.headers.get("Content-Length", 0))
    except Exception:
        return False, 0


def _download(url, dest, nthreads=N_THREADS):
    ok, total = probe_size(url)
    if not ok or total <= 0:
        print("!! 该变体的官方下载源当前不可用（HTTP 403 / 链接已失效）。")
        print("   源地址:", url)
        print("   请改用其它变体，或到 Roboflow 官方仓库确认最新地址。")
        return False
    print("文件大小: %.1f MB" % (total / 1048576))

    # 预分配文件（已存在且大小一致则视为续传）
    if not os.path.exists(dest) or os.path.getsize(dest) != total:
        with open(dest, "wb") as f:
            f.truncate(total)

    part = total // nthreads
    ranges = []
    for i in range(nthreads):
        s = i * part
        e = total - 1 if i == nthreads - 1 else (s + part - 1)
        ranges.append((s, e))

    progress = {"done": 0}
    lock = threading.Lock()
    errors = []

    def worker(idx, start, end):
        try:
            with open(dest, "r+b") as f:
                f.seek(start)
                got = 0
                need = end - start + 1
                headers = {"Range": "bytes=%d-%d" % (start, end)}
                resp = requests.get(url, headers=headers, stream=True, timeout=60)
                resp.raise_for_status()
                for buf in resp.iter_content(CHUNK):
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    with lock:
                        progress["done"] += len(buf)
                if got != need:
                    errors.append("分片 %d 长度不符: %d != %d" % (idx, got, need))
        except Exception as e:
            errors.append("分片 %d 失败: %s" % (idx, e))

    t0 = time.time()
    threads = [
        threading.Thread(target=worker, args=(i, s, e), daemon=True)
        for i, (s, e) in enumerate(ranges)
    ]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(2)
        d = progress["done"]
        el = time.time() - t0
        spd = d / 1048576 / el if el > 0 else 0
        pct = d / total * 100
        eta = (total - d) / (d / el) if d > 0 else 0
        sys.stdout.write(
            "\r  %.1f%% | %.1f/%.1f MB | %.2f MB/s | 剩余 %.0fs   "
            % (pct, d / 1048576, total / 1048576, spd, eta)
        )
        sys.stdout.flush()

    for t in threads:
        t.join()

    el = time.time() - t0
    print("\n\n下载耗时 %.1fs，平均 %.2f MB/s" % (el, total / 1048576 / el))
    if errors:
        print("!! 有分片出错（重跑本脚本可续传）:")
        for e in errors[:5]:
            print("   ", e)
        return False
    size = os.path.getsize(dest)
    if size != total:
        print("!! 文件大小不符: %d != %d（重跑本脚本续传）" % (size, total))
        return False

    # ---- 关键：必须校验 MD5，不能只看大小 ----
    # 曾经踩过坑：分片写入异常时「大小一致但内容错误」，
    # 训练时 RF-DETR 会判定权重损坏并重新下载（白等几十分钟）。
    expected = expected_md5(variant)
    if expected is None:
        print("!! 无法获取官方 MD5，仅校验大小（不保证内容正确）")
        print("   大小校验通过: %s (%.1f MB)" % (dest, size / 1048576))
        return True

    print("校验 MD5（可能需要十几秒）...")
    got = md5_of(dest)
    if got != expected:
        print("!! MD5 不一致！文件已损坏：")
        print("   期望: %s" % expected)
        print("   实际: %s" % got)
        print("   请删除后重跑本脚本：del \"%s\"" % dest)
        return False
    print("MD5 校验通过: %s (%.1f MB)" % (dest, size / 1048576))
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    variant = args[0] if args else "nano"
    if variant not in URLS:
        print("未知变体:", variant, "| 可选:", list(URLS))
        return 1

    cache = args[1] if len(args) > 1 else DEFAULT_CACHE
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, FILENAMES[variant])

    print("=" * 60)
    print("变体:", variant)
    print("目标:", dest)
    print("线程:", N_THREADS)
    print("预估体积: ~%d MB" % SIZES_MB.get(variant, 0))
    print("=" * 60 + "\n")

    # 已存在时：先校验 MD5（大小一致不代表内容对），通过才跳过
    if os.path.exists(dest):
        exp = expected_md5(variant)
        if exp is None:
            ok, remote = probe_size(URLS[variant])
            if ok and os.path.getsize(dest) == remote:
                print("权重已存在且大小一致（%.1f MB），跳过。" % (remote / 1048576))
                return 0
            if not ok:
                print("远端校验失败，按正常流程下载。")
        else:
            print("发现已存在的文件，校验 MD5...")
            got = md5_of(dest)
            if got == exp:
                print("MD5 一致（%.1f MB），无需重复下载。" % (os.path.getsize(dest) / 1048576))
                return 0
            print("!! MD5 不一致，文件已损坏，删除后重新下载。")
            os.remove(dest)

    ok = _download(URLS[variant], dest)
    if ok:
        print("\n完成。程序会自动从该目录加载；命令行单独使用时请先设置：")
        print("    set RF_HOME=%s" % cache)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
