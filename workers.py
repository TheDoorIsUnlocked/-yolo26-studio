import datetime
import os
import random
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import yaml

# Add project root to sys.path (works on Windows & Linux)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Hikvision SDK Imports ---
# Adjust path if needed or use relative
import sys

import mss
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage

import ultralytics as ultralytics_pkg
from ultralytics import YOLO
from val_report import collect_val_results

# Assume MvImport is in a sibling directory "Python/MvImport" or similar
# We need to add it to sys.path to import.
# Based on user's directory structure: e:\yolo\ultralytics-26_2\Python\MvImport
# workers.py 已位于项目根目录，SDK_PATH 直接指向根目录下的 Python\MvImport
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SDK_PATH = os.path.join(PROJECT_ROOT, "Python", "MvImport")
if SDK_PATH not in sys.path:
    sys.path.append(SDK_PATH)

# Add Runtime DLL Path
# User requested a safer location. We use "libs" directory in project root.
DLL_PATH = os.path.join(PROJECT_ROOT, "libs")
# Also keep win64 as fallback just in case
DLL_PATH_WIN64 = os.path.join(PROJECT_ROOT, "win64")

# Add to PATH environment variable for ctypes to find it
for path in [DLL_PATH, DLL_PATH_WIN64]:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        # Also add for Python 3.8+ DLL search mechanism
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except Exception as e:
                print(f"Warning: Failed to add DLL directory {path}: {e}")

# Try to load MvCameraControl.dll explicitly if needed, or rely on MVFGControl_class imports
# The user mentioned MVFGControl.dll, but LS showed MvCameraControl.lib in win64.
# Usually MVFGControl.dll depends on MvCameraControl.dll.
# Let's ensure we are importing the right thing.
# The error "Could not find module 'MVFGControl.dll'" suggests it's still not found.
# Maybe it's named differently or depends on other DLLs not in win64?
# But user said "MVFGControl.dll文件我已经放在目录的win64文件夹下了".
# We trust the user.

try:
    from CameraParams_const import *
    from CameraParams_header import *
    from MvCameraControl_class import *
    from MvErrorDefine_const import *
    # from MVFGControl_class import * # Removed as we use standard camera SDK
except Exception as e:
    print(f"Warning: Hikvision SDK import failed: {e}")
    # Define placeholders for missing classes
    MvCamera = None
    MV_CC_Initialize = None
    MV_CC_Finalize = None


def depth_stats(r):
    """提取深度结果的全局距离统计(单位:米)。无有效深度时返回 None。.

    Args:
        r: 单个 ultralytics Results 对象。

    Returns:
        dict 含 mean/min/max/center，或 None。
    """
    if getattr(r, "depth", None) is None:
        return None
    d = r.depth.data
    if hasattr(d, "cpu"):
        d = d.detach().cpu().float().numpy()
    else:
        d = np.asarray(d, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > 0  # 深度>0 视为有效像素
    if not valid.any():
        return None
    vals = d[valid]
    mean = float(vals.mean())
    nearest = float(vals.min())
    farthest = float(vals.max())
    h, w = d.shape
    center = float(d[h // 2, w // 2])
    if not center > 0:
        center = mean
    return {"mean": mean, "min": nearest, "max": farthest, "center": center}


def annotate_depth(frame, stats):
    """在深度热力图上叠加距离标注(BGR 图)。stats 为 None 时原样返回。."""
    if stats is None or frame is None:
        return frame
    color = (0, 255, 255)  # BGR 黄色
    cv2.putText(
        frame, f"Center: {stats['center']:.2f} m", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Mean: {stats['mean']:.2f} m | Near: {stats['min']:.2f} m | Far: {stats['max']:.2f} m",
        (12, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    # 图像中心十字标记
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    return frame


# 检测框颜色轮换(BGR)
_CLASS_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (0, 165, 255),
    (255, 255, 0),
    (203, 192, 255),
    (0, 255, 127),
    (255, 191, 0),
]


def annotate_object_distances(frame, det, depth_arr, depth_info):
    """在深度热力图(frame)上叠加目标检测框，并标注每个物体的距离(米)。.

    Args:
        frame: 深度热力图(BGR, 原图尺寸)。
        det: 目标检测单帧 Results(含 boxes 与 names)。
        depth_arr: 深度图 numpy (H, W, 米)，与 frame 同尺寸。
        depth_info: 全局深度统计(用于无检测时回退)。
    """
    # 无检测能力或无目标时回退到全局距离标注
    if det is None or getattr(det, "boxes", None) is None or not len(det.boxes):
        return annotate_depth(frame, depth_info)
    names = det.names
    h, w = depth_arr.shape[:2]
    for box in det.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        # 目标距离 = 框中心像素深度，无效则取框内有效深度均值
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = float(depth_arr[cy, cx]) if (0 <= cy < h and 0 <= cx < w) else 0.0
        if not dist > 0:
            region = depth_arr[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
            vals = region[region > 0]
            dist = float(vals.mean()) if len(vals) else 0.0
        label = f"{names[cls]} {dist:.2f}m" if dist > 0 else f"{names[cls]} n/a"
        color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 标签底框保证可读
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 8 if y1 - 8 >= th else y1 + th + 4
        cv2.rectangle(frame, (x1, ty - th), (x1 + tw, ty + baseline), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def build_annotated_frame(r, det_model=None, det_input=None, conf=0.25, iou=0.45, device="cpu"):
    """根据单个推理结果生成标注帧，并返回深度统计。.

    当结果含深度且提供了检测模型时，会叠加"类别+距离"目标框； 否则对深度结果叠加全局距离标注(中心/均值)，对普通任务走默认 plot。 返回 (annotated_frame, depth_info)。
    """
    annotated = r.plot()
    depth_info = None
    if getattr(r, "depth", None) is not None:
        depth_info = depth_stats(r)
        if det_model is not None and det_input is not None:
            det = det_model(det_input, conf=conf, iou=iou, device=device, verbose=False)[0]
            depth_arr = r.depth.data
            if hasattr(depth_arr, "cpu"):
                depth_arr = depth_arr.detach().cpu().float().numpy()
            annotated = annotate_object_distances(annotated, det, depth_arr, depth_info)
        else:
            annotated = annotate_depth(annotated, depth_info)
    return annotated, depth_info


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)  # New signal for errors

    def __init__(self, model_path="yolo26n.pt", source=0, device="cpu", tracker=None, det_model_path=None):
        super().__init__()
        self.model_path = model_path
        self.det_model_path = det_model_path
        self.source = source
        self.device = device
        self.tracker = tracker
        self.is_running = True
        self.model = None
        self.det_model = None
        self.conf = 0.25
        self.iou = 0.45
        self.save_video = False
        self.video_writer = None
        self.save_dir = "runs/detect"
        self.rotation_angle = 0

        # Screen Capture
        self.sct = None

        # Hikvision
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = 0

    def run(self):
        # Load model
        try:
            print(f"Loading model {self.model_path} on {self.device}...")
            if not os.path.exists(self.model_path) and not self.model_path.endswith(".pt"):
                pass
            self.model = YOLO(self.model_path)
            if self.det_model_path:
                self.det_model = YOLO(self.det_model_path)
        except Exception as e:
            err_msg = f"Error loading model: {e}"
            print(err_msg)
            self.error_signal.emit(err_msg)
            return

        cap = None

        # Source Initialization
        if self.source == "screen":
            try:
                self.sct = mss.mss()
                # Capture primary monitor by default
                monitor = self.sct.monitors[1]
                width = monitor["width"]
                height = monitor["height"]
                print(f"Screen capture started: {width}x{height}")
            except Exception as e:
                err_msg = f"Failed to start screen capture: {e}"
                self.error_signal.emit(err_msg)
                return

        elif self.source == "hikvision":
            if MvCamera is None:
                self.error_signal.emit("Hikvision SDK not available.")
                return
            try:
                # Initialize Hikvision
                self.cam = MvCamera()

                # Enum devices
                deviceList = MV_CC_DEVICE_INFO_LIST()
                tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

                ret = self.cam.MV_CC_EnumDevices(tlayerType, deviceList)
                if ret != 0:
                    raise Exception(f"Enum devices failed: {ret}")

                if deviceList.nDeviceNum == 0:
                    raise Exception("No Hikvision devices found.")

                # Select first device
                stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

                ret = self.cam.MV_CC_CreateHandle(stDeviceList)
                if ret != 0:
                    raise Exception(f"Create handle failed: {ret}")

                ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                if ret != 0:
                    raise Exception(f"Open device failed: {ret}")

                # Set Trigger Mode off
                ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
                if ret != 0:
                    print(f"Warning: Set TriggerMode failed: {ret}")

                # Get Payload Size
                stParam = MVCC_INTVALUE()
                memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

                ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
                if ret != 0:
                    raise Exception(f"Get PayloadSize failed: {ret}")

                self.nPayloadSize = stParam.nCurValue
                self.data_buf = (c_ubyte * self.nPayloadSize)()

                # Start Grabbing
                ret = self.cam.MV_CC_StartGrabbing()
                if ret != 0:
                    raise Exception(f"Start grabbing failed: {ret}")

                print("Hikvision camera started.")

                # Placeholder for width/height, updated on first frame
                width = 640
                height = 480

            except Exception as e:
                err_msg = f"Hikvision error: {e}"
                print(err_msg)
                self.error_signal.emit(err_msg)
                self.cleanup_hikvision()
                return

        else:
            # Webcam / File
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                err_msg = f"Failed to open camera source {self.source}"
                print(err_msg)
                self.error_signal.emit(err_msg)
                return
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup Video Writer if saving
        if self.save_video:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # If screen or hikvision, we might need to update width/height after first frame
            # For now, if webcam, we use cap props.

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"video_{timestamp}.avi")
            # Using XVID codec for AVI
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            # FPS: Webcam default 30, Screen/Hikvision might vary.
            fps = 30.0
            if cap:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30

            # Defer writer creation for screen/hikvision until first frame?
            # Or just create if we know dimensions.
            if self.source not in ["screen", "hikvision"]:
                # Swap width/height if rotated 90 or 270
                if self.rotation_angle in [90, 270]:
                    w, h = height, width
                else:
                    w, h = width, height
                self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                print(f"Recording video to {save_path}")

        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        while self.is_running:
            frame = None

            if self.source == "screen":
                try:
                    monitor = self.sct.monitors[1]
                    sct_img = self.sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    print(f"Screen grab error: {e}")
                    break

            elif self.source == "hikvision":
                try:
                    ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.nPayloadSize, stFrameInfo, 1000)
                    if ret == 0:
                        # Success
                        nWidth = stFrameInfo.nWidth
                        nHeight = stFrameInfo.nHeight
                        enPixelType = stFrameInfo.enPixelType

                        # Debug print only once or occasionally to verify data
                        # print(f"Got frame: {nWidth}x{nHeight}, Type: {hex(enPixelType)}")

                        # Convert to numpy
                        # Need to handle pixel format.
                        # If RGB8 or BGR8:
                        if enPixelType == PixelType_Gvsp_BGR8_Packed:
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight * 3).reshape(
                                nHeight, nWidth, 3
                            )
                        elif enPixelType == PixelType_Gvsp_RGB8_Packed:
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight * 3).reshape(
                                nHeight, nWidth, 3
                            )
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        elif enPixelType == PixelType_Gvsp_Mono8:
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight).reshape(
                                nHeight, nWidth
                            )
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif enPixelType == PixelType_Gvsp_BayerRG8:
                            # Handle Bayer RG 8
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight).reshape(
                                nHeight, nWidth
                            )
                            frame = cv2.cvtColor(
                                frame, cv2.COLOR_BayerRG2RGB
                            )  # or COLOR_BayerRG2BGR depending on preference, usually BGR for OpenCV
                        else:
                            # Try generic decode if standard reshape fails?
                            # we might need to use MV_CC_ConvertPixelType.
                            # But for now, let's try to convert to RGB using SDK if possible, or just print warning.
                            # print(f"Unsupported pixel type: {hex(enPixelType)}")

                            # Fallback: Try to use SDK to convert to RGB8 if possible?
                            # For simplicity in this fix, we'll try to just skip.
                            # But if the camera is Mono8 (common), the above check handles it.
                            # If it's Bayer, we need conversion.

                            # Try generic decode if standard reshape fails?
                            # Let's add Bayer support if common.
                            # PixelType_Gvsp_BayerGR8 = 0x01080008, etc.
                            # For now, let's assume user might be using Bayer.

                            # Attempt to convert using SDK (more robust)
                            # We need a buffer for converted image.
                            # Let's just create a buffer large enough for RGB

                            nConvertSize = nWidth * nHeight * 3
                            if self.nPayloadSize < nConvertSize:
                                # We might need a separate buffer for conversion
                                pass

                            # Use internal buffer for now if possible?
                            # Since we are in Python, calling ConvertPixelType is a bit complex with ctypes pointers.
                            # Let's try to assume it works if we configured camera to RGB/BGR if possible?
                            # But we didn't set pixel format.

                            continue

                    else:
                        # Timeout or error
                        continue
                except Exception as e:
                    print(f"Hikvision grab error: {e}")
                    continue

            else:
                ret, frame = cap.read()
                if not ret:
                    break

            if frame is None:
                continue

            # Init writer for Screen/Hik if needed
            if self.save_video and self.video_writer is None:
                h, w = frame.shape[:2]
                if self.rotation_angle in [90, 270]:
                    w, h = h, w
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f"video_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))
                print(f"Recording video to {save_path}")

            # Rotate Frame
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Inference
            try:
                if self.tracker:
                    results = self.model.track(
                        frame,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        tracker=self.tracker,
                        verbose=False,
                        persist=True,
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)

                # Get annotated frame
                annotated_frame, depth_info = build_annotated_frame(
                    results[0], self.det_model, frame, self.conf, self.iou, self.device
                )

                # Save frame if recording
                if self.video_writer:
                    self.video_writer.write(annotated_frame)

                # Convert to Qt Image
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

                self.change_pixmap_signal.emit(p.copy())

                # Emit stats
                speed = results[0].speed
                inference_time = speed.get("inference", 0.0)
                fps = 1000.0 / inference_time if inference_time > 0 else 0

                # Detailed object stats
                obj_stats = {}
                total_objects = 0
                if hasattr(results[0], "boxes") and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if results[0].names:
                            name = results[0].names[cls_id]
                        else:
                            name = str(cls_id)
                        obj_stats[name] = obj_stats.get(name, 0) + 1
                        total_objects += 1
                elif hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                    total_objects = len(results[0].keypoints)
                    obj_stats["Keypoints"] = total_objects
                elif hasattr(results[0], "obb") and results[0].obb is not None:
                    total_objects = len(results[0].obb)
                    obj_stats["OBB"] = total_objects

                stats = {
                    "fps": fps,
                    "inference_ms": inference_time,
                    "objects": total_objects,
                    "details": obj_stats,
                    "depth": depth_info,
                }
                self.stats_signal.emit(stats)
            except Exception as e:
                print(f"Inference error: {e}")
                # Don't emit error for every frame failure, but maybe log it?
                # self.error_signal.emit(f"Inference error: {e}")
                continue

        # Cleanup
        if cap:
            cap.release()
        if self.video_writer:
            self.video_writer.release()
            print("Video recording saved.")
        self.cleanup_hikvision()

    def cleanup_hikvision(self):
        if self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except:
                pass
        self.cam = None

    def stop(self):
        self.is_running = False
        self.wait()

    def update_params(self, conf, iou):
        self.conf = conf
        self.iou = iou

    def set_save(self, save):
        self.save_video = save

    def set_rotation(self, angle):
        self.rotation_angle = angle

    def update_device(self, device):
        self.device = device


class ImageWorker(QThread):
    result_signal = pyqtSignal(QImage, dict)
    error_signal = pyqtSignal(str)  # New signal

    def __init__(self, model_path="yolo26n.pt", image_path=None, auto_save=False, device="cpu", det_model_path=None):
        super().__init__()
        self.model_path = model_path
        self.det_model_path = det_model_path
        self.image_path = image_path
        self.device = device
        self.conf = 0.25
        self.iou = 0.45
        self.auto_save = auto_save
        self.save_dir = "runs/detect"
        self.annotated_frame = None  # Store for manual save

    def run(self):
        if not self.image_path:
            return

        try:
            print(f"Processing image {self.image_path}...")
            model = YOLO(self.model_path)
            det_model = YOLO(self.det_model_path) if self.det_model_path else None
            results = model(self.image_path, conf=self.conf, iou=self.iou, device=self.device)

            # 诊断：打印 GUI 实际加载的类别名，用于排查标签是否颠倒
            import ultralytics as _ul

            print(f"[DIAG] model_path={self.model_path}")
            print(f"[DIAG] ultralytics={_ul.__version__} names={model.names}")
            print(f"[DIAG] det_model_path={self.det_model_path}")

            self.annotated_frame, depth_info = build_annotated_frame(
                results[0], det_model, self.image_path, self.conf, self.iou, self.device
            )

            # Auto Save
            if self.auto_save:
                self.save_result()

            # Convert to Qt Image
            rgb_image = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

            # Detailed object stats (数量 + 每类最高置信度)
            obj_stats = {}
            obj_conf = {}
            total_objects = 0
            if hasattr(results[0], "boxes") and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = results[0].names[cls_id]
                    obj_stats[name] = obj_stats.get(name, 0) + 1
                    obj_conf[name] = max(obj_conf.get(name, 0.0), conf)
                    total_objects += 1
            elif hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                total_objects = len(results[0].keypoints)
                obj_stats["Keypoints"] = total_objects
            elif hasattr(results[0], "obb") and results[0].obb is not None:
                total_objects = len(results[0].obb)
                obj_stats["OBB"] = total_objects

            stats = {
                "objects": total_objects,
                "inference_ms": results[0].speed.get("inference", 0.0),
                "details": obj_stats,
                "conf": obj_conf,
                "depth": depth_info,
            }

            self.result_signal.emit(qt_image, stats)
            print("Image processing done.")

        except Exception as e:
            err_msg = f"Error in ImageWorker: {e}"
            print(err_msg)
            self.error_signal.emit(err_msg)

    def save_result(self):
        if self.annotated_frame is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            basename = os.path.basename(self.image_path)
            name, ext = os.path.splitext(basename)
            save_path = os.path.join(self.save_dir, f"{name}_{timestamp}{ext}")
            cv2.imwrite(save_path, self.annotated_frame)
            print(f"Saved image to {save_path}")


class VideoFileWorker(QThread):
    progress_signal = pyqtSignal(int)
    frame_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)  # New signal

    def __init__(self, model_path, video_path, device="cpu", save_video=False, tracker=None, det_model_path=None):
        super().__init__()
        self.model_path = model_path
        self.det_model_path = det_model_path
        self.video_path = video_path
        self.device = device
        self.tracker = tracker
        self.conf = 0.25
        self.iou = 0.45
        self.save_video = save_video
        self.is_running = True

    def run(self):
        print(f"Starting video processing: {self.video_path}, save={self.save_video}")
        try:
            model = YOLO(self.model_path)
            det_model = YOLO(self.det_model_path) if self.det_model_path else None
            # stream=True is a generator
            if self.tracker:
                results = model.track(
                    self.video_path,
                    save=self.save_video,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    tracker=self.tracker,
                    stream=True,
                    persist=True,
                )
            else:
                results = model.predict(
                    self.video_path, save=self.save_video, conf=self.conf, iou=self.iou, device=self.device, stream=True
                )

            # Get total frames for progress
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Total frames: {total_frames}")

            count = 0
            save_dir = ""
            for r in results:
                if not self.is_running:
                    print("Video processing stopped by user.")
                    break
                count += 1
                save_dir = r.save_dir

                # Update progress
                if total_frames > 0:
                    progress = int((count / total_frames) * 100)
                    self.progress_signal.emit(progress)

                # Emit Frame
                annotated_frame, depth_info = build_annotated_frame(
                    r, det_model, getattr(r, "orig_img", None), self.conf, self.iou, self.device
                )
                if annotated_frame is not None:
                    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    self.frame_signal.emit(qt_image)

                # Detailed object stats
                obj_stats = {}
                total_objects = 0
                if hasattr(r, "boxes") and r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        name = r.names[cls_id]
                        obj_stats[name] = obj_stats.get(name, 0) + 1
                        total_objects += 1

                speed = r.speed
                inference_time = speed.get("inference", 0.0)
                fps = 1000.0 / inference_time if inference_time > 0 else 0

                stats = {
                    "fps": fps,
                    "inference_ms": inference_time,
                    "objects": total_objects,
                    "details": obj_stats,
                    "depth": depth_info,
                }
                self.stats_signal.emit(stats)

            print(f"Video processing finished. Save dir: {save_dir}")
            self.finished_signal.emit(str(save_dir) if save_dir else "")

        except Exception as e:
            err_msg = f"Error in VideoFileWorker: {e}"
            print(err_msg)
            import traceback

            traceback.print_exc()
            self.error_signal.emit(err_msg)
            self.finished_signal.emit("")

    def stop(self):
        self.is_running = False


# ====================== 离线图像增强 (训练前预处理) ======================


class AugmentWorker(QThread):
    """离线图像增强: 读取数据集 -> 生成增强副本到新目录 -> 产出新的 data.yaml。.

    设计要点(重要):
    * 原图与对应 label 原样**复制**到输出目录(自包含完整数据集),
        增强副本追加在后; 用户的原始数据集文件绝不会被修改或覆盖。
    * 只增强 train 集; val/test 原样复制, 避免验证集泄漏增强样本导致指标虚高。
    * 几何变换(旋转/镜像)会同步变换 YOLO label; 颜色变换不影响 label。
    """

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # (current, total)
    finished_signal = pyqtSignal(str, dict)  # (new_yaml_path, stats)

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, data_yaml, cfg):
        super().__init__()
        self.data_yaml = data_yaml
        self.cfg = dict(cfg or {})
        self._stop = False

    # ---------------- YOLO label 几何 ----------------
    @staticmethod
    def _yolo_to_corners(cx, cy, w, h, iw, ih):
        """YOLO 归一化 (cx,cy,w,h) -> 4 个角点像素坐标。."""
        x1 = (cx - w / 2.0) * iw
        y1 = (cy - h / 2.0) * ih
        x2 = (cx + w / 2.0) * iw
        y2 = (cy + h / 2.0) * ih
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    @staticmethod
    def _corners_to_yolo(corners, iw, ih):
        """4 个角点像素坐标 -> YOLO 归一化 (cx,cy,w,h); 目标被裁出画面返回 None。."""
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        x_min = max(0.0, float(x_min))
        y_min = max(0.0, float(y_min))
        x_max = min(float(iw), float(x_max))
        y_max = min(float(ih), float(y_max))
        # 目标几乎被裁没了 -> 丢弃该框, 避免产生噪声标签
        if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
            return None
        return ((x_min + x_max) / 2.0 / iw, (y_min + y_max) / 2.0 / ih, (x_max - x_min) / iw, (y_max - y_min) / ih)

    @staticmethod
    def _load_labels(path):
        """读取 YOLO txt: 每行 `cls cx cy w h`(归一化)。."""
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                try:
                    rows.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                except ValueError:
                    continue
        return rows

    @staticmethod
    def _save_labels(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cls, cx, cy, w, h in rows)

    # ---------------- 单类变换 ----------------
    def _apply_rotate(self, img, labels, angle):
        """旋转图像, 并用同一仿射矩阵变换 label 角点。."""
        ih, iw = img.shape[:2]
        M = cv2.getRotationMatrix2D((iw / 2.0, ih / 2.0), angle, 1.0)
        out = cv2.warpAffine(img, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        new = []
        for cls, cx, cy, w, h in labels:
            pts = self._yolo_to_corners(cx, cy, w, h, iw, ih)
            ones = np.ones((4, 1), dtype=np.float32)
            moved = (M @ np.hstack([pts, ones]).T).T
            yolo = self._corners_to_yolo(moved, iw, ih)
            if yolo:
                new.append((cls, *yolo))
        return out, new

    @staticmethod
    def _apply_flip(img, labels, code):
        """Code: 1=水平, 0=垂直, -1=水平+垂直。label 只需镜像中心点。."""
        out = cv2.flip(img, code)
        new = []
        for cls, cx, cy, w, h in labels:
            ncx, ncy = cx, cy
            if code in (1, -1):
                ncx = 1.0 - cx
            if code in (0, -1):
                ncy = 1.0 - cy
            new.append((cls, ncx, ncy, w, h))
        return out, new

    @staticmethod
    def _apply_brightness(img, delta):
        return cv2.convertScaleAbs(img, alpha=1.0, beta=float(delta))

    @staticmethod
    def _apply_contrast(img, percent):
        return cv2.convertScaleAbs(img, alpha=1.0 + float(percent) / 100.0, beta=0)

    @staticmethod
    def _apply_saturation(img, percent):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + float(percent) / 100.0), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ---------------- 变体生成 ----------------
    def _make_variants(self, img, labels):
        """按配置生成变体: {文件名后缀: (图, labels)}。.

        颜色类变换随机取正/负方向, 让样本分布更对称。
        """
        out = {}

        # 镜像
        m = str(self.cfg.get("mirror", "none") or "none").lower()
        if m in ("horizontal", "h"):
            out["mirror_h"] = self._apply_flip(img, labels, 1)
        elif m in ("vertical", "v"):
            out["mirror_v"] = self._apply_flip(img, labels, 0)
        elif m in ("both", "b"):
            out["mirror_b"] = self._apply_flip(img, labels, -1)

        # 旋转步长: 90 -> 90/180/270; 180 -> 180; 0 -> 不生成
        step = int(self.cfg.get("rot_step", 0) or 0)
        if step > 0 and 360 % step == 0:
            for ang in range(step, 360, step):
                out[f"rot{ang}"] = self._apply_rotate(img, labels, ang)

        # 旋转范围: 随机 ±range
        if self.cfg.get("rot_range", False):
            r = float(self.cfg.get("rot_range_val", 0) or 0)
            if r > 0:
                out["rotr"] = self._apply_rotate(img, labels, random.uniform(-r, r))

        # 亮度 (随机 ±)
        if self.cfg.get("brightness", False):
            d = float(self.cfg.get("brightness_val", 0) or 0)
            if d:
                out["bright"] = (self._apply_brightness(img, random.choice([-d, d])), list(labels))

        # 亮度变化点 (固定取正向, 作为第二档亮度)
        if self.cfg.get("brightness_pt", False):
            d = float(self.cfg.get("brightness_pt_val", 0) or 0)
            if d:
                out["brightpt"] = (self._apply_brightness(img, d), list(labels))

        # 对比度 (随机 ±)
        if self.cfg.get("contrast", False):
            p = float(self.cfg.get("contrast_val", 0) or 0)
            if p:
                out["contrast"] = (self._apply_contrast(img, random.choice([-p, p])), list(labels))

        # 饱和度 (随机 ±)
        if self.cfg.get("saturation", False):
            p = float(self.cfg.get("saturation_val", 0) or 0)
            if p:
                out["satur"] = (self._apply_saturation(img, random.choice([-p, p])), list(labels))

        return out

    # ---------------- 主流程 ----------------
    def run(self):
        stats = {"orig": 0, "gen": 0, "skipped": 0, "out_dir": ""}
        try:
            with open(self.data_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            yaml_dir = os.path.dirname(os.path.abspath(self.data_yaml))
            base = data.get("path", "") or ""
            root = base if os.path.isabs(base) else os.path.normpath(os.path.join(yaml_dir, base))

            train_rel = data.get("train", "") or ""
            if not train_rel:
                raise RuntimeError("data.yaml 中缺少 train 字段")
            src_img_dir = os.path.normpath(train_rel if os.path.isabs(train_rel) else os.path.join(root, train_rel))
            src_lbl_dir = src_img_dir.replace("images", "labels")
            if not os.path.isdir(src_img_dir):
                raise FileNotFoundError(f"训练集图片目录不存在: {src_img_dir}")

            train_rel_in_root = os.path.relpath(src_img_dir, root)  # 如 train/images
            out_root = root.rstrip("\\/") + "_aug"
            out_img_dir = os.path.join(out_root, train_rel_in_root)
            out_lbl_dir = out_img_dir.replace("images", "labels")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)
            stats["out_dir"] = out_root

            imgs = [f for f in sorted(os.listdir(src_img_dir)) if f.lower().endswith(self.IMG_EXTS)]
            total = len(imgs)
            if total == 0:
                raise RuntimeError(f"训练集目录中没有图片: {src_img_dir}")

            self.log_signal.emit(f"[增强] 输出目录: {out_root}")
            self.log_signal.emit(f"[增强] 训练集: {src_img_dir} ({total} 张), 标签: {src_lbl_dir}")

            # ---- 1) 复制原图 + label (自包含; 原文件只读, 绝不改动) ----
            self.log_signal.emit("[增强] 步骤 1/2: 复制原图与标签 ...")
            for i, name in enumerate(imgs):
                if self._stop:
                    break
                shutil.copy2(os.path.join(src_img_dir, name), os.path.join(out_img_dir, name))
                stem = os.path.splitext(name)[0]
                src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
                if os.path.isfile(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(out_lbl_dir, stem + ".txt"))
                stats["orig"] += 1
                if i % 50 == 0 or i == total - 1:
                    self.progress_signal.emit(i + 1, total * 2)

            # ---- 2) 生成增强副本 ----
            if not self._stop:
                percent = int(self.cfg.get("percent", 100) or 100)
                pool = imgs if percent >= 100 else random.sample(imgs, max(1, round(total * percent / 100.0)))
                self.log_signal.emit(f"[增强] 步骤 2/2: 生成增强副本 (对 {len(pool)}/{total} 张原图) ...")

                for i, name in enumerate(pool):
                    if self._stop:
                        break
                    stem, ext = os.path.splitext(name)
                    img = cv2.imread(os.path.join(src_img_dir, name))
                    if img is None:
                        stats["skipped"] += 1
                        continue
                    labels = self._load_labels(os.path.join(src_lbl_dir, stem + ".txt"))
                    for suffix, (vimg, vlbl) in self._make_variants(img, labels).items():
                        cv2.imwrite(os.path.join(out_img_dir, f"{stem}_{suffix}{ext}"), vimg)
                        self._save_labels(os.path.join(out_lbl_dir, f"{stem}_{suffix}.txt"), vlbl)
                        stats["gen"] += 1
                    if i % 20 == 0 or i == len(pool) - 1:
                        self.progress_signal.emit(total + i + 1, total * 2)

            # ---- 3) 原样复制 val/test (不增强, 避免验证集泄漏) ----
            if not self._stop:
                for key in ("val", "test"):
                    rel = data.get(key, "") or ""
                    if not rel:
                        continue
                    src = os.path.normpath(rel if os.path.isabs(rel) else os.path.join(root, rel))
                    if not os.path.isdir(src):
                        continue
                    dst = os.path.join(out_root, os.path.relpath(src, root))
                    os.makedirs(dst, exist_ok=True)
                    n = 0
                    for f in os.listdir(src):
                        if f.lower().endswith(self.IMG_EXTS):
                            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                            n += 1
                    slbl, dlbl = src.replace("images", "labels"), dst.replace("images", "labels")
                    if os.path.isdir(slbl):
                        os.makedirs(dlbl, exist_ok=True)
                        for f in os.listdir(slbl):
                            if f.lower().endswith(".txt"):
                                shutil.copy2(os.path.join(slbl, f), os.path.join(dlbl, f))
                    self.log_signal.emit(f"[增强] 复制 {key} 集 {n} 张 (原样, 不增强)")

            # ---- 4) 写出新的 data.yaml ----
            new_yaml = os.path.join(yaml_dir, os.path.basename(out_root) + ".yaml")
            new_data = dict(data)
            # path 写绝对路径: YOLO 对相对 path 是按"当前工作目录"解析的
            # (见 ultralytics/data/utils.py:604 check_det_dataset), 若程序从别的目录启动
            # 会因找不到数据而报错, 故这里固定为绝对路径。
            new_data["path"] = out_root.replace("\\", "/")
            new_data["train"] = train_rel_in_root.replace("\\", "/")
            for key in ("val", "test"):
                if data.get(key):
                    new_data[key] = os.path.relpath(os.path.join(root, data[key]), root).replace("\\", "/")
            with open(new_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(new_data, f, allow_unicode=True, sort_keys=False)

            self.log_signal.emit(f"[增强] 新配置: {new_yaml}")
            self.log_signal.emit(
                f"[增强] 完成: 原图 {stats['orig']} 张 + 新增 {stats['gen']} 张"
                + (f" (跳过 {stats['skipped']} 张)" if stats["skipped"] else "")
            )
            self.finished_signal.emit(new_yaml, stats)

        except Exception as e:
            self.log_signal.emit(f"[增强] 失败: {e}")
            self.finished_signal.emit("", stats)

    def stop(self):
        self._stop = True


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, data_yaml, epochs, batch, imgsz, device, resume=False, amp=True):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.resume = resume
        self.amp = amp
        self.is_running = True
        self.start_time = None

        # 实时计时字段: 主线程 QTimer 每秒读取这些值, 让 Duration/ETA/Est.Finish
        # 在一个 epoch 期间也能持续变化(否则这些值只在 epoch 边界跳一次)。
        self.t0 = None  # 训练开始时刻 (datetime)
        self.total_epochs = 0  # 总 epoch 数
        self.epoch_done = 0  # 已完成 epoch 数
        self.avg_epoch = 0.0  # 平均每个 epoch 耗时(秒)
        self.last_epoch_end = None  # 上一个 epoch 结束时刻
        self.batch_i = 0  # 当前 epoch 内已完成 batch 数
        self.nb = 0  # 每个 epoch 的总 batch 数

    def run(self):
        try:
            action = "Resuming" if self.resume else "Starting"
            self.log_signal.emit(f"{action} training with model: {self.model_path}")
            model = YOLO(self.model_path)

            self.start_time = time.time()
            self.t0 = datetime.datetime.now()
            self.total_epochs = self.epochs

            # Add custom callback to capture logs
            def on_train_epoch_end(trainer):
                if not self.is_running:
                    raise InterruptedError("Training stopped by user")

                epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                metrics = trainer.metrics
                map50 = metrics.get("metrics/mAP50(B)", 0.0)

                # Calculate timing
                current_time = time.time()
                elapsed = current_time - self.start_time

                # Let's track local epochs
                if not hasattr(self, "local_epoch_count"):
                    self.local_epoch_count = 0
                self.local_epoch_count += 1

                avg_time_per_epoch = elapsed / self.local_epoch_count
                remaining_epochs = total_epochs - epoch
                eta_seconds = avg_time_per_epoch * remaining_epochs

                # 供主线程 QTimer 每秒实时计算 Duration/ETA/Est.Finish
                self.epoch_done = self.local_epoch_count
                self.avg_epoch = avg_time_per_epoch
                self.last_epoch_end = datetime.datetime.now()
                self.total_epochs = total_epochs

                # Format strings
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

                speed_str = f"{1 / avg_time_per_epoch:.2f} epochs/s" if avg_time_per_epoch > 0 else "N/A"
                if avg_time_per_epoch > 1:
                    speed_str = f"{avg_time_per_epoch:.2f} s/epoch"

                stats = {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "map50": map50,
                    "elapsed": elapsed_str,
                    "eta": eta_str,
                    "speed": speed_str,
                    "eta_timestamp": (datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)).strftime(
                        "%H:%M:%S"
                    ),
                }

                self.log_signal.emit(f"Epoch {epoch}/{total_epochs} - mAP50: {map50:.4f} - ETA: {eta_str}")
                self.progress_signal.emit(stats)

            def on_train_batch_end(trainer):
                """记录 epoch 内的 batch 进度。.

                第一个 epoch 还没跑完时尚无 avg_epoch, UI 可据此按 batch 比例外推 ETA。
                这里只做两次整数赋值, 不影响训练速度。
                """
                try:
                    self.batch_i = int(getattr(trainer, "batch", 0) or 0) + 1
                    self.nb = int(getattr(trainer, "nb", 0) or 0)
                except Exception:
                    pass

            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch,
                imgsz=self.imgsz,
                device=self.device,
                resume=self.resume,
                amp=self.amp,
            )

            self.log_signal.emit("Training completed successfully!")
            self.log_signal.emit(f"Results saved to {results.save_dir}")

        except InterruptedError:
            self.log_signal.emit("Training stopped by user.")
        except Exception as e:
            self.log_signal.emit(f"Training failed: {e}")
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False


class ValWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    results_signal = pyqtSignal(dict)

    def __init__(self, model_path, data_yaml, batch, imgsz, device):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.batch = batch
        self.imgsz = imgsz
        self.device = device

    def run(self):
        try:
            self.log_signal.emit(f"Starting validation with model: {self.model_path}")
            model = YOLO(self.model_path)

            metrics = model.val(data=self.data_yaml, batch=self.batch, imgsz=self.imgsz, device=self.device)

            self.log_signal.emit("Validation completed successfully!")
            self.log_signal.emit(f"mAP50: {metrics.box.map50:.4f}")
            self.log_signal.emit(f"mAP50-95: {metrics.box.map:.4f}")
            self.log_signal.emit(f"Results saved to {metrics.save_dir}")

            # 整理结构化验证结果（含总体指标、逐类指标、耗时、混淆矩阵与自动评价）
            try:
                import torch

                cuda_device = (
                    torch.cuda.get_device_name(0)
                    if (self.device != "cpu" and torch.cuda.is_available())
                    else ("CPU" if self.device == "cpu" else str(self.device))
                )
                torch_version = torch.__version__
            except Exception:
                cuda_device = str(self.device)
                torch_version = "unknown"

            meta = {
                "model_path": self.model_path,
                "data_yaml": self.data_yaml,
                "batch": self.batch,
                "imgsz": self.imgsz,
                "device": self.device,
                "task": getattr(model, "task", "detect"),
                "nc": len(getattr(model, "names", {}) or {}),
                "names": getattr(model, "names", {}) or {},
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ultralytics_version": getattr(ultralytics_pkg, "__version__", "unknown"),
                "torch_version": torch_version,
                "cuda_device": cuda_device,
            }
            results = collect_val_results(metrics, meta)
            self.results_signal.emit(results)
            self.log_signal.emit("Validation results collected for display/export.")

        except Exception as e:
            self.log_signal.emit(f"Validation failed: {e}")
        finally:
            self.finished_signal.emit()


class BenchmarkWorker(QThread):
    log_signal = pyqtSignal(str)
    results_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, fmt, imgsz, device):
        super().__init__()
        self.model_path = model_path
        self.fmt = fmt
        self.imgsz = imgsz
        self.device = device

    def run(self):
        try:
            # 仅 CPU 才能跑的格式（OpenVINO/TFLite 等在本机不支持 GPU）
            bench_device = self.device if self.fmt in ("pytorch", "onnx", "torchscript", "engine") else "cpu"
            self.log_signal.emit(f"Benchmarking {self.fmt} at imgsz={self.imgsz} on {bench_device}...")
            model = YOLO(self.model_path)

            # 参数量
            params = 0
            try:
                model_model = getattr(model, "model", None)
                if model_model is not None:
                    params = sum(p.numel() for p in model_model.parameters())
                else:
                    params = int(model.info().get("parameters", 0))
            except Exception:
                params = 0

            # 非 PyTorch 格式需先导出
            run_path = self.model_path
            if self.fmt != "pytorch":
                run_path = model.export(format=self.fmt, imgsz=self.imgsz, device=bench_device)

            bm = YOLO(run_path)
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)

            # 预热
            for _ in range(3):
                bm.predict(dummy, imgsz=self.imgsz, device=bench_device, verbose=False)

            # 计时
            n = 20
            t0 = time.perf_counter()
            for _ in range(n):
                bm.predict(dummy, imgsz=self.imgsz, device=bench_device, verbose=False)
            latency = (time.perf_counter() - t0) / n * 1000.0

            result = {
                "format": self.fmt,
                "device": bench_device,
                "imgsz": self.imgsz,
                "params": params / 1e6,
                "latency_ms": latency,
                "fps": 1000.0 / latency if latency > 0 else 0.0,
            }
            self.results_signal.emit(result)
            self.log_signal.emit(f"[{self.fmt}] {latency:.2f} ms/image, {1000.0 / latency:.1f} FPS")

        except Exception as e:
            self.log_signal.emit(f"Benchmark failed for {self.fmt}: {e}")
        finally:
            self.finished_signal.emit()


class ExportWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, format, imgsz, half, int8, dynamic, simplify, device, calib_data=None):
        super().__init__()
        self.model_path = model_path
        self.format = format
        self.imgsz = imgsz
        self.half = half
        self.int8 = int8
        self.dynamic = dynamic
        self.simplify = simplify
        self.device = device
        self.calib_data = calib_data

    def run(self):
        try:
            self.log_signal.emit(f"Exporting model to {self.format}...")
            model = YOLO(self.model_path)
            path = model.export(
                format=self.format,
                imgsz=self.imgsz,
                half=self.half,
                int8=self.int8,
                data=self.calib_data,
                dynamic=self.dynamic,
                simplify=self.simplify,
                device=self.device,
            )
            self.log_signal.emit(f"Export successful: {path}")
        except Exception as e:
            self.log_signal.emit(f"Export failed: {e}")
        finally:
            self.finished_signal.emit()


# ---------------------------------------------------------------------------
# DINOv3 异常检测 worker(建库 / 验证 / 导出 ONNX)
# ---------------------------------------------------------------------------


def _import_dino_anomaly():
    """按需导入 anomaly 核心库,并把它所在目录加入 sys.path。."""
    anomaly_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "DINVo3", "anomaly")
    if anomaly_dir not in sys.path:
        sys.path.insert(0, anomaly_dir)
    import dino_anomaly

    return dino_anomaly


class AnomalyBuildWorker(QThread):
    """良品特征库构建:良品目录 → .npz(内部同步产出 .fbin)。."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    result_signal = pyqtSignal(dict)

    def __init__(self, model_name, weights, repo_dir, good_dir, out_path, bank_size, device):
        super().__init__()
        self.model_name = model_name
        self.weights = weights
        self.repo_dir = repo_dir
        self.good_dir = good_dir
        self.out_path = out_path
        self.bank_size = bank_size
        self.device = device

    def run(self):
        try:
            da = _import_dino_anomaly()
            self.log_signal.emit(f"加载骨干 {self.model_name} ...")
            model, patch_size, device = da.load_model(
                self.model_name,
                "facebookresearch/dinov2",
                self.device,
                weights_path=self.weights,
                repo_dir=self.repo_dir,
            )
            paths = da.list_images(self.good_dir)
            if not paths:
                self.log_signal.emit(f"错误:目录下没有可用图片 {self.good_dir}")
                return
            self.log_signal.emit(f"开始建库:{len(paths)} 张良品图")
            info = da.build_bank(
                model,
                patch_size,
                paths,
                self.out_path,
                device,
                self.bank_size,
                log=lambda m: self.log_signal.emit(str(m)),
                model_name=self.model_name,
            )
            self.result_signal.emit(info)
            self.log_signal.emit("建库完成")
        except Exception as e:
            self.log_signal.emit(f"建库失败:{e}")
        finally:
            self.finished_signal.emit()


class AnomalyValidateWorker(QThread):
    """验证特征库:良品/缺陷目录 → AUROC、建议阈值、热力图。."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    result_signal = pyqtSignal(list)

    def __init__(self, model_name, weights, repo_dir, bank_path, good_dir, ng_dir, out_dir, device, top_k_heatmap=20):
        super().__init__()
        self.model_name = model_name
        self.weights = weights
        self.repo_dir = repo_dir
        self.bank_path = bank_path
        self.good_dir = good_dir
        self.ng_dir = ng_dir
        self.out_dir = out_dir
        self.device = device
        self.top_k_heatmap = top_k_heatmap

    def run(self):
        try:
            import torch

            da = _import_dino_anomaly()
            device = da.norm_device(self.device)
            bank, meta = da.load_bank(self.bank_path)
            self.log_signal.emit(f"特征库:{bank.shape[0]} 条 ({meta})")
            model, patch_size, device = da.load_model(
                self.model_name, "facebookresearch/dinov2", device, weights_path=self.weights, repo_dir=self.repo_dir
            )
            bank_t = torch.from_numpy(bank).to(device)
            items = [(p, 0) for p in da.list_images(self.good_dir)]
            items += [(p, 1) for p in da.list_images(self.ng_dir)]
            if not items:
                self.log_signal.emit("错误:良品与缺陷目录都为空")
                return
            self.log_signal.emit(f"开始验证:{len(items)} 张图")
            lines = da.validate_dataset(
                model,
                patch_size,
                bank_t,
                items,
                self.out_dir,
                device,
                self.top_k_heatmap,
                log=lambda m: self.log_signal.emit(str(m)),
            )
            self.result_signal.emit(list(lines))
            self.log_signal.emit("验证完成")
        except Exception as e:
            self.log_signal.emit(f"验证失败:{e}")
        finally:
            self.finished_signal.emit()


class AnomalyExportWorker(QThread):
    """导出 DINO 骨干为 ONNX(供服务端 / C++ 推理使用)。."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    result_signal = pyqtSignal(str)

    def __init__(self, model_name, weights, repo_dir, out_path, device):
        super().__init__()
        self.model_name = model_name
        self.weights = weights
        self.repo_dir = repo_dir
        self.out_path = out_path
        self.device = device

    def run(self):
        try:
            _import_dino_anomaly()  # 确保 anomaly 目录已在 sys.path 中
            import export_dino_onnx

            path, img_size, shape = export_dino_onnx.export_onnx(
                self.model_name,
                self.weights,
                self.repo_dir,
                self.out_path,
                self.device,
                log=lambda m: self.log_signal.emit(str(m)),
            )
            self.result_signal.emit(path)
            self.log_signal.emit(f"导出完成:{path}")
            self.log_signal.emit(f"  输入 1x3x{img_size}x{img_size},输出 {shape}")
        except Exception as e:
            self.log_signal.emit(f"导出失败:{e}")
        finally:
            self.finished_signal.emit()


# ==========================================================================
# RF-DETR（可选功能）
# --------------------------------------------------------------------------
# 设计要点：
#   * rfdetr 是**可选依赖**，所有 import 都放在 run() 里（子线程执行），
#     未安装时其余功能完全不受影响。
#   * `import rfdetr` 实测约 22 秒（拉起 transformers/PL/supervision），
#     绝不能在主线程做 —— UI 层的可用性探测走 find_spec（见 rfdetr_adapter）。
#   * 训练进度用 PyTorch Lightning Callback 上报（on_train_epoch_end），
#     不解析 stdout/tqdm —— 后者格式一变就崩。
#   * 权重缓存目录由 ensure_rf_home() 重定向到项目所在盘，避免占 C 盘。
# ==========================================================================

# 变体名 -> rfdetr 顶层类名（XLarge/2XLarge 为 PML 1.0 许可，刻意不开放）
RFDETR_VARIANTS = {
    "Nano": "RFDETRNano",
    "Small": "RFDETRSmall",
    "Medium": "RFDETRMedium",
    "Base": "RFDETRBase",
    "Large": "RFDETRLarge",
}


class _LogEmitter:
    r"""把 RF-DETR / PyTorch Lightning 的 stdout+stderr 按行转发到 Qt 信号。.

    tqdm 进度条用 \\r 而不是 \\n 刷行，所以 \\r 也按行边界处理， 否则进度条会把日志区刷爆。
    """

    def __init__(self, emit, on_line=None):
        self._emit = emit
        self._on_line = on_line  # 可选：逐行钩子，用于从日志里解析进度
        self._buf = ""

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        while True:
            # 同时兼容 \n 与 \r（tqdm）
            i_n = self._buf.find("\n")
            i_r = self._buf.find("\r")
            idxs = [i for i in (i_n, i_r) if i >= 0]
            if not idxs:
                break
            i = min(idxs)
            line, self._buf = self._buf[:i], self._buf[i + 1 :]
            line = line.strip()
            if line:
                self._emit(line)
                if self._on_line is not None:
                    try:
                        self._on_line(line)
                    except Exception:
                        pass
        return len(s)

    def flush(self):
        if self._buf.strip():
            line = self._buf.strip()
            self._emit(line)
            if self._on_line is not None:
                try:
                    self._on_line(line)
                except Exception:
                    pass
            self._buf = ""

    def isatty(self):
        return False


def _import_rfdetr_variant(variant: str):
    """导入并返回指定变体类。variant 为 RFDETR_VARIANTS 的键（如 "Nano"）。."""
    import rfdetr

    cls_name = RFDETR_VARIANTS.get(variant)
    if cls_name is None:
        raise ValueError(f"未知 RF-DETR 变体: {variant}（可选: {list(RFDETR_VARIANTS)}）")
    return getattr(rfdetr, cls_name)


# 说明：RF-DETR 的 TrainConfig 是 pydantic 模型，**不接受** PyTorch Lightning
# 的 callbacks 参数（传进去会报 "Unknown parameter(s): 'callbacks'"）。
# 因此 epoch 进度改为在 RFDETRTrainWorker._on_log_line 里正则解析训练日志获得。


class RFDETRTrainWorker(QThread):
    """RF-DETR 训练（子进程隔离版）。.

    根因说明：RF-DETR 训练在「GUI 同一进程内」运行时，会在首个训练步（或更早的模型 构建阶段）触发无 traceback 的进程级硬崩，GUI 表现为 "Unhandled Python exception"。
    经系统二分定位确认：

    * 同一份训练在独立 Python 进程（CLI 同步/异步）、在后台 threading 线程、
        甚至在「YOLO 推理模型常驻显存」的前提下跑，都 100% 正常完成；
    * 唯独在 PyQt GUI 进程内崩溃，且崩溃点不固定、无 Python traceback、
        faulthandler 也抓不到 → 属于 C 层 abort（CUDA 上下文与 Qt 事件循环 /
        常驻推理模型在同一进程内的线程/显存冲突）。

    因此这里的实现改为：在**独立子进程**里跑已经验证可用的 ``train_rfdetr.py``， GUI 只负责流式读取子进程的标准输出、解析 epoch 进度、以及按需终止子进程。 这样既彻底隔离了 GUI 进程的
    CUDA/Qt 冲突，又让训练拿到一块干净的显存池， 对外信号接口（log/progress/finished）与计时字段保持不变，主窗口逻辑无需改动。
    """

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(bool, str)  # (是否成功, 说明)

    def __init__(
        self,
        dataset_yaml,
        variant,
        epochs,
        batch_size,
        grad_accum_steps,
        lr,
        output_dir,
        resolution=None,
        resume=None,
        use_ema=False,
        num_workers=0,
        device=None,
        early_stopping=False,
        script_path=None,
    ):
        super().__init__()
        self.dataset_yaml = dataset_yaml
        self.variant = variant
        self.dataset_dir = None  # 由 run() 在启动子进程前填充，仅用于日志
        self.epochs = epochs
        self.batch_size = batch_size  # int（GUI 侧已把 auto 解析为具体整数）
        self.grad_accum_steps = grad_accum_steps
        self.lr = lr
        self.output_dir = output_dir
        self.resolution = resolution
        self.resume = resume
        self.use_ema = use_ema
        self.num_workers = num_workers
        self.device = device
        self.early_stopping = early_stopping
        self.script_path = script_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_rfdetr.py")

        # 计时字段（主窗口 QTimer 每秒读取，用于 Duration/ETA/Est.Finish）
        self.t0 = None
        self.total_epochs = epochs
        self.epoch_done = 0
        self.batch_i = 0
        self._stop_requested = False  # 必须在 __init__ 里初始化：
        # stop() 可能早于 run() 被调用
        self._proc = None

    def stop(self):
        """协作式停止：设置标志并终止子进程。."""
        self._stop_requested = True
        self.log_signal.emit("收到停止请求，正在终止训练子进程…")
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _emit_progress(self, line: str):
        """从训练日志里解析 epoch 进度并转发（形如「Val (Epoch 1/10) …」）。."""
        try:
            m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line)
            if not m:
                return
            ep, total = int(m.group(1)), int(m.group(2))
            self.epoch_done = ep
            self.progress_signal.emit(
                {
                    "epoch": ep,
                    "total": total,
                    "elapsed": time.time() - (self.t0 or time.time()),
                }
            )
        except Exception:
            pass

    def run(self):
        self._stop_requested = False
        self.t0 = time.time()

        # 1) 数据集适配（生成 RF-DETR 用的 data.yaml，路径全部绝对化）
        try:
            from rfdetr_adapter import ensure_rf_home, prepare_rfdetr_dataset

            ensure_rf_home()
            info = prepare_rfdetr_dataset(self.dataset_yaml)
            self.dataset_dir = info["dataset_dir"]
            data_yaml = info["data_yaml"]
            for _n in info.get("notes", []):
                self.log_signal.emit("· " + str(_n))
            self.log_signal.emit(f"数据集就绪：{info['dataset_dir']}（{info['num_classes']} 类）")
        except Exception as e:
            self.log_signal.emit(f"数据集适配失败：{e}")
            self.finished_signal.emit(False, str(e))
            return

        # 2) 组装与 train_rfdetr.py 完全一致的命令行
        cmd = [
            sys.executable,
            self.script_path,
            "--data",
            data_yaml,
            # 子进程脚本 train_rfdetr.py 的 --variant 只接受小写
            # （nano/small/medium/base/large），GUI 下拉框值是首字母大写（Nano），需归一化
            "--variant",
            str(self.variant).lower(),
            "--epochs",
            str(self.epochs),
            "--batch",
            str(self.batch_size),
            "--grad-accum",
            str(self.grad_accum_steps),
            "--lr",
            str(self.lr),
        ]
        out_parent = os.path.dirname(self.output_dir.rstrip(os.sep)) or "."
        run_name = os.path.basename(self.output_dir.rstrip(os.sep))
        cmd += ["--output", out_parent, "--run-name", run_name]
        if self.resolution:
            cmd += ["--resolution", str(self.resolution)]
        if not self.use_ema:
            cmd += ["--no-ema"]
        if self.device:
            cmd += ["--device", str(self.device)]
        if self.resume:
            cmd += ["--resume", str(self.resume)]
        if self.early_stopping:
            cmd += ["--early-stop"]

        self.log_signal.emit("启动独立训练子进程（隔离 GUI 进程的 CUDA/线程冲突，避免闪退）…")
        self.log_signal.emit("CMD: " + " ".join(cmd))

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        # 3) 启动子进程并流式转发输出
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                cwd=os.path.dirname(os.path.abspath(self.script_path)),
                creationflags=creationflags,
            )
        except Exception as e:
            self.log_signal.emit(f"启动训练子进程失败：{e}")
            self.finished_signal.emit(False, str(e))
            return

        epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
        try:
            for line in self._proc.stdout:
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                self.log_signal.emit(line)
                if epoch_re.search(line):
                    self._emit_progress(line)
                if self._stop_requested:
                    self._proc.terminate()
                    break
        except Exception as e:
            self.log_signal.emit(f"读取训练输出异常：{e}")

        # 4) 等待子进程结束并汇报结果
        try:
            rc = self._proc.wait(timeout=60)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
            rc = self._proc.wait()

        if self._stop_requested:
            self.log_signal.emit("已取消训练（子进程已终止）")
            self.finished_signal.emit(False, "已取消")
            return

        if rc == 0:
            self.finished_signal.emit(True, f"训练完成，产物目录: {self.output_dir}")
        else:
            self.log_signal.emit(f"训练子进程异常退出（返回码 {rc}）")
            self.finished_signal.emit(False, f"训练失败，返回码 {rc}")


class RFDETREvalWorker(QThread):
    """RF-DETR 验证：加载 checkpoint 并在指定 split 上跑 COCO 评估。."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    result_signal = pyqtSignal(dict)  # COCO 指标字典

    def __init__(self, ckpt_path, dataset_dir, split="test", num_workers=0, batch_size=1, resolution=None, device=None):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.resolution = resolution
        self.device = device

    def run(self):
        import contextlib

        try:
            try:
                from rfdetr_adapter import ensure_rf_home

                ensure_rf_home()
            except Exception:
                pass

            self.log_signal.emit("正在导入 rfdetr（首次较慢，约 20 秒）…")
            import rfdetr

            self.log_signal.emit(f"加载权重: {self.ckpt_path}")
            model = rfdetr.RFDETR.from_checkpoint(self.ckpt_path)

            kwargs = {
                "dataset_dir": self.dataset_dir,
                "dataset_file": "yolo",
                "split": self.split,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
            }
            if self.resolution:
                kwargs["resolution"] = self.resolution
            if self.device:
                kwargs["device"] = self.device

            self.log_signal.emit(f"开始评估（split={self.split}）…")
            emitter = _LogEmitter(self.log_signal.emit)
            with contextlib.redirect_stdout(emitter), contextlib.redirect_stderr(emitter):
                metrics = model.evaluate(**kwargs)
            emitter.flush()

            metrics = dict(metrics or {})
            self.log_signal.emit("评估完成:")
            for k, v in metrics.items():
                try:
                    self.log_signal.emit(f"  {k}: {float(v):.4f}")
                except Exception:
                    self.log_signal.emit(f"  {k}: {v}")
            self.result_signal.emit(metrics)
            self.finished_signal.emit(True, "评估完成")
        except Exception as e:
            import traceback

            self.log_signal.emit("RF-DETR 验证失败:")
            for ln in traceback.format_exc().splitlines()[-12:]:
                self.log_signal.emit("  " + ln)
            self.finished_signal.emit(False, str(e))


class RFDETRExportWorker(QThread):
    """RF-DETR 导出（本期仅 ONNX）。."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    result_signal = pyqtSignal(str)  # 导出的 onnx 路径

    def __init__(
        self, ckpt_path, output_dir, opset_version=17, resolution=None, batch_size=1, dynamic_batch=False, fp16=True
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.opset_version = opset_version
        self.resolution = resolution
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch
        self.fp16 = fp16

    def run(self):
        import contextlib

        try:
            try:
                from rfdetr_adapter import ensure_rf_home

                ensure_rf_home()
            except Exception:
                pass

            self.log_signal.emit("正在导入 rfdetr（首次较慢，约 20 秒）…")
            import rfdetr

            self.log_signal.emit(f"加载权重: {self.ckpt_path}")
            model = rfdetr.RFDETR.from_checkpoint(self.ckpt_path)

            kwargs = {
                "output_dir": self.output_dir,
                "format": "onnx",
                "opset_version": self.opset_version,
                "batch_size": self.batch_size,
                "dynamic_batch": self.dynamic_batch,
                "fp16": self.fp16,
                "verbose": False,
            }
            if self.resolution:
                # export 的 shape 是 (height, width)，且必须是 patch_size*num_windows 的倍数
                kwargs["shape"] = (self.resolution, self.resolution)

            self.log_signal.emit(
                f"开始导出 ONNX | opset {self.opset_version} | "
                f"shape {kwargs.get('shape', '默认')} | batch {self.batch_size}"
            )
            emitter = _LogEmitter(self.log_signal.emit)
            with contextlib.redirect_stdout(emitter), contextlib.redirect_stderr(emitter):
                path = model.export(**kwargs)
            emitter.flush()

            path_str = str(path)
            self.log_signal.emit(f"导出完成: {path_str}")
            self.result_signal.emit(path_str)
            self.finished_signal.emit(True, path_str)
        except Exception as e:
            import traceback

            self.log_signal.emit("RF-DETR 导出失败:")
            for ln in traceback.format_exc().splitlines()[-12:]:
                self.log_signal.emit("  " + ln)
            self.finished_signal.emit(False, str(e))
