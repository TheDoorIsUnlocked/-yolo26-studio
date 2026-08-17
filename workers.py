import cv2
import time
import datetime
import os
import numpy as np
import sys

# Add project root to sys.path (works on Windows & Linux)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage
from ultralytics import YOLO
import mss

# --- Hikvision SDK Imports ---
# Adjust path if needed or use relative
import sys
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
        if hasattr(os, 'add_dll_directory'):
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
    from MvCameraControl_class import *
    from CameraParams_const import *
    from CameraParams_header import *
    from MvErrorDefine_const import *
    # from MVFGControl_class import * # Removed as we use standard camera SDK
except Exception as e:
    print(f"Warning: Hikvision SDK import failed: {e}")
    # Define placeholders for missing classes
    MvCamera = None
    MV_CC_Initialize = None
    MV_CC_Finalize = None
    
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str) # New signal for errors
    
    def __init__(self, model_path='yolo26n.pt', source=0, device='cpu', tracker=None):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.device = device
        self.tracker = tracker
        self.is_running = True
        self.model = None
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
            if not os.path.exists(self.model_path) and not self.model_path.endswith('.pt'):
                 pass
            self.model = YOLO(self.model_path)
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
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # FPS: Webcam default 30, Screen/Hikvision might vary.
            fps = 30.0 
            if cap:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: fps = 30
            
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
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight * 3).reshape(nHeight, nWidth, 3)
                        elif enPixelType == PixelType_Gvsp_RGB8_Packed:
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight * 3).reshape(nHeight, nWidth, 3)
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        elif enPixelType == PixelType_Gvsp_Mono8:
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight).reshape(nHeight, nWidth)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif enPixelType == PixelType_Gvsp_BayerRG8:
                            # Handle Bayer RG 8
                            frame = np.frombuffer(self.data_buf, dtype=np.uint8, count=nWidth * nHeight).reshape(nHeight, nWidth)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB) # or COLOR_BayerRG2BGR depending on preference, usually BGR for OpenCV
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
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
                    results = self.model.track(frame, conf=self.conf, iou=self.iou, device=self.device, tracker=self.tracker, verbose=False, persist=True)
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
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
                inference_time = speed.get('inference', 0.0)
                fps = 1000.0 / inference_time if inference_time > 0 else 0
                
                # Detailed object stats
                obj_stats = {}
                total_objects = 0
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if results[0].names:
                            name = results[0].names[cls_id]
                        else:
                            name = str(cls_id)
                        obj_stats[name] = obj_stats.get(name, 0) + 1
                        total_objects += 1
                elif hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    total_objects = len(results[0].keypoints)
                    obj_stats['Keypoints'] = total_objects
                elif hasattr(results[0], 'obb') and results[0].obb is not None:
                    total_objects = len(results[0].obb)
                    obj_stats['OBB'] = total_objects
                
                stats = {
                    'fps': fps,
                    'inference_ms': inference_time,
                    'objects': total_objects,
                    'details': obj_stats
                }
                self.stats_signal.emit(stats)
            except Exception as e:
                print(f"Inference error: {e}")
                # Don't emit error for every frame failure, but maybe log it?
                # self.error_signal.emit(f"Inference error: {e}") 
                continue

        # Cleanup
        if cap: cap.release()
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
    error_signal = pyqtSignal(str) # New signal
    
    def __init__(self, model_path='yolo26n.pt', image_path=None, auto_save=False, device='cpu'):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.device = device
        self.conf = 0.25
        self.iou = 0.45
        self.auto_save = auto_save
        self.save_dir = "runs/detect"
        self.annotated_frame = None # Store for manual save

    def run(self):
        if not self.image_path:
            return

        try:
            print(f"Processing image {self.image_path}...")
            model = YOLO(self.model_path)
            results = model(self.image_path, conf=self.conf, iou=self.iou, device=self.device)
            
            self.annotated_frame = results[0].plot()
            
            # Auto Save
            if self.auto_save:
                self.save_result()
            
            # Convert to Qt Image
            rgb_image = cv2.cvtColor(self.annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # Detailed object stats
            obj_stats = {}
            total_objects = 0
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    name = results[0].names[cls_id]
                    obj_stats[name] = obj_stats.get(name, 0) + 1
                    total_objects += 1
            elif hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                total_objects = len(results[0].keypoints)
                obj_stats['Keypoints'] = total_objects
            elif hasattr(results[0], 'obb') and results[0].obb is not None:
                total_objects = len(results[0].obb)
                obj_stats['OBB'] = total_objects

            stats = {
                'objects': total_objects,
                'inference_ms': results[0].speed.get('inference', 0.0),
                'details': obj_stats
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
    error_signal = pyqtSignal(str) # New signal
    
    def __init__(self, model_path, video_path, device='cpu', save_video=False, tracker=None):
        super().__init__()
        self.model_path = model_path
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
            # stream=True is a generator
            if self.tracker:
                results = model.track(self.video_path, save=self.save_video, conf=self.conf, iou=self.iou, device=self.device, tracker=self.tracker, stream=True, persist=True)
            else:
                results = model.predict(self.video_path, save=self.save_video, conf=self.conf, iou=self.iou, device=self.device, stream=True)
            
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
                annotated_frame = r.plot()
                if annotated_frame is not None:
                    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    self.frame_signal.emit(qt_image)

                # Detailed object stats
                obj_stats = {}
                total_objects = 0
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        name = r.names[cls_id]
                        obj_stats[name] = obj_stats.get(name, 0) + 1
                        total_objects += 1
                
                speed = r.speed
                inference_time = speed.get('inference', 0.0)
                fps = 1000.0 / inference_time if inference_time > 0 else 0
                
                stats = {
                    'fps': fps,
                    'inference_ms': inference_time,
                    'objects': total_objects,
                    'details': obj_stats
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

class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    
    def __init__(self, model_path, data_yaml, epochs, batch, imgsz, device, resume=False):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.resume = resume
        self.is_running = True
        self.start_time = None
        
    def run(self):
        try:
            action = "Resuming" if self.resume else "Starting"
            self.log_signal.emit(f"{action} training with model: {self.model_path}")
            model = YOLO(self.model_path)
            
            self.start_time = time.time()
            
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
                if not hasattr(self, 'local_epoch_count'):
                    self.local_epoch_count = 0
                self.local_epoch_count += 1
                
                avg_time_per_epoch = elapsed / self.local_epoch_count
                remaining_epochs = total_epochs - epoch
                eta_seconds = avg_time_per_epoch * remaining_epochs
                
                # Format strings
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                speed_str = f"{1/avg_time_per_epoch:.2f} epochs/s" if avg_time_per_epoch > 0 else "N/A"
                if avg_time_per_epoch > 1:
                     speed_str = f"{avg_time_per_epoch:.2f} s/epoch"
                
                stats = {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "map50": map50,
                    "elapsed": elapsed_str,
                    "eta": eta_str,
                    "speed": speed_str,
                    "eta_timestamp": (datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)).strftime("%H:%M:%S")
                }
                
                self.log_signal.emit(f"Epoch {epoch}/{total_epochs} - mAP50: {map50:.4f} - ETA: {eta_str}")
                self.progress_signal.emit(stats)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch,
                imgsz=self.imgsz,
                device=self.device,
                resume=self.resume
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
            
            metrics = model.val(
                data=self.data_yaml,
                batch=self.batch,
                imgsz=self.imgsz,
                device=self.device
            )
            
            self.log_signal.emit("Validation completed successfully!")
            self.log_signal.emit(f"mAP50: {metrics.box.map50:.4f}")
            self.log_signal.emit(f"mAP50-95: {metrics.box.map:.4f}")
            self.log_signal.emit(f"Results saved to {metrics.save_dir}")
            
        except Exception as e:
            self.log_signal.emit(f"Validation failed: {e}")
        finally:
            self.finished_signal.emit()

class BenchmarkWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, model_path, data_yaml, imgsz, device):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.device = device
        
    def run(self):
        try:
            self.log_signal.emit(f"Starting benchmark with model: {self.model_path}")
            model = YOLO(self.model_path)
            
            # Benchmark doesn't strictly require data_yaml usually, but can use it.
            # model.benchmark() args: model, data, imgsz, half, int8, device, etc.
            # Using basic args for now.
            model.benchmark(
                data=self.data_yaml,
                imgsz=self.imgsz,
                device=self.device
            )
            
            self.log_signal.emit("Benchmark completed successfully!")
            
        except Exception as e:
            self.log_signal.emit(f"Benchmark failed: {e}")
        finally:
            self.finished_signal.emit()

class ExportWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, model_path, format, imgsz, half, int8, dynamic, simplify, device):
        super().__init__()
        self.model_path = model_path
        self.format = format
        self.imgsz = imgsz
        self.half = half
        self.int8 = int8
        self.dynamic = dynamic
        self.simplify = simplify
        self.device = device
        
    def run(self):
        try:
            self.log_signal.emit(f"Exporting model to {self.format}...")
            model = YOLO(self.model_path)
            path = model.export(
                format=self.format,
                imgsz=self.imgsz,
                half=self.half,
                int8=self.int8,
                dynamic=self.dynamic,
                simplify=self.simplify,
                device=self.device
            )
            self.log_signal.emit(f"Export successful: {path}")
        except Exception as e:
            self.log_signal.emit(f"Export failed: {e}")
        finally:
            self.finished_signal.emit()
