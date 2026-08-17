
import sys
import os
import glob
import multiprocessing
import cv2

# Add project root to sys.path (works on Windows & Linux)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFrame, QDockWidget, QComboBox,
                             QSlider, QGroupBox, QListWidget, QTextEdit, QTabWidget,
                             QFileDialog, QProgressBar, QSplitter, QScrollArea, QCheckBox, QSpinBox, QButtonGroup, QRadioButton, QToolButton, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QSize, pyqtSlot, QUrl, QTimer, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon, QAction, QShortcut, QKeySequence, QPainter, QFont, QColor, QWheelEvent
from PyQt6.QtMultimedia import QMediaDevices

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from styles import Theme
from workers import VideoThread, ImageWorker, TrainWorker, VideoFileWorker, ExportWorker, ValWorker, BenchmarkWorker
from config import Config
from val_report import build_markdown, build_json

def emoji_to_pixmap(emoji, size=48):
    """Render an emoji to a QPixmap."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Use a standard emoji-capable font if possible, though default usually works
    # Ensure size is positive to avoid QFont warnings
    font_size = max(1, size - 10)
    font = QFont("Segoe UI Emoji", font_size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    painter.setFont(font)
    
    # Let's use a neutral or theme-based color. 
    # Current style uses TEXT_SUB for NavButton. 
    text_color = Theme.get_colors()["TEXT_SUB"]
    painter.setPen(QColor(text_color))
    
    painter.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, emoji)
    painter.end()
    return pixmap

class ZoomableLabel(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
        self.setAcceptDrops(True)
        
        self._pixmap = None
        self._scale_factor = 1.0
        self._pan_start = QPoint()
        self._pan_offset = QPoint(0, 0)
        self._is_panning = False

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        if not self._pixmap:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Calculate scaled dimensions
        scaled_w = int(self._pixmap.width() * self._scale_factor)
        scaled_h = int(self._pixmap.height() * self._scale_factor)

        # Calculate position to center the image + pan offset
        x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
        y = (self.height() - scaled_h) // 2 + self._pan_offset.y()

        # Draw
        target_rect = QRect(x, y, scaled_w, scaled_h)
        painter.drawPixmap(target_rect, self._pixmap)

    def wheelEvent(self, event: QWheelEvent):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        
        old_scale = self._scale_factor
        new_scale = self._scale_factor * factor
        
        # Limit scale
        if 0.1 < new_scale < 10.0:
            self._scale_factor = new_scale
            
            # Adjust pan offset to zoom towards mouse cursor (optional but better UX)
            # For simplicity, we just zoom center for now or keep existing pan relative?
            # Keeping it simple: just zoom, existing pan offset remains valid-ish but might drift.
            # Ideally we adjust offset to keep mouse point stable.
            
            # Calculate mouse position relative to image center (before zoom)
            # center_x = self.width() // 2 + self._pan_offset.x()
            # center_y = self.height() // 2 + self._pan_offset.y()
            # mouse_rel_x = event.position().x() - center_x
            # mouse_rel_y = event.position().y() - center_y
            
            # This is complex to get right quickly without glitching. 
            # Let's stick to simple zoom first, maybe reset pan if zoomed out?
            
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_offset += delta
            self._pan_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """Reset zoom and pan on double click."""
        self._scale_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            # Emit signal for the first file
            self.fileDropped.emit(files[0])

class ResizableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(100, 100) # Allow shrinking
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow expansion
        self._original_pixmap = None
        self._scale_factor = 1.0

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self.update_display()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)
        
    def wheelEvent(self, event: QWheelEvent):
        angle = event.angleDelta().y()
        if angle > 0:
            self._scale_factor *= 1.1
        else:
            self._scale_factor *= 0.9
        
        # Clamp
        self._scale_factor = max(0.1, min(self._scale_factor, 5.0))
        self.update_display()

    def mouseDoubleClickEvent(self, event):
        """Reset zoom on double click."""
        self._scale_factor = 1.0
        self.update_display()

    def update_display(self):
        if self._original_pixmap:
            # Calculate target size based on widget size * zoom
            target_w = int(self.width() * self._scale_factor)
            target_h = int(self.height() * self._scale_factor)
            
            scaled = self._original_pixmap.scaled(
                 target_w, target_h,
                 Qt.AspectRatioMode.KeepAspectRatio,
                 Qt.TransformationMode.SmoothTransformation
             )
            super().setPixmap(scaled)




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(Config.get("title"))
        self.resize(1280, 800)
        
        # State
        self.current_model = "yolo26n.pt"
        self.thread = None
        self.image_worker = None
        self.train_worker = None
        self.video_file_worker = None
        self.export_worker = None
        self.val_worker = None
        self.benchmark_worker = None
        self.last_val_results = None
        self.current_device = 'cpu'
        
        # Apply Theme
        self.apply_theme()
        
        # Setup UI
        self.init_ui()
        
        # Shortcuts
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self.open_file)
        
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_result)
        
        # Performance Monitor Timer
        self.start_perf_monitoring()

        self.image_list = []
        self.current_image_index = -1

    def apply_theme(self):
        self.setStyleSheet(Theme.get_stylesheet())

    def init_ui(self):
        # 1. Top Bar
        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(10, 5, 10, 5)
        
        self.logo_label = QLabel(f"🤖 {Config.get('title')}")
        self.logo_label.setObjectName("LogoTitle")
        
        # Language Switcher
        self.lang_btn = QPushButton(Config.get("lang_en") if Config.LANG == "CN" else Config.get("lang_cn"))
        self.lang_btn.setProperty("class", "SecondaryButton")
        self.lang_btn.clicked.connect(self.toggle_lang)
        
        # Theme Switcher
        self.theme_btn = QPushButton(Config.get("theme_light") if Theme.CURRENT_THEME == "Dark" else Config.get("theme_dark"))
        self.theme_btn.setProperty("class", "SecondaryButton")
        self.theme_btn.clicked.connect(self.toggle_theme)

        self.status_label = QLabel(Config.get("status_idle"))
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setStyleSheet(f"color: {Theme.get_colors()['ACCENT']};")
        
        top_layout.addWidget(self.logo_label)
        top_layout.addStretch()
        top_layout.addWidget(self.lang_btn)
        top_layout.addWidget(self.theme_btn)
        top_layout.addWidget(self.status_label)
        
        # 2. Side Navigation
        self.side_nav = QFrame()
        self.side_nav.setObjectName("SideNav")
        nav_layout = QVBoxLayout(self.side_nav)
        nav_layout.setContentsMargins(5, 10, 5, 10)
        nav_layout.setSpacing(10)
        
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = []
        # Store items as instance variable for access in update_ui_text
        self.nav_items = [
            ("🔍", "realtime", 0), # Dashboard removed/merged into realtime as default
            ("🖼️", "image", 1),
            ("📹", "video", 2),
            ("📊", "train", 3),
            ("✅", "val", 4),
            ("📤", "export", 5),
            ("📈", "benchmark", 6),
            ("⚙️", "settings", 7)
        ]
        
        for icon, key, idx in self.nav_items:
            name = Config.get(key)
            btn = QToolButton()
            btn.setText(name)
            # Create icon from emoji
            btn.setIcon(QIcon(emoji_to_pixmap(icon, 48)))
            btn.setIconSize(QSize(40, 40))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            
            btn.setCheckable(True)
            btn.setProperty("class", "NavButton")
            btn.setFixedSize(90, 80)
            self.nav_group.addButton(btn, idx)
            btn.clicked.connect(lambda checked, i=idx: self.switch_tab(i))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        # Select first one by default
        self.nav_buttons[0].setChecked(True)
        nav_layout.addStretch()
        
        # 3. Main Workspace
        self.main_workspace = QWidget()
        main_layout = QVBoxLayout(self.main_workspace)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(False) # Disable closing for main tabs
        self.tabs.tabBar().setVisible(False) # Hide tab bar, use side nav
        
        # Initialize Tabs
        self.realtime_tab = self.create_realtime_tab()
        self.image_tab = self.create_image_tab()
        self.video_tab = self.create_video_tab()
        self.train_tab = self.create_train_tab()
        self.val_tab = self.create_val_tab()
        self.export_tab = self.create_export_tab()
        self.benchmark_tab = self.create_benchmark_tab()
        self.settings_tab = QWidget() # Placeholder
        
        self.tabs.addTab(self.realtime_tab, "Real-time")
        self.tabs.addTab(self.image_tab, "Image")
        self.tabs.addTab(self.video_tab, "Video")
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.val_tab, "Val")
        self.tabs.addTab(self.export_tab, "Export")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")
        self.tabs.addTab(self.settings_tab, "Settings")
        
        # Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setPlaceholderText("System logs...")
        
        # Splitter for Tabs and Console
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.console)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # 4. Side Panel (Dock)
        self.dock = QDockWidget(Config.get("control_panel"), self)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        
        dock_content = QWidget()
        dock_content.setObjectName("SidePanelContent")
        dock_layout = QVBoxLayout(dock_content)
        
        # Model Selector
        self.model_group = QGroupBox(Config.get("model_sel"))
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.refresh_btn = QPushButton(Config.get("refresh_models"))
        self.refresh_btn.setProperty("class", "SecondaryButton")
        self.refresh_btn.clicked.connect(self.refresh_models)
        
        self.refresh_models()
        self.model_combo.currentTextChanged.connect(self.change_model)

        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.refresh_btn)
        self.model_group.setLayout(model_layout)

        # Device Selector
        self.device_group = QGroupBox(Config.get("device"))
        device_layout = QVBoxLayout()
        self.radio_cpu = QRadioButton("CPU")
        self.radio_gpu = QRadioButton("GPU (CUDA)")
        # Disable the GPU option if CUDA is not usable in this environment (e.g. CPU-only torch)
        try:
            import torch
            _cuda_available = torch.cuda.is_available()
        except Exception:
            _cuda_available = False
        if not _cuda_available:
            self.radio_gpu.setEnabled(False)
            self.radio_gpu.setToolTip("CUDA 不可用：当前 PyTorch 为 CPU 版本或未检测到可用显卡驱动，请安装 CUDA 版 PyTorch")
        self.radio_cpu.setChecked(True)
        self.radio_cpu.toggled.connect(self.change_device)
        self.radio_gpu.toggled.connect(self.change_device)
        device_layout.addWidget(self.radio_cpu)
        device_layout.addWidget(self.radio_gpu)
        self.device_group.setLayout(device_layout)
        
        # Performance Monitor
        self.perf_group = QGroupBox(Config.get("perf_mon"))
        perf_layout = QVBoxLayout()
        self.lbl_cpu = QLabel("CPU: 0%")
        self.lbl_ram = QLabel("RAM: 0%")
        self.lbl_gpu = QLabel("GPU: N/A")
        perf_layout.addWidget(self.lbl_cpu)
        perf_layout.addWidget(self.lbl_ram)
        perf_layout.addWidget(self.lbl_gpu)
        self.perf_group.setLayout(perf_layout)
        
        # Parameters
        self.param_group = QGroupBox(Config.get("params"))
        param_layout = QVBoxLayout()
        
        self.lbl_conf = QLabel(Config.get("conf"))
        param_layout.addWidget(self.lbl_conf)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_params)
        param_layout.addWidget(self.conf_slider)
        
        self.lbl_iou = QLabel(Config.get("iou"))
        param_layout.addWidget(self.lbl_iou)
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_params)
        param_layout.addWidget(self.iou_slider)
        
        # Initialize labels
        self.update_params()
        
        self.param_group.setLayout(param_layout)
        
        # Results
        self.result_group = QGroupBox(Config.get("results"))
        result_layout = QVBoxLayout()
        self.result_list = QListWidget()
        self.fps_label = QLabel("FPS: 0")
        result_layout.addWidget(self.fps_label)
        result_layout.addWidget(self.result_list)
        self.result_group.setLayout(result_layout)
        
        dock_layout.addWidget(self.model_group)
        dock_layout.addWidget(self.device_group)
        dock_layout.addWidget(self.perf_group)
        dock_layout.addWidget(self.param_group)
        dock_layout.addWidget(self.result_group)
        dock_layout.addStretch()
        
        self.dock.setWidget(dock_content)
        
        # Assemble Main Layout
        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        
        central_layout.addWidget(self.side_nav)
        
        # Right side container (TopBar + Workspace)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.top_bar)
        right_layout.addWidget(self.main_workspace)
        
        central_layout.addWidget(right_container)
        
        self.setCentralWidget(central_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        
        self.log("System initialized.")

    def create_realtime_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Use ZoomableLabel for automatic scaling and zoom
        self.video_label = ZoomableLabel()
        self.video_label.fileDropped.connect(self.process_dropped_file)
        # Remove fixed size constraint logic that might prevent shrinking
        # self.video_label.setMinimumSize(640, 480) 
        
        controls = QHBoxLayout()
        
        # Camera Index Selector
        self.lbl_cam_idx = QLabel(Config.get("cam_idx"))
        controls.addWidget(self.lbl_cam_idx)
        self.cam_combo = QComboBox()
        
        # Populate cameras
        cameras = QMediaDevices.videoInputs()
        cam_list = []
        for i, cam in enumerate(cameras):
            cam_list.append(f"{i}: {cam.description()}")
            
        # Add "Screen" and "Hikvision" options
        self.cam_combo.addItems(cam_list + ["Screen", "Hikvision"])
        
        # If no cameras found, fallback or at least show Screen/Hikvision
        if not cameras:
             # Just in case user has no standard cameras but wants to use Screen/Hikvision
             # The combo box will just have Screen and Hikvision
             pass
             
        controls.addWidget(self.cam_combo)
        
        # Rotation
        controls.addWidget(QLabel("Rotation:"))
        self.rot_combo = QComboBox()
        self.rot_combo.addItems(["0", "90", "180", "270"])
        self.rot_combo.setCurrentText("180")
        self.rot_combo.currentTextChanged.connect(self.change_rotation)
        controls.addWidget(self.rot_combo)
        
        # Tracker (New)
        controls.addWidget(QLabel(Config.get("tracker")))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["None", "bytetrack.yaml", "botsort.yaml"])
        controls.addWidget(self.tracker_combo)
        
        self.btn_start = QPushButton(Config.get("start_cam"))
        self.btn_start.setProperty("class", "ActionButton")
        self.btn_start.clicked.connect(self.toggle_camera)
        
        self.chk_auto_save_video = QCheckBox(Config.get("auto_save"))
        self.chk_auto_save_video.stateChanged.connect(self.toggle_video_save)
        
        controls.addStretch()
        controls.addWidget(self.btn_start)
        controls.addWidget(self.chk_auto_save_video)
        controls.addStretch()
        
        layout.addWidget(self.video_label)
        layout.addLayout(controls)
        return tab

    def create_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.image_label = ZoomableLabel()
        
        controls = QHBoxLayout()
        self.btn_open_img = QPushButton(Config.get("open_img"))
        self.btn_open_img.setProperty("class", "ActionButton")
        self.btn_open_img.clicked.connect(self.open_image)
        
        self.btn_open_folder = QPushButton(Config.get("open_folder"))
        self.btn_open_folder.setProperty("class", "ActionButton")
        self.btn_open_folder.clicked.connect(self.open_folder)
        
        self.btn_prev_img = QPushButton(Config.get("prev_image"))
        self.btn_prev_img.setProperty("class", "ActionButton")
        self.btn_prev_img.clicked.connect(self.prev_image)
        self.btn_prev_img.setEnabled(False)
        
        self.btn_next_img = QPushButton(Config.get("next_image"))
        self.btn_next_img.setProperty("class", "ActionButton")
        self.btn_next_img.clicked.connect(self.next_image)
        self.btn_next_img.setEnabled(False)

        self.btn_save_img = QPushButton(Config.get("save_res"))
        self.btn_save_img.setProperty("class", "SecondaryButton")
        self.btn_save_img.clicked.connect(self.save_result)
        
        self.chk_auto_save_img = QCheckBox(Config.get("auto_save"))
        
        controls.addStretch()
        controls.addWidget(self.btn_open_img)
        controls.addWidget(self.btn_open_folder)
        controls.addWidget(self.btn_prev_img)
        controls.addWidget(self.btn_next_img)
        controls.addWidget(self.btn_save_img)
        controls.addWidget(self.chk_auto_save_img)
        controls.addStretch()
        
        layout.addWidget(self.image_label)
        layout.addLayout(controls)
        return tab

    def create_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.video_file_label = ZoomableLabel()
        self.video_file_label.fileDropped.connect(self.process_dropped_file)
        
        self.video_progress = QProgressBar()
        self.video_progress.setValue(0)
        
        controls = QHBoxLayout()
        
        # Tracker (New)
        self.video_tracker_combo = QComboBox()
        self.video_tracker_combo.addItems(["None", "bytetrack.yaml", "botsort.yaml"])
        controls.addWidget(QLabel(Config.get("tracker")))
        controls.addWidget(self.video_tracker_combo)
        
        self.btn_open_video = QPushButton(Config.get("open_video"))
        self.btn_open_video.setProperty("class", "ActionButton")
        self.btn_open_video.clicked.connect(self.open_video)
        
        # Changed: Use Process Video toggle button
        self.btn_process_video = QPushButton(Config.get("process_video"))
        self.btn_process_video.setProperty("class", "ActionButton")
        self.btn_process_video.clicked.connect(self.toggle_video_process)
        self.btn_process_video.setEnabled(False) # Disabled until loaded
        
        self.chk_save_video_file = QCheckBox(Config.get("save_res"))
        self.chk_save_video_file.stateChanged.connect(self.toggle_video_file_save)
        
        controls.addStretch()
        controls.addWidget(self.btn_open_video)
        controls.addWidget(self.chk_save_video_file)
        controls.addWidget(self.btn_process_video)
        controls.addStretch()
        
        layout.addWidget(self.video_file_label)
        layout.addWidget(self.video_progress)
        layout.addLayout(controls)
        return tab

    def create_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form
        form_layout = QVBoxLayout()

        # Model Path (New)
        model_layout = QHBoxLayout()
        self.lbl_train_model = QLabel(Config.get("model_path") if Config.get("model_path") else "Model:")
        model_layout.addWidget(self.lbl_train_model)
        self.train_model_edit = QLabel(self.current_model)
        self.train_model_edit.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        btn_browse_model = QPushButton("...")
        btn_browse_model.clicked.connect(self.browse_train_model)
        model_layout.addWidget(self.train_model_edit)
        model_layout.addWidget(btn_browse_model)
        form_layout.addLayout(model_layout)
        
        # Data Path
        data_layout = QHBoxLayout()
        self.lbl_data = QLabel(Config.get("data_path"))
        data_layout.addWidget(self.lbl_data)
        self.train_data_edit = QLabel("coco8.yaml") # Default
        btn_browse = QPushButton("...")
        btn_browse.clicked.connect(self.browse_data)
        data_layout.addWidget(self.train_data_edit)
        data_layout.addWidget(btn_browse)
        form_layout.addLayout(data_layout)
        
        # Hyperparams
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        self.lbl_epochs = QLabel(Config.get("epochs"))
        form_layout.addWidget(self.lbl_epochs)
        form_layout.addWidget(self.spin_epochs)
        
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(16)
        self.lbl_batch = QLabel(Config.get("batch"))
        form_layout.addWidget(self.lbl_batch)
        form_layout.addWidget(self.spin_batch)
        
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 1280)
        self.spin_imgsz.setValue(640)
        self.lbl_imgsz = QLabel(Config.get("imgsz"))
        form_layout.addWidget(self.lbl_imgsz)
        form_layout.addWidget(self.spin_imgsz)
        
        # Resume Checkbox
        self.chk_resume = QCheckBox("Resume Training (from last.pt)")
        form_layout.addWidget(self.chk_resume)
        
        # Training Stats
        stats_group = QGroupBox("Training Status")
        stats_layout = QVBoxLayout()
        self.lbl_train_time = QLabel("Duration: 00:00:00")
        self.lbl_train_speed = QLabel("Speed: -")
        self.lbl_train_eta = QLabel("ETA: -")
        self.lbl_train_end = QLabel("Est. Finish: -")
        stats_layout.addWidget(self.lbl_train_time)
        stats_layout.addWidget(self.lbl_train_speed)
        stats_layout.addWidget(self.lbl_train_eta)
        stats_layout.addWidget(self.lbl_train_end)
        stats_group.setLayout(stats_layout)
        form_layout.addWidget(stats_group)
        
        # Start Button
        self.btn_train = QPushButton(Config.get("start_train"))
        self.btn_train.setProperty("class", "ActionButton")
        self.btn_train.clicked.connect(self.start_training)
        form_layout.addWidget(self.btn_train)
        
        self.btn_stop_train = QPushButton(Config.get("stop_train"))
        self.btn_stop_train.setProperty("class", "SecondaryButton")
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)
        form_layout.addWidget(self.btn_stop_train)
        
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setFormat("Epoch %v/%m")
        form_layout.addWidget(self.train_progress)

        layout.addLayout(form_layout)
        layout.addStretch()
        
        return tab

    def create_val_tab(self):
        tab = QWidget()
        outer = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form_layout = QVBoxLayout(content)

        # Model Path
        model_layout = QHBoxLayout()
        self.lbl_val_model = QLabel(Config.get("model_path"))
        model_layout.addWidget(self.lbl_val_model)
        self.val_model_edit = QLabel(self.current_model)
        self.val_model_edit.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        btn_browse_model = QPushButton("...")
        btn_browse_model.clicked.connect(self.browse_val_model)
        model_layout.addWidget(self.val_model_edit)
        model_layout.addWidget(btn_browse_model)
        form_layout.addLayout(model_layout)

        # Data Path
        data_layout = QHBoxLayout()
        self.lbl_val_data = QLabel(Config.get("data_path"))
        data_layout.addWidget(self.lbl_val_data)
        self.val_data_edit = QLabel("coco8.yaml")
        btn_browse_data = QPushButton("...")
        btn_browse_data.clicked.connect(self.browse_val_data)
        data_layout.addWidget(self.val_data_edit)
        data_layout.addWidget(btn_browse_data)
        form_layout.addLayout(data_layout)

        # Params
        self.spin_val_batch = QSpinBox()
        self.spin_val_batch.setRange(1, 512)
        self.spin_val_batch.setValue(16)
        form_layout.addWidget(QLabel(Config.get("batch")))
        form_layout.addWidget(self.spin_val_batch)

        self.spin_val_imgsz = QSpinBox()
        self.spin_val_imgsz.setRange(32, 1280)
        self.spin_val_imgsz.setValue(640)
        form_layout.addWidget(QLabel(Config.get("imgsz")))
        form_layout.addWidget(self.spin_val_imgsz)

        # Start
        self.btn_val = QPushButton(Config.get("start_val"))
        self.btn_val.setProperty("class", "ActionButton")
        self.btn_val.clicked.connect(self.start_validation)
        form_layout.addWidget(self.btn_val)

        # ---- 验证结果展示 ----
        self.lbl_val_results = QLabel(Config.get("val_results"))
        self.lbl_val_results.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 12px;")
        form_layout.addWidget(self.lbl_val_results)

        self.val_results_table = QTableWidget()
        self.val_results_table.setColumnCount(3)
        self.val_results_table.setHorizontalHeaderLabels(["指标", "数值", "评价"])
        self.val_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        form_layout.addWidget(self.val_results_table)

        self.lbl_val_perclass = QLabel(Config.get("per_class"))
        self.lbl_val_perclass.setStyleSheet("font-weight: bold; margin-top: 8px;")
        form_layout.addWidget(self.lbl_val_perclass)

        self.val_perclass_table = QTableWidget()
        self.val_perclass_table.setColumnCount(7)
        self.val_perclass_table.setHorizontalHeaderLabels(
            ["类别", "图像数", "实例数", "P", "R", "mAP50", "mAP50-95"]
        )
        self.val_perclass_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        form_layout.addWidget(self.val_perclass_table)

        self.lbl_val_eval = QLabel(Config.get("evaluation"))
        self.lbl_val_eval.setStyleSheet("font-weight: bold; margin-top: 8px;")
        form_layout.addWidget(self.lbl_val_eval)

        self.val_eval_text = QTextEdit()
        self.val_eval_text.setReadOnly(True)
        self.val_eval_text.setMinimumHeight(170)
        form_layout.addWidget(self.val_eval_text)

        self.btn_export_val = QPushButton(Config.get("export_report"))
        self.btn_export_val.setProperty("class", "ActionButton")
        self.btn_export_val.setEnabled(False)
        self.btn_export_val.clicked.connect(self.export_val_report)
        form_layout.addWidget(self.btn_export_val)

        form_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return tab

    def create_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QVBoxLayout()
        
        # Model Path
        model_layout = QHBoxLayout()
        self.lbl_export_model = QLabel(Config.get("model_path"))
        model_layout.addWidget(self.lbl_export_model)
        self.export_model_edit = QLabel(self.current_model)
        self.export_model_edit.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        btn_browse_model = QPushButton("...")
        btn_browse_model.clicked.connect(self.browse_export_model)
        model_layout.addWidget(self.export_model_edit)
        model_layout.addWidget(btn_browse_model)
        form_layout.addLayout(model_layout)
        
        # Format Selector
        self.combo_format = QComboBox()
        # Friendly names mapped to format strings
        self.export_formats = {
            "ONNX (.onnx) - CPU/GPU 通用，兼容性好，CPU 速度提升高达 3 倍": "onnx",
            "TensorRT (.engine) - NVIDIA GPU 专用，GPU 速度提升高达 5 倍": "engine",
            "OpenVINO (.xml) - Intel CPU 专用，CPU 速度提升高达 3 倍": "openvino",
            "TorchScript (.torchscript) - PyTorch 原生导出，适合移动端": "torchscript",
            "CoreML (.mlpackage) - macOS/iOS 专用": "coreml",
            "TFLite (.tflite) - Android/Edge 专用": "tflite",
            "TensorFlow SavedModel - TensorFlow 格式": "saved_model",
            "TensorFlow GraphDef (.pb) - TensorFlow 冻结图": "pb",
            "NCNN - 腾讯移动端框架": "ncnn",
            "PaddlePaddle - 百度飞桨框架": "paddle"
        }
        self.combo_format.addItems(self.export_formats.keys())
        form_layout.addWidget(QLabel(Config.get("export_format")))
        form_layout.addWidget(self.combo_format)
        
        # Export Options
        opts_layout = QVBoxLayout()
        
        # Image Size
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(QLabel(f"{Config.get('imgsz')} (导出模型的输入尺寸):"))
        self.spin_export_imgsz = QSpinBox()
        self.spin_export_imgsz.setRange(32, 1280)
        self.spin_export_imgsz.setValue(640)
        imgsz_layout.addWidget(self.spin_export_imgsz)
        opts_layout.addLayout(imgsz_layout)
        
        # Checkboxes
        self.chk_half = QCheckBox("FP16 (Half) - 半精度导出，减小体积并加速 (需 GPU 支持)")
        self.chk_int8 = QCheckBox("INT8 - 8位量化，极致加速和压缩 (需校准数据)")
        self.chk_dynamic = QCheckBox("Dynamic Axes - 动态输入尺寸 (仅 ONNX/TensorRT)")
        self.chk_simplify = QCheckBox("Simplify - 简化模型结构，提升兼容性 (仅 ONNX)")
        
        opts_layout.addWidget(self.chk_half)
        opts_layout.addWidget(self.chk_int8)
        opts_layout.addWidget(self.chk_dynamic)
        opts_layout.addWidget(self.chk_simplify)

        # 根据所选导出格式联动启用/禁用选项（防止导出失败）
        self.combo_format.currentIndexChanged.connect(self._on_export_format_changed)
        self._on_export_format_changed()  # 初始化一次当前格式的状态
        
        form_layout.addLayout(opts_layout)
        
        self.btn_export = QPushButton(Config.get("export_model"))
        self.btn_export.setProperty("class", "ActionButton")
        self.btn_export.clicked.connect(self.export_model)
        form_layout.addWidget(self.btn_export)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab

    def create_benchmark_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QVBoxLayout()
        
        # Model Path
        model_layout = QHBoxLayout()
        self.lbl_bench_model = QLabel(Config.get("model_path"))
        model_layout.addWidget(self.lbl_bench_model)
        self.bench_model_edit = QLabel(self.current_model)
        self.bench_model_edit.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        btn_browse_model = QPushButton("...")
        btn_browse_model.clicked.connect(self.browse_bench_model)
        model_layout.addWidget(self.bench_model_edit)
        model_layout.addWidget(btn_browse_model)
        form_layout.addLayout(model_layout)
        
        # Data Path
        data_layout = QHBoxLayout()
        self.lbl_bench_data = QLabel(Config.get("data_path"))
        data_layout.addWidget(self.lbl_bench_data)
        self.bench_data_edit = QLabel("coco8.yaml")
        btn_browse_data = QPushButton("...")
        btn_browse_data.clicked.connect(self.browse_bench_data)
        data_layout.addWidget(self.bench_data_edit)
        data_layout.addWidget(btn_browse_data)
        form_layout.addLayout(data_layout)
        
        self.spin_bench_imgsz = QSpinBox()
        self.spin_bench_imgsz.setRange(32, 1280)
        self.spin_bench_imgsz.setValue(640)
        form_layout.addWidget(QLabel(Config.get("imgsz")))
        form_layout.addWidget(self.spin_bench_imgsz)
        
        self.btn_bench = QPushButton(Config.get("start_benchmark"))
        self.btn_bench.setProperty("class", "ActionButton")
        self.btn_bench.clicked.connect(self.start_benchmark)
        form_layout.addWidget(self.btn_bench)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab

    def log(self, message):
        self.console.append(f">> {message}")
        print(message)

    def switch_tab(self, index):
        self.tabs.setCurrentIndex(index)

    def refresh_models(self):
        # Scan weights folder and current directory (support all exported model formats)
        exts = ("*.pt", "*.onnx", "*.engine", "*.tflite", "*.torchscript",
                "*.mlpackage", "*.xml", "*.pb", "*.param")
        models = []
        for ext in exts:
            models += glob.glob(os.path.join("weights", ext)) + glob.glob(ext)
        # Remove duplicates and get basenames
        model_names = sorted(list(set([os.path.basename(m) for m in models])))
        if not model_names:
            model_names = ["yolo26n.pt"] # Default fallback
            
        current_combo = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(model_names)
        
        # Priority:
        # 1. Previously selected item in combobox (if refreshing manually)
        # 2. self.current_model (on startup)
        # 3. First item in list
        
        target_model = None
        if current_combo and current_combo in model_names:
            target_model = current_combo
        elif os.path.basename(self.current_model) in model_names:
            target_model = os.path.basename(self.current_model)
        elif model_names:
            target_model = model_names[0]
            
        if target_model:
            self.model_combo.setCurrentText(target_model)
            # Ensure self.current_model matches what is displayed if we just loaded
            # trigger change_model to ensure consistency
            if self.current_model != target_model and os.path.basename(self.current_model) != target_model:
                 self.change_model(target_model)
            
        # Visual feedback
        self.refresh_btn.setText("✔")
        QTimer.singleShot(1000, lambda: self.refresh_btn.setText(Config.get("refresh_models")))

    def change_model(self, model_name):
        if not model_name: return
        
        # Check if it's in weights/ or root
        if os.path.exists(os.path.join("weights", model_name)):
            self.current_model = os.path.join("weights", model_name)
        else:
            self.current_model = model_name
            
        self.log(f"Model changed to {self.current_model}")
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.toggle_camera() # Restart with new model if running

    def change_device(self):
        if self.radio_cpu.isChecked():
            self.current_device = 'cpu'
        else:
            # Verify CUDA availability before committing to GPU to avoid hard crashes
            try:
                import torch
                _cuda_ok = torch.cuda.is_available()
            except Exception:
                _cuda_ok = False
            if _cuda_ok:
                self.current_device = '0' # Default GPU 0
            else:
                self.current_device = 'cpu'
                self.log("警告：当前环境 CUDA 不可用，已自动回退到 CPU 推理。")
            
        self.log(f"Inference device changed to: {self.current_device}")
        
        # Update running threads
        if self.thread:
            self.thread.update_device(self.current_device)
        if self.image_worker:
            self.image_worker.device = self.current_device

    def update_params(self):
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        
        self.lbl_conf.setText(f"{Config.get('conf')}: {conf:.2f}")
        self.lbl_iou.setText(f"{Config.get('iou')}: {iou:.2f}")

        if self.thread:
            self.thread.update_params(conf, iou)

    def update_ui_text(self):
        # Update all UI texts based on current Config.LANG
        self.setWindowTitle(Config.get("title"))
        self.logo_label.setText(f"🤖 {Config.get('title')}")
        self.lang_btn.setText(Config.get("lang_en") if Config.LANG == "CN" else Config.get("lang_cn"))
        self.theme_btn.setText(Config.get("theme_light") if Theme.CURRENT_THEME == "Dark" else Config.get("theme_dark"))
        self.dock.setWindowTitle(Config.get("control_panel"))
        
        # Nav Buttons
        # nav_keys logic is now replaced by self.nav_items iteration
        for i, btn in enumerate(self.nav_buttons):
            if i < len(self.nav_items):
                icon = self.nav_items[i][0]
                key = self.nav_items[i][1]
                name = Config.get(key)
                btn.setText(name)
                # Re-generate icon in case theme color changed (though pixmap is static, ideally we regen)
                # For now let's keep icon static or regenerate if we want theme awareness
                btn.setIcon(QIcon(emoji_to_pixmap(icon, 48)))
            
        # Group Boxes
        self.model_group.setTitle(Config.get("model_sel"))
        self.param_group.setTitle(Config.get("params"))
        self.result_group.setTitle(Config.get("results"))
        self.perf_group.setTitle(Config.get("perf_mon"))
        self.device_group.setTitle(Config.get("device"))
        
        # Labels & Buttons
        self.refresh_btn.setText(Config.get("refresh_models"))
        self.update_params() # Update slider labels with localized text and values
        
        self.lbl_cam_idx.setText(Config.get("cam_idx"))
        if self.thread and self.thread.isRunning():
            self.btn_start.setText(Config.get("stop_cam"))
            self.status_label.setText(Config.get("status_run"))
        else:
            self.btn_start.setText(Config.get("start_cam"))
            self.status_label.setText(Config.get("status_idle"))
            
        self.chk_auto_save_video.setText(Config.get("auto_save"))
        self.chk_auto_save_img.setText(Config.get("auto_save"))
        
        self.image_label.setText(Config.get("drop_hint"))
        self.video_file_label.setText(Config.get("drop_hint"))
        self.btn_open_img.setText(Config.get("open_img"))
        self.btn_open_folder.setText(Config.get("open_folder"))
        self.btn_prev_img.setText(Config.get("prev_image"))
        self.btn_next_img.setText(Config.get("next_image"))
        self.btn_save_img.setText(Config.get("save_res"))
        self.btn_open_video.setText(Config.get("open_video"))
        self.btn_stop_video.setText(Config.get("stop_process"))
        
        self.lbl_data.setText(Config.get("data_path"))
        self.lbl_epochs.setText(Config.get("epochs"))
        self.lbl_batch.setText(Config.get("batch"))
        self.lbl_imgsz.setText(Config.get("imgsz"))
        self.btn_train.setText(Config.get("start_train"))
        self.btn_stop_train.setText(Config.get("stop_train"))
        
        # Val
        self.lbl_val_model.setText(Config.get("model_path"))
        self.lbl_val_data.setText(Config.get("data_path"))
        self.btn_val.setText(Config.get("start_val"))
        
        # Export
        self.lbl_export_model.setText(Config.get("model_path"))
        self.btn_export.setText(Config.get("export_model"))
        
        # Benchmark
        self.lbl_bench_model.setText(Config.get("model_path"))
        self.lbl_bench_data.setText(Config.get("data_path"))
        self.btn_bench.setText(Config.get("start_benchmark"))

    def toggle_lang(self):
        Config.LANG = "CN" if Config.LANG == "EN" else "EN"
        self.update_ui_text()
        self.log("Language switched.")

    def toggle_theme(self):
        Theme.CURRENT_THEME = "Light" if Theme.CURRENT_THEME == "Dark" else "Dark"
        self.apply_theme()
        self.theme_btn.setText(Config.get("theme_light") if Theme.CURRENT_THEME == "Dark" else Config.get("theme_dark"))

    def change_rotation(self, text):
        if self.thread:
            self.thread.set_rotation(int(text))

    def update_performance(self):
        pass
        
    def start_perf_monitoring(self):
        self.lbl_cpu.setText("CPU: -")
        self.lbl_ram.setText("RAM: -")
        self.lbl_gpu.setText("GPU: -")
        
    def _update_perf_stats(self):
        pass

    # Real-time Logic
    def toggle_camera(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.btn_start.setText(Config.get("start_cam"))
            self.status_label.setText(Config.get("status_idle"))
            self.video_label.setText(Config.get("cam_off"))
        else:
            source_text = self.cam_combo.currentText()
            if source_text == "Screen":
                source = "screen"
            elif source_text == "Hikvision":
                source = "hikvision"
            else:
                # Parse index from "0: Camera Name"
                try:
                    source = int(source_text.split(':')[0])
                except ValueError:
                    source = 0 # Fallback
                
            tracker = self.tracker_combo.currentText()
            if tracker == "None": tracker = None
            
            self.thread = VideoThread(self.current_model, source=source, device=self.current_device, tracker=tracker)
            self.thread.change_pixmap_signal.connect(self.update_video_image)
            self.thread.stats_signal.connect(self.update_stats)
            self.thread.error_signal.connect(self.handle_worker_error) # Connect error signal
            self.thread.set_save(self.chk_auto_save_video.isChecked())
            self.thread.set_rotation(int(self.rot_combo.currentText())) # Apply initial rotation
            self.thread.start()
            self.btn_start.setText(Config.get("stop_cam"))
            self.status_label.setText(Config.get("status_run"))
            self.log(f"Source {source} started.")
            self.update_params()
    
    def handle_worker_error(self, err_msg):
        self.log(f"ERROR: {err_msg}")
        self.status_label.setText("Error")
        # Optionally show a message box
        # QMessageBox.critical(self, "Error", err_msg)

    def toggle_video_save(self, state):
        if self.thread:
            self.thread.set_save(state == Qt.CheckState.Checked.value)

    @pyqtSlot(QImage)
    def update_video_image(self, qt_img):
        # Use setPixmap directly on ZoomableLabel, it handles display logic
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    @pyqtSlot(dict)
    def update_stats(self, stats):
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        # Update list
        self.result_list.clear()
        self.result_list.addItem(f"Total Objects: {stats['objects']}")
        self.result_list.addItem(f"Inference: {stats['inference_ms']:.1f}ms")
        
        # Add detailed object counts
        if 'details' in stats:
            for name, count in stats['details'].items():
                self.result_list.addItem(f"{name}: {count}")

    @pyqtSlot(QImage, dict)
    def update_image_result(self, qt_img, stats):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        
        # Display stats
        self.result_list.clear()
        self.fps_label.setText(f"Inference: {stats['inference_ms']:.1f}ms")
        
        self.result_list.addItem(f"Total Objects: {stats['objects']}")
        for name, count in stats['details'].items():
            self.result_list.addItem(f"{name}: {count}")

    @pyqtSlot(QImage)
    def update_video_file_image(self, qt_img):
        self.video_file_label.setPixmap(QPixmap.fromImage(qt_img))

    # Image Logic
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, Config.get("open_img"), "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_list = [file_name]
            self.current_image_index = 0
            self.process_dropped_file(file_name)
            self.btn_prev_img.setEnabled(False)
            self.btn_next_img.setEnabled(False)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, Config.get("open_folder"))
        if folder_path:
            # Get all image files
            exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            self.image_list = []
            for ext in exts:
                self.image_list.extend(glob.glob(os.path.join(folder_path, ext)))
            
            self.image_list.sort()
            
            if self.image_list:
                self.current_image_index = 0
                self.log(f"Loaded {len(self.image_list)} images from {folder_path}")
                self.process_dropped_file(self.image_list[0])
                self.btn_prev_img.setEnabled(len(self.image_list) > 1)
                self.btn_next_img.setEnabled(len(self.image_list) > 1)
            else:
                self.log("No images found in folder.")

    def prev_image(self):
        if not self.image_list: return
        
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.image_list) - 1 # Loop to end
            
        file_name = self.image_list[self.current_image_index]
        self.log(f"Processing prev image ({self.current_image_index + 1}/{len(self.image_list)}): {os.path.basename(file_name)}")
        self.process_dropped_file(file_name)

    def next_image(self):
        if not self.image_list: return
        
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_list):
            self.current_image_index = 0 # Loop back to start
            
        file_name = self.image_list[self.current_image_index]
        self.log(f"Processing next image ({self.current_image_index + 1}/{len(self.image_list)}): {os.path.basename(file_name)}")
        self.process_dropped_file(file_name)

    def open_file(self):
        # Generic open handler for shortcut
        idx = self.tabs.currentIndex()
        if idx == 1: # Image
            self.open_image()
        elif idx == 2: # Video
            self.open_video()

    def process_dropped_file(self, file_name):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
             self.tabs.setCurrentIndex(2) # Switch to Video Tab
             self.nav_buttons[2].setChecked(True)
             self.start_video_process(file_name)
        else:
            self.tabs.setCurrentIndex(1) # Switch to Image Tab
            self.nav_buttons[1].setChecked(True)
            self.log(f"Processing image: {file_name}")
            self.image_worker = ImageWorker(self.current_model, file_name, self.chk_auto_save_img.isChecked(), device=self.current_device)
            self.image_worker.result_signal.connect(self.update_static_image)
            self.image_worker.error_signal.connect(self.handle_worker_error) # Connect error signal
            self.image_worker.conf = self.conf_slider.value() / 100.0
            self.image_worker.iou = self.iou_slider.value() / 100.0
            self.image_worker.start()

    @pyqtSlot(QImage, dict)
    def update_static_image(self, qt_img, stats):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.result_list.clear()
        self.result_list.addItem(f"Total Objects: {stats['objects']}")
        self.result_list.addItem(f"Inference: {stats['inference_ms']:.1f}ms")
        
        # Add detailed object counts
        if 'details' in stats:
            for name, count in stats['details'].items():
                self.result_list.addItem(f"{name}: {count}")
        
        self.log("Image processing complete.")

    def save_result(self):
        if self.tabs.currentIndex() == 1 and self.image_worker:
            self.image_worker.save_result()
        elif self.tabs.currentIndex() == 0 and self.thread:
             self.log("Video recording is handled by Auto Save checkbox.")

    # Video File Logic
    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, Config.get("open_video"), "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_name:
            self.current_video_path = file_name
            self.btn_process_video.setEnabled(True)
            self.log(f"Selected video: {file_name}")
            # Preview first frame
            cap = cv2.VideoCapture(file_name)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_file_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_file_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            cap.release()

    def toggle_video_process(self):
        if self.video_file_worker is None or not self.video_file_worker.isRunning():
            self.start_video_process()
        else:
            self.stop_video_process()

    def start_video_process(self):
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            return

        self.log(f"Processing video: {self.current_video_path}")
        tracker = self.video_tracker_combo.currentText()
        if tracker == "None": tracker = None
        
        self.video_file_worker = VideoFileWorker(self.current_model, self.current_video_path, device=self.current_device, save_video=self.chk_save_video_file.isChecked(), tracker=tracker)
        self.video_file_worker.progress_signal.connect(self.video_progress.setValue)
        self.video_file_worker.frame_signal.connect(self.update_video_file_image)
        self.video_file_worker.stats_signal.connect(self.update_stats)
        self.video_file_worker.error_signal.connect(self.handle_worker_error)
        self.video_file_worker.finished_signal.connect(self.video_finished)
        
        self.btn_open_video.setEnabled(False)
        self.btn_process_video.setText(Config.get("stop_process"))
        self.video_file_worker.start()
        self.status_label.setText(Config.get("status_run"))

    @pyqtSlot(QImage)
    def update_video_file_image(self, qt_img):
        self.video_file_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_file_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def stop_video_process(self):
        if self.video_file_worker:
            self.video_file_worker.stop()
            self.log("Stopping video process...")
        self.status_label.setText(Config.get("status_idle"))
    
    def toggle_video_file_save(self, state):
        if self.video_file_worker:
            self.video_file_worker.set_save(state == Qt.CheckState.Checked.value)

    def video_finished(self, save_dir):
        self.btn_open_video.setEnabled(True)
        self.btn_process_video.setText(Config.get("process_video"))
        self.video_progress.setValue(100)
        self.status_label.setText(Config.get("status_idle"))
        if save_dir:
            self.log(f"Video processing complete. Saved to: {save_dir}")
        else:
            self.log("Video processing stopped or finished.")

    # Train Logic
    def browse_train_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Pretrained Model", "", "Model Files (*.pt)")
        if file_name:
            self.train_model_edit.setText(file_name)

    def browse_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data YAML", "", "YAML Files (*.yaml)")
        if file_name:
            self.train_data_edit.setText(file_name)

    def start_training(self):
        model_path = self.train_model_edit.text()
        data = self.train_data_edit.text()
        epochs = self.spin_epochs.value()
        batch = self.spin_batch.value()
        imgsz = self.spin_imgsz.value()
        resume = self.chk_resume.isChecked()
        
        # Use globally selected device for training too, unless overridden? 
        # For now let's use the CPU/GPU radio button selection or keep training specific?
        # Let's use the general selection for consistency
        device = self.current_device
        
        # Reset stats
        self.lbl_train_time.setText("Duration: 00:00:00")
        self.lbl_train_speed.setText("Speed: Calculating...")
        self.lbl_train_eta.setText("ETA: Calculating...")
        self.lbl_train_end.setText("Est. Finish: Calculating...")
        
        self.train_worker = TrainWorker(model_path, data, epochs, batch, imgsz, device, resume=resume)
        self.train_worker.log_signal.connect(self.log)
        self.train_worker.progress_signal.connect(self.update_train_progress)
        self.train_worker.finished_signal.connect(self.training_finished)
        
        self.btn_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.train_progress.setRange(0, epochs)
        self.train_progress.setValue(0)
        self.train_worker.start()

    def stop_training(self):
        if self.train_worker:
            self.train_worker.stop()
            self.log("Stopping training...")
            self.btn_stop_train.setEnabled(False) # Prevent multiple clicks

    def training_finished(self):
        self.btn_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.log("Training worker finished.")
        self.lbl_train_speed.setText("Speed: Finished")
        self.lbl_train_eta.setText("ETA: 00:00:00")

    def update_train_progress(self, stats):
        # stats is now a dict
        if isinstance(stats, dict):
            epoch = stats.get('epoch', 0)
            map50 = stats.get('map50', 0)
            elapsed = stats.get('elapsed', "")
            eta = stats.get('eta', "")
            speed = stats.get('speed', "")
            eta_ts = stats.get('eta_timestamp', "")
            
            self.train_progress.setValue(epoch)
            self.train_progress.setFormat(f"Epoch {epoch}/{stats.get('total_epochs', '?')} - mAP50: {map50:.4f}")
            
            self.lbl_train_time.setText(f"Duration: {elapsed}")
            self.lbl_train_speed.setText(f"Speed: {speed}")
            self.lbl_train_eta.setText(f"ETA: {eta}")
            self.lbl_train_end.setText(f"Est. Finish: {eta_ts}")
        else:
            # Fallback for old signal format if any
            epoch = stats
            self.train_progress.setValue(epoch)

    # Val Logic
    def browse_val_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_name:
            self.val_model_edit.setText(file_name)

    def browse_val_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data YAML", "", "YAML Files (*.yaml)")
        if file_name:
            self.val_data_edit.setText(file_name)

    def start_validation(self):
        model_path = self.val_model_edit.text()
        data = self.val_data_edit.text()
        batch = self.spin_val_batch.value()
        imgsz = self.spin_val_imgsz.value()
        device = self.current_device
        
        self.log(f"Starting validation on {model_path}...")
        self.btn_val.setEnabled(False)
        
        self.val_worker = ValWorker(model_path, data, batch, imgsz, device)
        self.val_worker.log_signal.connect(self.log)
        self.val_worker.results_signal.connect(self.on_val_results)
        self.val_worker.finished_signal.connect(lambda: self.btn_val.setEnabled(True))
        self.val_worker.start()

    def on_val_results(self, results):
        """接收 ValWorker 的结构化验证结果，填充界面并启用导出。"""
        self.last_val_results = results
        scalar = results.get("scalar", {})
        ev = results.get("evaluation", {})
        grades = ev.get("metrics", {})

        # 总体指标表
        rows = [
            ("精确率 Precision", scalar.get("precision"), "精确率 (Precision)"),
            ("召回率 Recall", scalar.get("recall"), "召回率 (Recall)"),
            ("F1 分数", scalar.get("f1"), None),
            ("mAP@0.5", scalar.get("map50"), "mAP@0.5"),
            ("mAP@0.5:0.95", scalar.get("map"), "mAP@0.5:0.95"),
            ("mAP@0.75", scalar.get("map75"), None),
            ("综合适应度 Fitness", scalar.get("fitness"), None),
        ]
        self.val_results_table.setRowCount(len(rows))
        for i, (name, val, key) in enumerate(rows):
            self.val_results_table.setItem(i, 0, QTableWidgetItem(name))
            self.val_results_table.setItem(
                i, 1, QTableWidgetItem(f"{val:.4f}" if isinstance(val, (int, float)) else "N/A")
            )
            grade = grades.get(key, {}).get("grade", "") if key else ""
            self.val_results_table.setItem(i, 2, QTableWidgetItem(grade))

        # 逐类结果表
        def _cell(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            if v is None:
                return "N/A"
            return str(v)

        pc = results.get("per_class", [])
        self.val_perclass_table.setRowCount(len(pc))
        for i, row in enumerate(pc):
            vals = [
                str(row.get("class")),
                _cell(row.get("images")),
                _cell(row.get("instances")),
                _cell(row.get("precision")),
                _cell(row.get("recall")),
                _cell(row.get("map50")),
                _cell(row.get("map")),
            ]
            for j, v in enumerate(vals):
                self.val_perclass_table.setItem(i, j, QTableWidgetItem(v))

        # 综合评价文本
        self.val_eval_text.setPlainText(self._format_evaluation(results))
        self.btn_export_val.setEnabled(True)
        self.log("验证结果已显示，可点击“导出验证报告”。")

    def _format_evaluation(self, results):
        scalar = results.get("scalar", {})
        ev = results.get("evaluation", {})
        grades = ev.get("metrics", {})
        lines = []
        lines.append(f"综合评价：{ev.get('overall_grade', '-')}（mAP@0.5:0.95 = {scalar.get('map', 0):.4f}）")
        lines.append("")
        lines.append("逐指标评价：")
        for name, g in grades.items():
            lines.append(f"  - {name}：{g.get('grade', '-')}（{g.get('value', 0):.4f}）")
        lines.append("")
        lines.append("改进建议：")
        for i, s in enumerate(ev.get("suggestions", []), 1):
            lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    def export_val_report(self):
        """将验证结果导出为 Markdown 报告或 JSON 数据。"""
        if not self.last_val_results:
            self.log("暂无验证结果可供导出，请先执行验证。")
            return
        meta = self.last_val_results.get("meta", {})
        save_dir = meta.get("save_dir") or os.getcwd()
        default_path = os.path.join(save_dir, "val_report")
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出验证报告", default_path,
            "Markdown 报告 (*.md);;JSON 数据 (*.json)",
        )
        if not file_name:
            return
        try:
            if file_name.lower().endswith(".json"):
                content = build_json(self.last_val_results)
            else:
                content = build_markdown(self.last_val_results)
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
            self.log(f"验证报告已导出：{file_name}")
        except Exception as e:
            self.log(f"导出失败：{e}")

    # Benchmark Logic
    def browse_bench_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_name:
            self.bench_model_edit.setText(file_name)
            
    def browse_bench_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data YAML", "", "YAML Files (*.yaml)")
        if file_name:
            self.bench_data_edit.setText(file_name)

    def start_benchmark(self):
        model_path = self.bench_model_edit.text()
        data = self.bench_data_edit.text()
        imgsz = self.spin_bench_imgsz.value()
        device = self.current_device
        
        self.log(f"Starting benchmark on {model_path}...")
        self.btn_bench.setEnabled(False)
        
        self.benchmark_worker = BenchmarkWorker(model_path, data, imgsz, device)
        self.benchmark_worker.log_signal.connect(self.log)
        self.benchmark_worker.finished_signal.connect(lambda: self.btn_bench.setEnabled(True))
        self.benchmark_worker.start()

    # Export Logic
    def browse_export_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if file_name:
            self.export_model_edit.setText(file_name)

    def _on_export_format_changed(self):
        # 根据所选导出格式联动启用/禁用选项，防止不支持的组合导致导出失败
        fmt = self.export_formats.get(self.combo_format.currentText(), "onnx")
        if fmt == "onnx":
            # ultralytics 不支持 ONNX 的 int8 量化，自动取消勾选并禁用
            self.chk_int8.setChecked(False)
            self.chk_int8.setEnabled(False)
            # simplify 仅支持 ONNX，其余格式禁用
            self.chk_simplify.setEnabled(True)
            # dynamic 支持 ONNX/TensorRT
            self.chk_dynamic.setEnabled(True)
        elif fmt == "engine":
            # TensorRT 支持 int8/dynamic，不支持 simplify
            self.chk_int8.setEnabled(True)
            self.chk_simplify.setChecked(False)
            self.chk_simplify.setEnabled(False)
            self.chk_dynamic.setEnabled(True)
        else:
            # 其余格式：int8 可用，dynamic/simplify 不适用
            self.chk_int8.setEnabled(True)
            self.chk_simplify.setChecked(False)
            self.chk_simplify.setEnabled(False)
            self.chk_dynamic.setChecked(False)
            self.chk_dynamic.setEnabled(False)

    def export_model(self):
        selected_text = self.combo_format.currentText()
        fmt = self.export_formats.get(selected_text, "onnx") # Default to onnx
        
        model_path = self.export_model_edit.text()
        
        # Args
        imgsz = self.spin_export_imgsz.value()
        half = self.chk_half.isChecked()
        int8 = self.chk_int8.isChecked()
        dynamic = self.chk_dynamic.isChecked()
        simplify = self.chk_simplify.isChecked()
        device = self.current_device
        
        self.log(f"Starting export of {model_path} to {fmt}...")
        self.btn_export.setEnabled(False)
        
        self.export_worker = ExportWorker(model_path, fmt, imgsz, half, int8, dynamic, simplify, device)
        self.export_worker.log_signal.connect(self.log)
        self.export_worker.finished_signal.connect(lambda: self.btn_export.setEnabled(True))
        self.export_worker.start()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
