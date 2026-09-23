import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def register_chinese_fonts():
    """注册中文字体."""
    font_paths = [
        ("SimSun", "C:/Windows/Fonts/simsun.ttc", 0),
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc", 0),
        ("Microsoft YaHei Bold", "C:/Windows/Fonts/msyhbd.ttc", 0),
    ]

    registered_fonts = []
    for font_name, font_path, *args in font_paths:
        if os.path.exists(font_path):
            try:
                if args:
                    pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=args[0]))
                else:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                registered_fonts.append(font_name)
                print(f"已注册字体: {font_name}")
            except Exception as e:
                print(f"注册字体失败 {font_name}: {e}")

    if not registered_fonts:
        raise Exception("未找到可用的中文字体！")

    return registered_fonts[0]


def create_sop_training_pdf():
    output_path = "e:/yolo/ultralytics-26_2/工厂SOP遵守情况检测模型训练方案.pdf"

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    styles = getSampleStyleSheet()

    chinese_font = register_chinese_fonts()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#1a5490"),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName=chinese_font,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=10,
        spaceBefore=15,
        fontName=chinese_font,
    )

    subheading_style = ParagraphStyle(
        "CustomSubHeading",
        parent=styles["Heading3"],
        fontSize=12,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=8,
        spaceBefore=10,
        fontName=chinese_font,
    )

    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName=chinese_font
    )

    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontSize=8,
        leading=12,
        fontName="Courier",
        leftIndent=20,
        rightIndent=20,
        backColor=colors.HexColor("#f5f5f5"),
        borderPadding=5,
    )

    ParagraphStyle("TableText", parent=styles["Normal"], fontSize=9, leading=12, fontName=chinese_font)

    story = []

    story.append(Paragraph("工厂SOP遵守情况检测模型训练方案", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "基于YOLO26框架",
            ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                fontSize=12,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#7f8c8d"),
                fontName=chinese_font,
            ),
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("一、项目需求分析", heading_style))

    story.append(Paragraph("核心功能", subheading_style))
    story.append(Paragraph("1. <b>物品识别</b> - 检测操作员手中持有的各类工具、零件、防护用品等", body_style))
    story.append(Paragraph("2. <b>动作识别</b> - 判断操作员是否按照SOP规定的步骤执行操作", body_style))
    story.append(Paragraph("3. <b>异常检测</b> - 识别违反SOP的行为", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("技术挑战", subheading_style))
    story.append(Paragraph("• 不同光照条件（白天/夜间、自然光/人工光）", body_style))
    story.append(Paragraph("• 复杂背景环境（机器设备、其他工人）", body_style))
    story.append(Paragraph("• 不同操作员个体差异（身高、体型、操作习惯）", body_style))
    story.append(Paragraph("• 物品遮挡和部分可见情况", body_style))
    story.append(Paragraph("• 实时性要求", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("二、数据收集与标注规范", heading_style))

    story.append(Paragraph("2.1 数据收集策略", subheading_style))

    data_table = Table(
        [
            ["数据类型", "收集建议", "最小数量"],
            ["正常操作视频", "多角度、多时间段、多操作员录制", "50+ 小时"],
            ["异常操作视频", "故意违反SOP的各种情况", "20+ 小时"],
            ["物品图像", "单独拍摄各类工具、零件", "1000+ 张/类"],
            ["环境样本", "不同光照、背景的空场景", "500+ 张"],
        ],
        colWidths=[2.5 * inch, 2.5 * inch, 1.5 * inch],
    )

    data_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), chinese_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]
        )
    )

    story.append(data_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("2.2 数据增强策略", subheading_style))
    story.append(Paragraph("使用以下数据增强技术提高模型鲁棒性：", body_style))
    story.append(Paragraph("• Mosaic增强: 1.0", body_style))
    story.append(Paragraph("• Mixup增强: 0.15", body_style))
    story.append(Paragraph("• 色调变化 (HSV_H): 0.015", body_style))
    story.append(Paragraph("• 饱和度变化 (HSV_S): 0.7", body_style))
    story.append(Paragraph("• 明度变化 (HSV_V): 0.4", body_style))
    story.append(Paragraph("• 旋转角度: ±10°", body_style))
    story.append(Paragraph("• 平移: 0.1", body_style))
    story.append(Paragraph("• 缩放: 0.5", body_style))
    story.append(Paragraph("• 翻转: 上下/左右 0.5", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("2.3 标注规范", subheading_style))
    story.append(Paragraph("类别定义示例：", body_style))

    class_text = """
<b>物品类别：</b>
• tool_screwdriver: 螺丝刀
• tool_wrench: 扳手
• tool_hammer: 锤子
• part_bolt: 螺栓
• part_nut: 螺母
• ppe_gloves: 手套
• ppe_helmet: 安全帽

<b>动作类别：</b>
• action_pickup: 拿取
• action_install: 安装
• action_inspect: 检查
• action_adjust: 调整

<b>SOP状态：</b>
• sop_compliant: 符合SOP
• sop_violation: 违反SOP
"""
    story.append(Paragraph(class_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("标注格式（YOLO格式）：", body_style))
    story.append(Paragraph("class_id center_x center_y width height", code_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("三、模型架构选择", heading_style))

    story.append(Paragraph("3.1 推荐模型组合", subheading_style))

    model_table = Table(
        [
            ["任务", "推荐模型", "原因"],
            ["物品检测", "YOLO26m 或 YOLO26l", "平衡速度与精度，适合多类物品识别"],
            ["姿态估计", "YOLO26m-pose", "检测人体关键点，分析动作"],
            ["动作识别", "YOLO26m + 时序分析", "结合检测+跟踪+时序逻辑"],
        ],
        colWidths=[2 * inch, 2 * inch, 2.5 * inch],
    )

    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), chinese_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]
        )
    )

    story.append(model_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("3.2 模型选择建议", subheading_style))

    config_text = """
<b>实时性要求高：使用 YOLO26n</b>
• 模型: yolo26n.pt
• 输入尺寸: 640
• FPS: >30
• 适用场景: 实时监控

<b>平衡方案：使用 YOLO26m</b>
• 模型: yolo26m.pt
• 输入尺寸: 640
• FPS: >20
• 适用场景: 标准部署

<b>高精度要求：使用 YOLO26l</b>
• 模型: yolo26l.pt
• 输入尺寸: 1280
• FPS: >10
• 适用场景: 离线分析
"""
    story.append(Paragraph(config_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())

    story.append(Paragraph("四、训练流程设计", heading_style))

    story.append(Paragraph("4.1 数据集结构", subheading_style))

    dataset_text = """
datasets/sop_detection/
├── images/
│   ├── train/     # 训练集图像
│   ├── val/       # 验证集图像
│   └── test/      # 测试集图像
├── labels/
│   ├── train/     # 训练集标注
│   ├── val/       # 验证集标注
│   └── test/      # 测试集标注
└── data.yaml      # 数据集配置文件
"""
    story.append(Paragraph(dataset_text, code_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("4.2 数据集配置文件 (data.yaml)", subheading_style))

    yaml_text = """
path: datasets/sop_detection
train: images/train
val: images/val
test: images/test

nc: 15  # 类别数量
names:
  - tool_screwdriver
  - tool_wrench
  - tool_hammer
  - part_bolt
  - part_nut
  - ppe_gloves
  - ppe_helmet
  - action_pickup
  - action_install
  - action_inspect
  - action_adjust
  - sop_compliant
  - sop_violation
  - person
  - hand
"""
    story.append(Paragraph(yaml_text, code_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("4.3 训练命令", subheading_style))

    train_text = """
<b>基础训练：</b>
yolo detect train \\
  data=datasets/sop_detection/data.yaml \\
  model=yolo26m.pt \\
  epochs=200 \\
  batch=16 \\
  imgsz=640 \\
  device=0 \\
  patience=50 \\
  save=True \\
  plots=True \\
  val=True

<b>带超参数优化的训练：</b>
yolo detect train \\
  data=datasets/sop_detection/data.yaml \\
  model=yolo26m.pt \\
  epochs=300 \\
  batch=16 \\
  imgsz=640 \\
  device=0 \\
  lr0=0.01 \\
  lrf=0.01 \\
  momentum=0.937 \\
  weight_decay=0.0005 \\
  warmup_epochs=3
"""
    story.append(Paragraph(train_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("4.4 分阶段训练策略", subheading_style))

    stage_text = """
<b>阶段1：预训练微调 (50 epochs)</b>
yolo detect train data=data.yaml model=yolo26m.pt epochs=50 batch=16 imgsz=640

<b>阶段2：高分辨率训练 (100 epochs)</b>
yolo detect train data=data.yaml model=runs/detect/train/weights/last.pt \\
  epochs=100 batch=8 imgsz=1280

<b>阶段3：精细化训练 (50 epochs)</b>
yolo detect train data=data.yaml model=runs/detect/train2/weights/last.pt \\
  epochs=50 batch=16 imgsz=640 lr0=0.001
"""
    story.append(Paragraph(stage_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("五、评估指标设定", heading_style))

    story.append(Paragraph("5.1 核心评估指标", subheading_style))

    metrics_table = Table(
        [
            ["指标", "说明", "目标值"],
            ["mAP@0.5", "平均精度 (IoU=0.5)", ">0.85"],
            ["mAP@0.5:0.95", "平均精度 (IoU=0.5-0.95)", ">0.70"],
            ["Precision", "精确率", ">0.90"],
            ["Recall", "召回率", ">0.85"],
            ["F1-Score", "F1分数", ">0.87"],
            ["FPS", "推理速度", ">20 (实时)"],
        ],
        colWidths=[2 * inch, 2.5 * inch, 1.5 * inch],
    )

    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), chinese_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]
        )
    )

    story.append(metrics_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("5.2 业务指标", subheading_style))

    business_text = """
<b>SOP遵守率：</b> >95%
<b>误报率：</b> <5%
<b>漏检率：</b> <3%
<b>响应时间：</b> <2秒
<b>系统可用性：</b> >99%
"""
    story.append(Paragraph(business_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("5.3 评估命令", subheading_style))

    eval_text = """
<b>验证集评估：</b>
yolo detect val \\
  data=datasets/sop_detection/data.yaml \\
  model=runs/detect/train/weights/best.pt \\
  batch=16 \\
  imgsz=640 \\
  device=0 \\
  plots=True \\
  save_json=True

<b>测试集评估：</b>
yolo detect val \\
  data=datasets/sop_detection/data.yaml \\
  model=runs/detect/train/weights/best.pt \\
  split=test \\
  batch=16 \\
  imgsz=640 \\
  device=0
"""
    story.append(Paragraph(eval_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())

    story.append(Paragraph("六、部署与优化建议", heading_style))

    story.append(Paragraph("6.1 硬件环境要求", subheading_style))

    hw_table = Table(
        [
            ["部署场景", "CPU", "GPU", "内存", "存储"],
            ["开发/训练", "Intel i7/i9 或 AMD Ryzen 7/9", "NVIDIA RTX 3060+ (8GB+)", "32GB+", "500GB SSD"],
            ["推理服务器", "Intel Xeon 或 AMD EPYC", "NVIDIA T4/A10 (16GB+)", "16GB+", "256GB SSD"],
            ["边缘设备", "Intel Core i5+", "NVIDIA Jetson Xavier NX", "8GB+", "128GB SSD"],
            ["轻量级部署", "ARM Cortex-A76", "无GPU或NPU", "4GB+", "64GB eMMC"],
        ],
        colWidths=[1.5 * inch, 2 * inch, 2 * inch, 1 * inch, 1.5 * inch],
    )

    hw_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), chinese_font),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    story.append(hw_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("6.2 模型导出与优化", subheading_style))

    export_text = """
<b>导出为 ONNX 格式：</b>
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 simplify=True

<b>导出为 TensorRT (GPU加速)：</b>
yolo export model=runs/detect/train/weights/best.pt format=engine half=True

<b>导出为 OpenVINO (Intel CPU优化)：</b>
yolo export model=runs/detect/train/weights/best.pt format=openvino half=True

<b>导出为 TFLite (移动端/边缘设备)：</b>
yolo export model=runs/detect/train/weights/best.pt format=tflite int8=True
"""
    story.append(Paragraph(export_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("6.3 实际场景挑战处理", subheading_style))

    story.append(Paragraph("6.3.1 光照条件处理", body_style))
    story.append(Paragraph("使用自适应预处理技术：", body_style))
    story.append(Paragraph("• 自动直方图均衡化 (CLAHE)", body_style))
    story.append(Paragraph("• 自适应亮度调整", body_style))
    story.append(Paragraph("• 色彩空间转换", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("6.3.2 复杂背景处理", body_style))
    story.append(Paragraph("使用前景提取技术：", body_style))
    story.append(Paragraph("• GrabCut算法自动提取前景", body_style))
    story.append(Paragraph("• ROI (感兴趣区域) 提取", body_style))
    story.append(Paragraph("• 背景减除", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("6.3.3 个体差异处理", body_style))
    story.append(Paragraph("使用多尺度检测策略：", body_style))
    story.append(Paragraph("• 多分辨率输入 (640, 800, 1024)", body_style))
    story.append(Paragraph("• 结果融合与NMS", body_style))
    story.append(Paragraph("• 自适应缩放", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("6.4 实时推理优化", subheading_style))

    story.append(Paragraph("关键优化技术：", body_style))
    story.append(Paragraph("• 模型量化 (INT8)", body_style))
    story.append(Paragraph("• TensorRT加速", body_style))
    story.append(Paragraph("• 批处理推理", body_style))
    story.append(Paragraph("• 多线程/多进程", body_style))
    story.append(Paragraph("• GPU内存优化", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("6.5 部署架构建议", subheading_style))

    arch_text = """
<b>监控摄像头网络</b>
    ↓
<b>边缘推理节点 (Jetson/工控机)</b>    <b>中央推理服务器 (GPU服务器)</b>
• 实时检测                           • 批量分析
• 即时报警                           • 模型更新
• 本地存储                           • 数据归档
    ↓                                    ↓
        └──────────────┬───────────────┘
                       ↓
              <b>数据管理平台</b>
              • 数据存储
              • 报表生成
              • 趋势分析
"""
    story.append(Paragraph(arch_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())

    story.append(Paragraph("七、项目实施时间表", heading_style))

    timeline_table = Table(
        [
            ["阶段", "任务", "时间估算"],
            ["第1-2周", "数据收集与标注", "2周"],
            ["第3周", "数据预处理与增强", "1周"],
            ["第4-5周", "模型训练与调优", "2周"],
            ["第6周", "模型评估与优化", "1周"],
            ["第7周", "部署测试与集成", "1周"],
            ["第8周", "现场调试与验收", "1周"],
        ],
        colWidths=[1.5 * inch, 2.5 * inch, 2 * inch],
    )

    timeline_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), chinese_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]
        )
    )

    story.append(timeline_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("八、关键成功因素", heading_style))

    success_text = """
<b>1. 数据质量</b> - 高质量、多样化的标注数据是成功的关键

<b>2. 场景覆盖</b> - 确保训练数据覆盖所有实际场景

<b>3. 持续迭代</b> - 根据实际使用情况持续优化模型

<b>4. 人机协作</b> - 结合人工审核，降低误报率

<b>5. 系统稳定性</b> - 确保系统7x24小时稳定运行
"""
    story.append(Paragraph(success_text, body_style))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("九、下一步行动建议", heading_style))

    action_text = """
<b>1. 立即开始</b> - 使用yolo26_studio图形界面进行快速原型验证

<b>2. 数据准备</b> - 开始收集和标注工厂实际场景数据

<b>3. 小规模测试</b> - 先用少量数据训练一个基础模型进行验证

<b>4. 逐步扩展</b> - 根据测试结果逐步扩大数据集和模型规模
"""
    story.append(Paragraph(action_text, body_style))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("十、技术支持", heading_style))

    support_text = """
本方案基于Ultralytics YOLO26框架开发，充分利用了其高性能和易用性。

相关资源：
• Ultralytics官方文档: https://docs.ultralytics.com
• YOLO26 Studio: e:/yolo/ultralytics-26_2/yolo26_studio/main.py
• 项目主页: https://github.com/ultralytics/ultralytics

技术支持：
• GitHub Issues: https://github.com/ultralytics/ultralytics/issues
• Discord社区: https://discord.com/invite/ultralytics
• 论坛: https://community.ultralytics.com/
"""
    story.append(Paragraph(support_text, body_style))
    story.append(Spacer(1, 0.5 * inch))

    story.append(
        Paragraph(
            "文档版本: 1.0",
            ParagraphStyle(
                "Footer",
                parent=styles["Normal"],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#95a5a6"),
                fontName=chinese_font,
            ),
        )
    )
    story.append(
        Paragraph(
            "生成日期: 2026-02-11",
            ParagraphStyle(
                "Footer",
                parent=styles["Normal"],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#95a5a6"),
                fontName=chinese_font,
            ),
        )
    )

    doc.build(story)
    print(f"PDF文件已生成: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sop_training_pdf()
