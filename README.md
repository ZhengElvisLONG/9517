# AgroPest-12 Insect Detection Project

本仓库面向 UNSW COMP9517 2025 T3 小组项目，实现并比较多种昆虫目标检测与分类方法。当前版本已完成数据加载、YOLOv8 快速评估、Faster R-CNN 训练框架、自动可视化与结果分析，为后续模型改进、报告撰写与演示提供统一平台。

---

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [核心脚本与模块](#核心脚本与模块)
- [运行指南](#运行指南)
- [可视化与分析](#可视化与分析)
- [项目结构](#项目结构)
- [后续工作建议](#后续工作建议)

---

## 环境配置

建议使用已创建的 Conda 环境 `elvis`。如需从零开始，可执行：

```bash
conda create -n agropest python=3.10 -y
conda activate agropest
pip install -r requirements.txt
```

关键依赖包含：

- `torch`, `torchvision`（Faster R-CNN 与深度学习框架）
- `ultralytics`（YOLOv8 训练与评估）
- `albumentations`, `opencv-python`（数据增强与图像处理）
- `pycocotools`, `scikit-learn`, `matplotlib`, `pandas`（评估与可视化）

---

## 数据准备

项目使用 Kaggle 提供的 **AgroPest-12** 数据集，目录结构须保持为：

```
data/
  ├── train/
  │   ├── images/*.jpg
  │   └── labels/*.txt   # YOLO 标注，cls xc yc w h（归一化）
  ├── valid/
  │   ├── images/*.jpg
  │   └── labels/*.txt
  └── test/
      ├── images/*.jpg
      └── labels/*.txt
```

顶层数据描述文件为 `agropest.yaml`，其中 `names` 列出了 12 种昆虫类别，确保与标注文件中的类别索引一一对应。

---

## 核心脚本与模块

| 路径 | 描述 |
| ---- | ---- |
| `src/datasets/agropest.py` | 自定义 PyTorch `Dataset`，负责读取 YOLO 标签、转换为 VOC 坐标，兼容 Albumentations。 |
| `src/models/faster_rcnn.py` | 构建 Faster R-CNN（ResNet50/101、MobileNetV3 等骨干可选），返回可训练模型。 |
| `src/training/faster_rcnn_trainer.py` | 封装 Faster R-CNN 训练流程（AMP、调度器、早停、COCO mAP & 分类指标记录）。 |
| `scripts/train_yolov8.py` | 调用 Ultralytics 的 `model.train()` 训练 YOLOv8（支持自定义参数）。 |
| `scripts/evaluate_yolov8.py` | 统一评估检测与分类性能，并输出 JSON 指标。 |
| `scripts/train_faster_rcnn.py` / `scripts/evaluate_faster_rcnn.py` | 读取配置训练 Faster R-CNN，并在验证/测试集上计算 mAP 与分类指标。 |
| `scripts/generate_visuals.py` | 自动生成数据分布、指标对比、预测类别分布、质性案例等可视化，保存至 `reports/`。 |

辅助配置：

- `configs/faster_rcnn_default.json`：Faster R-CNN 默认训练参数。
- `experiments/`：存放训练日志与评估结果。
- `reports/`：集中存放图像、表格与文本分析。

---

## 运行指南

### 1. 快速评估 YOLOv8 基线

```bash
set PYTHONPATH=%CD%
python scripts\evaluate_yolov8.py ^
  --weights yolov8n.pt ^
  --data agropest.yaml ^
  --split val ^
  --conf 0.25 ^
  --imgsz 640 ^
  --iou-threshold 0.5 ^
  --save-json experiments\yolov8\val_metrics_baseline.json
```

输出示例：

```
{
  "split": "valid",
  "detection": {"metrics/precision(B)": 0.0166, ... },
  "classification": {"precision": 0.0442, "recall": 0.10, ... }
}
```

> 说明：直接使用 COCO 预训练权重进行推理，指标极低，仅用于验证流程通畅；必须基于 AgroPest 数据重新训练。

### 2. 训练 YOLOv8（示例）

```bash
set PYTHONPATH=%CD%
python scripts\train_yolov8.py ^
  --data agropest.yaml ^
  --model yolov8n.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --project experiments\yolov8 ^
  --name agropest_finetune
```

完成后再次运行 `scripts/evaluate_yolov8.py` 收集新指标。

### 3. 训练 Faster R-CNN

```bash
set PYTHONPATH=%CD%
python scripts\train_faster_rcnn.py --config configs\faster_rcnn_default.json
```

关键参数（可在 JSON 中调整）：

- `backbone`: `"resnet50"`, `"resnet101"` 或 `"mobilenet_v3_large"`
- `epochs`, `batch_size`, `lr`, `scheduler`, `classification_iou`
- `amp`: 是否启用混合精度

评估：

```bash
python scripts\evaluate_faster_rcnn.py ^
  --checkpoint experiments\faster_rcnn\faster_rcnn_best.pth ^
  --split valid ^
  --save-json experiments\faster_rcnn\val_metrics.json
```

### 4. 生成可视化与分析

```bash
python scripts\generate_visuals.py
```

默认输出：

- 图像：`reports/figures/*.png`
- 表格：`reports/tables/*.csv`
- 文本分析：`reports/analysis/yolov8_baseline_analysis.md`

---

## 可视化与分析

自动生成的关键图表：

- `class_distribution.png`：展示 12 类昆虫在 train/valid/test 中的数量差异。
- `yolov8_detection_metrics.png` / `yolov8_classification_metrics.png`：整体指标柱状图。
- `yolov8_prediction_categories.png`：基线预测仍集中在 COCO 类别，直观体现迁移失败。
- `qualitative_*.png`：真值与预测的视觉对比。

文本汇总见 `reports/analysis/yolov8_baseline_analysis.md`，对数据分布、指标解读、失败原因及建议进行了详细说明。

---

## 项目结构

```
├── src/                     # 核心代码（数据集、模型、训练、评估工具）
├── scripts/                 # 训练、评估、可视化脚本
├── configs/                 # 训练配置
├── experiments/             # 训练输出与评估结果
├── reports/
│   ├── figures/             # 自动生成的图像
│   ├── tables/              # 指标表格
│   └── analysis/            # 文字分析
├── data/                    # AgroPest-12 数据（需手动放置）
├── agropest.yaml            # 数据集描述
├── requirements.txt         # Python 依赖
└── README.md
```

---

## 后续工作建议

1. **模型微调**：使用 YOLOv8、Faster R-CNN 在训练集上充分训练，比较不同骨干、增强策略与超参数。
2. **扩展实验**：
   - 类别不平衡处理（重采样、类别加权、Focal Loss）
   - 数据增强与对抗性扰动（模糊、噪声、亮度）
   - 可解释性（Grad-CAM、注意力可视化）
3. **指标完善**：记录 per-class mAP、PR 曲线、混淆矩阵，整合训练与推理时间对比。
4. **报告与演示**：基于 `reports/` 内容补充文字描述、图表，便于生成 IEEE 报告与 10 分钟视频脚本。

> 建议每次完成训练或评估后，运行 `scripts/generate_visuals.py` 以刷新可视化与表格，实现分析自动化。

---

如需进一步协助（代码调试、实验设计、报告撰写、演示准备等），欢迎继续提出需求。祝项目顺利推进！


