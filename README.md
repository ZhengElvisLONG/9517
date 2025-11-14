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
| `src/models/hog_svm.py` | HOG+SVM 基线模型（传统目标检测方法）。 |
| `src/models/efficientdet.py` | EfficientDet 模型实现（支持 D0-D7 变体）。 |
| `src/models/rt_detr.py` | RT-DETR（实时 DETR）模型实现。 |
| `src/training/faster_rcnn_trainer.py` | 封装 Faster R-CNN 训练流程（AMP、调度器、早停、COCO mAP & 分类指标记录）。 |
| `scripts/train_yolov8.py` | 调用 Ultralytics 的 `model.train()` 训练 YOLOv8（支持自定义参数）。 |
| `scripts/train_faster_rcnn.py` / `scripts/evaluate_faster_rcnn.py` | 读取配置训练 Faster R-CNN，并在验证/测试集上计算 mAP 与分类指标。 |
| `scripts/train_hog_svm.py` / `scripts/evaluate_hog_svm.py` | 训练和评估 HOG+SVM 基线模型。 |
| `scripts/train_efficientdet.py` / `scripts/evaluate_efficientdet.py` | 训练和评估 EfficientDet 模型。 |
| `scripts/train_rt_detr.py` / `scripts/evaluate_rt_detr.py` | 训练和评估 RT-DETR 模型。 |
| `scripts/evaluate_yolov8.py` | 统一评估检测与分类性能，并输出 JSON 指标。 |
| `scripts/generate_visuals.py` | 自动生成数据分布、指标对比、预测类别分布、质性案例等可视化，保存至 `reports/`。 |
| `scripts/evaluate_all_models.py` | 统一评估所有模型（使用统一接口）。 |
| `scripts/generate_model_comparison.py` | 生成模型对比报告和可视化。 |
| `scripts/generate_unified_visuals.py` | 生成统一的可视化报告，支持所有模型。 |
| `src/evaluation/unified_evaluator.py` | 统一的模型评估接口，支持所有模型类型。 |
| `src/visualization/model_comparison.py` | 模型对比可视化工具。 |
| `src/datasets/data_utils.py` | 数据集工具函数，确保所有模型兼容。 |

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

### 3.1 训练 HOG+SVM 基线模型

```bash
python scripts\train_hog_svm.py ^
  --data-root data ^
  --split train ^
  --output experiments\hog_svm\hog_svm_model.pkl ^
  --window-size 64 64
```

评估：

```bash
python scripts\evaluate_hog_svm.py ^
  --model experiments\hog_svm\hog_svm_model.pkl ^
  --data-root data ^
  --split test ^
  --save-json experiments\hog_svm\test_metrics.json
```

### 3.2 训练 EfficientDet

```bash
python scripts\train_efficientdet.py ^
  --data-root data ^
  --compound-coef 0 ^
  --epochs 50 ^
  --batch-size 8 ^
  --output-dir experiments\efficientdet
```

评估：

```bash
python scripts\evaluate_efficientdet.py ^
  --checkpoint experiments\efficientdet\efficientdet_best.pth ^
  --data-root data ^
  --split test ^
  --save-json experiments\efficientdet\test_metrics.json
```

### 3.3 训练 RT-DETR

```bash
python scripts\train_rt_detr.py ^
  --data agropest.yaml ^
  --model rtdetr-l.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --project experiments\rt_detr ^
  --name agropest_rtdetr
```

评估：

```bash
python scripts\evaluate_rt_detr.py ^
  --weights experiments\rt_detr\agropest_rtdetr\weights\best.pt ^
  --data agropest.yaml ^
  --split test ^
  --save-json experiments\rt_detr\test_metrics.json
```

### 4. 统一评估所有模型

使用统一接口评估多个模型：

```bash
python scripts\evaluate_all_models.py ^
  --models ^
    faster_rcnn:FRCNN:experiments\faster_rcnn\faster_rcnn_best.pth ^
    yolov8:YOLOv8:experiments\yolov8\best.pt ^
    hog_svm:HOG_SVM:experiments\hog_svm\hog_svm_model.pkl ^
    rtdetr:RT_DETR:experiments\rt_detr\best.pt ^
  --split test ^
  --output-dir experiments\unified_eval
```

### 5. 生成模型对比报告

```bash
python scripts\generate_model_comparison.py ^
  --results ^
    FRCNN:experiments\unified_eval\FRCNN_results.json ^
    YOLOv8:experiments\unified_eval\YOLOv8_results.json ^
    HOG_SVM:experiments\unified_eval\HOG_SVM_results.json ^
    RT_DETR:experiments\unified_eval\RT_DETR_results.json ^
  --output-dir reports\comparison
```

输出内容：
- **检测指标对比图**：`detection_comparison.png`
- **分类指标对比图**：`classification_comparison.png`
- **性能雷达图**：`performance_radar.png`
- **对比表格**：`comparison_table.csv`

### 6. 生成统一可视化报告

```bash
python scripts\generate_unified_visuals.py ^
  --results ^
    FRCNN:experiments\unified_eval\FRCNN_results.json ^
    YOLOv8:experiments\unified_eval\YOLOv8_results.json ^
  --data-root data ^
  --output-dir reports\unified
```

输出内容：
- 数据集统计图
- 模型对比可视化
- 汇总报告（Markdown）

### 7. 生成可视化与分析（传统方法）

```bash
python scripts\generate_visuals.py
```

默认输出：

- 图像：`reports/figures/*.png`
- 表格：`reports/tables/*.csv`
- 文本分析：`reports/analysis/yolov8_baseline_analysis.md`

### 5. 鲁棒性测试（Robustness Evaluation）

#### 5.1 生成失真测试集

对测试集副本应用各种失真（噪声、模糊、亮度/对比度、遮挡、JPEG 压缩）：

```bash
python robustness_tests\generate_corrupted_testset.py ^
  --test-images-dir data\test\images ^
  --output-dir test_corrupted ^
  --corruption-types noise blur brightness occlusion jpeg ^
  --levels 1 2
```

支持的失真类型：
- `noise`: 高斯噪声（强度：0.05, 0.10, 0.20）
- `blur`: 运动模糊（核大小：7, 15）
- `brightness`: 亮度/对比度调整（(0.6, 0.7), (0.4, 0.4)）
- `occlusion`: 随机遮挡（面积比例：20%, 35%）
- `jpeg`: JPEG 压缩（质量：30, 10）

输出目录结构：
```
test_corrupted/
├── noise_level1/
├── noise_level2/
├── blur_level1/
├── blur_level2/
└── ...
```

#### 5.2 运行鲁棒性评估

在失真测试集上评估模型性能：

```bash
# 首先在原始测试集上评估，获取基线结果
python scripts\evaluate_faster_rcnn.py ^
  --checkpoint experiments\faster_rcnn\faster_rcnn_best.pth ^
  --split test ^
  --save-json experiments\baseline\test_baseline.json

# 运行鲁棒性评估
python scripts\run_robustness_eval.py ^
  --corrupted-testset-dir test_corrupted ^
  --baseline-results experiments\baseline\test_baseline.json ^
  --frcnn-checkpoint experiments\faster_rcnn\faster_rcnn_best.pth ^
  --yolov8-weights experiments\yolov8\best.pt ^
  --output-dir experiments\robustness
```

#### 5.3 生成鲁棒性报告

生成可视化报告和对比分析：

```bash
python scripts\generate_robustness_report.py ^
  --results experiments\robustness\robustness_results.json ^
  --baseline experiments\baseline\test_baseline.json ^
  --output-dir reports\robustness
```

输出内容：
- **鲁棒性曲线**：`robustness_curve_<type>.png` - 每种失真的 mAP 下降曲线（两模型对比）
- **结果表格**：`robustness_table.csv` / `robustness_table.md` - 每类失真下的 mAP 和 ΔmAP
- **分析报告**：`robustness_analysis.md` - 关键发现和结论

关键分析点：
- 哪类失真对模型影响最大？（通常是强遮挡、强模糊、极低亮度）
- 两阶段（Faster R-CNN）vs 单阶段（YOLO）的鲁棒性对比
- 数据增强（Blur/ColorJitter/Mosaic）对鲁棒性的提升效果

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
│   ├── datasets/            # 数据集加载
│   │   └── data_utils.py    # 数据集工具函数（格式转换、验证等）
│   ├── models/              # 模型定义
│   ├── training/            # 训练器
│   ├── evaluation/          # 评估工具
│   │   └── unified_evaluator.py  # 统一评估接口
│   ├── robustness/          # 鲁棒性测试（失真注入）
│   └── visualization/       # 可视化工具
│       └── model_comparison.py   # 模型对比可视化
├── scripts/                 # 训练、评估、可视化脚本
│   ├── train_*.py           # 训练脚本
│   ├── evaluate_*.py        # 评估脚本
│   ├── evaluate_all_models.py      # 统一评估所有模型
│   ├── generate_model_comparison.py  # 模型对比报告
│   ├── generate_unified_visuals.py  # 统一可视化报告
│   ├── run_robustness_eval.py      # 鲁棒性评估
│   └── generate_robustness_report.py  # 鲁棒性报告生成
├── robustness_tests/        # 鲁棒性测试脚本
│   └── generate_corrupted_testset.py  # 生成失真测试集
├── configs/                 # 训练配置
├── experiments/             # 训练输出与评估结果
│   └── robustness/         # 鲁棒性评估结果
├── test_corrupted/          # 失真测试集（生成后）
├── reports/
│   ├── figures/             # 自动生成的图像
│   ├── tables/              # 指标表格
│   ├── analysis/            # 文字分析
│   └── robustness/          # 鲁棒性报告
├── data/                    # AgroPest-12 数据（需手动放置）
├── agropest.yaml            # 数据集描述
├── requirements.txt         # Python 依赖
└── README.md
```

---

## 后续工作建议

1. **模型扩展**：
   - ✅ EfficientDet 和 RT-DETR 的实现与评估
   - ✅ 基线模型 HOG+SVM 的实现
2. **模型微调**：使用 YOLOv8、Faster R-CNN 在训练集上充分训练，比较不同骨干、增强策略与超参数。
3. **鲁棒性研究**：
   - ✅ 失真测试集生成（噪声、模糊、亮度、遮挡、JPEG 压缩）
   - ✅ 鲁棒性评估框架
   - 数据增强对鲁棒性的影响（Blur/ColorJitter/Mosaic 消融实验）
   - 不平衡数据集下的鲁棒性（类别不平衡处理）
4. **扩展实验**：
   - 类别不平衡处理（重采样、类别加权、Focal Loss）
   - 可解释性（Grad-CAM、注意力可视化）
5. **指标完善**：记录 per-class mAP、PR 曲线、混淆矩阵，整合训练与推理时间对比。
6. **报告与演示**：基于 `reports/` 内容补充文字描述、图表，便于生成 IEEE 报告与 10 分钟视频脚本。

> 建议每次完成训练或评估后，运行 `scripts/generate_visuals.py` 以刷新可视化与表格，实现分析自动化。

---

如需进一步协助（代码调试、实验设计、报告撰写、演示准备等），欢迎继续提出需求。祝项目顺利推进！


