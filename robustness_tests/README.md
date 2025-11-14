# 鲁棒性测试使用指南

本目录包含用于生成失真测试集和评估模型鲁棒性的脚本。

## 快速开始

### 1. 生成失真测试集

```bash
python robustness_tests/generate_corrupted_testset.py \
  --test-images-dir data/test/images \
  --output-dir test_corrupted \
  --corruption-types noise blur brightness occlusion jpeg \
  --levels 1 2
```

### 2. 运行鲁棒性评估

首先在原始测试集上获取基线结果：

```bash
# Faster R-CNN 基线
python scripts/evaluate_faster_rcnn.py \
  --checkpoint experiments/faster_rcnn/faster_rcnn_best.pth \
  --split test \
  --save-json experiments/baseline/test_baseline_frcnn.json

# YOLOv8 基线
python scripts/evaluate_yolov8.py \
  --weights experiments/yolov8/best.pt \
  --data agropest.yaml \
  --split test \
  --save-json experiments/baseline/test_baseline_yolo.json
```

然后运行鲁棒性评估：

```bash
python scripts/run_robustness_eval.py \
  --corrupted-testset-dir test_corrupted \
  --baseline-results experiments/baseline/test_baseline.json \
  --frcnn-checkpoint experiments/faster_rcnn/faster_rcnn_best.pth \
  --yolov8-weights experiments/yolov8/best.pt \
  --output-dir experiments/robustness
```

### 3. 生成报告

```bash
python scripts/generate_robustness_report.py \
  --results experiments/robustness/robustness_results.json \
  --baseline experiments/baseline/test_baseline.json \
  --output-dir reports/robustness
```

## 失真类型说明

| 类型 | 参数 | Level 1 | Level 2 |
|------|------|---------|---------|
| **noise** | 高斯噪声标准差 | 0.05 | 0.10, 0.20 |
| **blur** | 运动模糊核大小 | 7 | 15 |
| **brightness** | 亮度/对比度因子 | (0.6, 0.7) | (0.4, 0.4) |
| **occlusion** | 遮挡面积比例 | 20% | 35% |
| **jpeg** | JPEG 压缩质量 | 30 | 10 |

## 输出文件说明

### 失真测试集
- `test_corrupted/<type>_level<level>/` - 每种失真类型的图像

### 评估结果
- `experiments/robustness/robustness_results.json` - 完整评估结果

### 报告
- `reports/robustness/robustness_curve_<type>.png` - 鲁棒性曲线图
- `reports/robustness/robustness_table.csv` - 结果表格
- `reports/robustness/robustness_analysis.md` - 分析报告

## 注意事项

1. 生成失真测试集可能需要一些时间，取决于测试集大小
2. 评估过程会为每个失真类型创建临时数据集，确保有足够的磁盘空间
3. 建议先在小规模测试集上验证流程，再运行完整评估

