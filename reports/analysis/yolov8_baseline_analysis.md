# YOLOv8 基线结果分析

本节基于预训练 `yolov8n` 模型在 AgroPest-12 验证集上的直接推理结果，结合 `experiments/yolov8/val_metrics_baseline.json` 及自动生成的可视化，对检测与分类表现进行细致剖析。由于模型尚未针对昆虫数据进行微调，输出类别仍对应 COCO 标签，导致指标整体偏低；但这些结果为后续改进提供了重要的诊断线索。

## 数据分布洞察

- `reports/figures/class_distribution.png` 展示了训练、验证与测试三个子集的边界框数量。不同虫类之间差异明显：`Grasshoppers`、`Caterpillars` 等类别占比显著高于 `Earwigs`、`Bees` 等长尾类别。
- 不同子集的分布保持一致，说明官方划分遵循同一统计特性；训练阶段应考虑类别不平衡（如 Focal Loss、类别重采样或加权指标）。

## 检测指标解读

- `reports/figures/yolov8_detection_metrics.png` 给出了整体检测指标：
  - `Precision` ≈ 0.0166、`Recall` ≈ 0.0031，意味着预测框大量偏离昆虫目标；
  - `mAP@0.50` ≈ 0.0089，`mAP@0.50:0.95` ≈ 0.0050，基本无法捕获正确边界。
- 结合 `runs/detect/val9/confusion_matrix.png` 可见，预测主要落在 COCO 中的 `person`、`bus` 等类别，说明模型仍按照原任务输出。

## 分类指标解读

- `reports/figures/yolov8_classification_metrics.png` 中，匹配上的极少数框对应的分类指标也非常低 (`F1` ≈ 0.043，`Accuracy` ≈ 0.069)。
- `reports/figures/yolov8_prediction_categories.png` 清晰地展示了模型最常预测的类别 ID 集中在 COCO 索引 0、2、5、7 等位置，与 AgroPest 类别完全不符，进一步验证了需要重新训练分类头。

## 质性案例

- `reports/figures/qualitative_*.png` 为同一张图像的“真值 vs 预测”对比。绿色框表示真实昆虫位置与标签，橙色框为模型预测：
  - 预测框要么缺失，要么覆盖背景区域；
  - 标签名称仍保留 COCO 类别 ID，导致无法与昆虫类别对应。
- 该现象凸显了必须进行数据迁移学习，否则模型难以适应农业场景。

## 关键结论与建议

1. **需进行专门训练**：当前结果仅能作为基线参照，后续应使用训练集对 YOLOv8 及 Faster R-CNN 进行充分微调。
2. **类别映射校验**：训练脚本需确保 YAML `names` 与模型输出完全一致；对于 YOLOv8，应从头训练或载入空权重的自定义头。
3. **关注长尾类别**：建议在训练阶段引入采样平衡策略，并在评估中做 per-class 分析，确保小众虫类不被忽视。
4. **扩展评估维度**：训练完成后，可重新运行 `scripts/generate_visuals.py` 自动生成新一轮的可视化与统计，便于量化改进幅度。

综上所述，基线分析虽表现不佳，但清晰指出了后续任务的重点方向，为项目推进提供了数据支撑与可视化证据。


