"""统一评估所有模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation.unified_evaluator import UnifiedEvaluator, ModelType


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models using unified interface.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model specifications in format 'type:name:path', e.g., 'faster_rcnn:FRCNN:experiments/faster_rcnn/best.pth'.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/unified_eval",
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    
    args = parser.parse_args()
    
    # 解析模型规格
    models_to_evaluate = []
    for model_spec in args.models:
        parts = model_spec.split(":")
        if len(parts) == 3:
            model_type, model_name, model_path = parts
            models_to_evaluate.append((model_type, model_name, Path(model_path)))
        else:
            print(f"Warning: Invalid model specification '{model_spec}'. Expected format: 'type:name:path'")
            continue
    
    if not models_to_evaluate:
        print("Error: No valid models to evaluate.")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估所有模型
    all_results = {}
    
    for model_type, model_name, model_path in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} ({model_type})...")
        print(f"{'='*60}")
        
        try:
            evaluator = UnifiedEvaluator(
                model_type=model_type,  # type: ignore
                model_path=model_path,
                data_root=Path(args.data_root),
                split=args.split,
                device=args.device,
                batch_size=args.batch_size,
                iou_threshold=args.iou_threshold,
                conf_threshold=args.conf_threshold,
            )
            
            results = evaluator.evaluate()
            all_results[model_name] = results
            
            # 保存单个模型的结果
            result_path = output_dir / f"{model_name}_results.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            print(f"✓ Results saved to {result_path}")
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    if all_results:
        summary_path = output_dir / "all_results.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ All evaluation results saved to {summary_path}")
        print(f"\nTo generate comparison report, run:")
        print(f"  python scripts/generate_model_comparison.py \\")
        print(f"    --results {' '.join([f'{name}:{output_dir}/{name}_results.json' for name in all_results.keys()])} \\")
        print(f"    --output-dir reports/comparison")


if __name__ == "__main__":
    main()

