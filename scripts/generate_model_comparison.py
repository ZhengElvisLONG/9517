"""生成模型对比报告和可视化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.visualization.model_comparison import (
    create_comparison_table,
    load_evaluation_results,
    plot_model_comparison,
    plot_radar_chart,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model comparison report.")
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Evaluation result JSON files in format 'model_name:path/to/results.json'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/comparison",
        help="Output directory for comparison reports.",
    )
    
    args = parser.parse_args()
    
    # 解析结果文件路径
    result_paths = {}
    for result_spec in args.results:
        if ":" in result_spec:
            model_name, path = result_spec.split(":", 1)
            result_paths[model_name] = Path(path)
        else:
            # 如果没有指定模型名，使用文件名
            path = Path(result_spec)
            model_name = path.stem
            result_paths[model_name] = path
    
    # 加载结果
    results = load_evaluation_results(result_paths)
    
    if not results:
        print("Error: No valid results found.")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating comparison report for {len(results)} models...")
    print(f"Models: {', '.join(results.keys())}")
    
    # 生成可视化
    print("\n1. Generating detection metrics comparison...")
    plot_model_comparison(
        results,
        output_dir / "detection_comparison.png",
        metric_type="detection",
    )
    
    print("2. Generating classification metrics comparison...")
    plot_model_comparison(
        results,
        output_dir / "classification_comparison.png",
        metric_type="classification",
    )
    
    print("3. Generating radar chart...")
    plot_radar_chart(results, output_dir / "performance_radar.png")
    
    print("4. Creating comparison table...")
    create_comparison_table(results, output_dir / "comparison_table.csv")
    
    print(f"\n✓ Comparison report generated in {output_dir}")


if __name__ == "__main__":
    main()

