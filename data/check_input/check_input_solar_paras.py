#!/usr/bin/env python3
"""简化版：仅统计csv文件缺失值并保存报告"""

import pandas as pd  # noqa
from pathlib import Path
from datetime import datetime


def main():
    base_path = Path("/home/phil/Files/lstmPaper/data/outputs/cycle_analysis")
    output_dir = Path("/home/phil/Files/lstmPaper/data/check_input")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in base_path.glob("*.csv"):
        df = pd.read_csv(file_path)
        missing_stats = df.isnull().sum()
        report_path = output_dir / f"{file_path.stem}_missing_report.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Missing Data Report for {file_path.name}\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write("Column,MissingCount,MissingPercentage\n")
            for col, count in missing_stats.items():
                f.write(f"{col},{count},{count/len(df)*100:.2f}%\n")
        print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()