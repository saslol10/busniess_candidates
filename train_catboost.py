"""python train_catboost.py file1.xlsx file2.xlsx --output-dir models"""
import argparse
import json
import sys
from catboost_training import train_and_save


def main():
    parser = argparse.ArgumentParser(description="Обучение CatBoost по экспертной разметке 0/1")
    parser.add_argument("files", nargs="+", help="Excel-файлы разметки")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--allow-unvalidated", action="store_true",
                        help="Разрешить экспериментальное обучение без отложенного теста; не для рабочих выводов")
    args = parser.parse_args()
    try:
        result = train_and_save(args.files, args.output_dir, args.allow_unvalidated, args.iterations)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Обучение не выполнено: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
