"""
run.py - Dissertation Pipeline
================================
Trains and evaluates a direct ResNet-50 classifier for
carbohydrate range prediction from food images.

OVERNIGHT TRAINING - just run:
    python run.py

Stages:
  1. data     - download Food-101, build train_eval/ and final_eval/
  2. train    - train ResNet-50 with ordinal loss
  3. evaluate - evaluate with test-time augmentation (TTA x8)
  4. compare  - generate dissertation figures + summary table

Individual stage (re-run one part):
    python run.py --stage data
    python run.py --stage train
    python run.py --stage evaluate
    python run.py --stage compare
"""

import argparse
import time
from dotenv import load_dotenv

load_dotenv()


def _header(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def stage_data():
    _header("STAGE 1 / 4 - Prepare Dataset")
    from pipeline.dataset import download_food101, build_datasets
    download_food101()
    build_datasets()


def stage_train():
    _header("STAGE 2 / 4 - Train ResNet-50 (Ordinal Focal Loss)")
    from pipeline.train_direct import train
    train()


def stage_evaluate():
    _header("STAGE 3 / 4 - Evaluate Model (TTA x8)")
    from pipeline.evaluate_direct import evaluate
    evaluate()


def stage_llm(limit: int):
    _header("STAGE: GPT-5 Evaluation")
    from evaluation.llm_eval import evaluate
    evaluate(limit=limit)


def stage_compare():
    _header("STAGE 4 / 4 - Generate Dissertation Figures")
    from evaluation.compare import generate_all
    generate_all()


def main():
    parser = argparse.ArgumentParser(
        description="Carbohydrate counting dissertation - ResNet-50 pipeline",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "evaluate", "llm", "compare"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit images for LLM eval (e.g. --limit 50)")
    args = parser.parse_args()

    t_start = time.time()
    print("\n" + "="*60)
    print("  AI Carbohydrate Counting - Dissertation Pipeline")
    print("  Jake Richardson-Price")
    print("="*60)

    if args.stage == "all":
        stage_data()
        stage_train()
        stage_evaluate()
        stage_compare()
    elif args.stage == "data":
        stage_data()
    elif args.stage == "train":
        stage_train()
    elif args.stage == "evaluate":
        stage_evaluate()
    elif args.stage == "llm":
        stage_llm(args.limit)
    elif args.stage == "compare":
        stage_compare()

    elapsed = time.time() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)

    print("\n" + "="*60)
    print(f"  Done!  Total time: {hours}h {mins}m {secs}s")
    print(f"  Results -> results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
