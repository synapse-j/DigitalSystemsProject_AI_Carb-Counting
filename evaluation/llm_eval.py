"""
evaluation/llm_eval.py
=======================
Evaluates GPT-5 on the same carbohydrate range classification task
as the ResNet-50 model, using food images from the final_eval test set.

Each image is sent to GPT-5 with a structured prompt asking it to
classify the food into one of 5 carbohydrate ranges. Results are
saved to results/gpt5_results.json for comparison with ResNet-50.

Usage:
    python run.py --stage llm                   # all test images
    python run.py --stage llm --limit 50        # 50 images (cheap test ~$0.50)

Requirements:
    OPENAI_API_KEY in your .env file
    pip install openai

NOTE: Check the current GPT-5 model ID at platform.openai.com/docs/models
      and update GPT5_MODEL below if needed.
"""

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR, RESULTS_DIR,
    NUM_CARB_RANGES, CARB_RANGE_LABELS,
)
from pipeline.train_direct import CarbRangeDataset
from torchvision import transforms

# -- Update this if OpenAI releases a different model ID --------------
GPT5_MODEL = "gpt-5.4"
# ---------------------------------------------------------------------

RANGE_LABELS = list(CARB_RANGE_LABELS.values())

SYSTEM_PROMPT = """You are a clinical nutrition expert helping people with
Type 1 diabetes count carbohydrates from food photographs.

Your task: look at the food image and estimate its total carbohydrate content,
then classify it into exactly one of these ranges:

  Range 0: 0–20g   (e.g. salad, clear soup, grilled fish, eggs)
  Range 1: 21–40g  (e.g. small sandwich, cup of soup with bread, sushi)
  Range 2: 41–60g  (e.g. standard pizza slice, bowl of pasta, burger)
  Range 3: 61–80g  (e.g. large bowl of rice, fish and chips, big burrito)
  Range 4: 81g+    (e.g. full plate of pasta, large dessert, full pizza)

Consider BOTH the type of food AND the portion size visible in the image.

Respond with ONLY valid JSON in this exact format:
{"range": <integer 0-4>, "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""


def encode_image(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_gpt5(client: OpenAI, img_path: str) -> dict:
    """Send one image to GPT-5 and return parsed response."""
    b64 = encode_image(img_path)
    response = client.chat.completions.create(
        model=GPT5_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",   # cheaper, sufficient for food classification
                        },
                    },
                    {
                        "type": "text",
                        "text": "What carbohydrate range does this food fall into?",
                    },
                ],
            },
        ],
        max_completion_tokens=150,
        temperature=0,   # deterministic output
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def evaluate(limit: int = None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n  ERROR: OPENAI_API_KEY not set in .env")
        print("  Add your key to .env:  OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the same test set as ResNet-50
    eval_dir = str(Path(DATASET_DIR) / "final_eval")
    test_ds   = CarbRangeDataset(eval_dir, transforms.ToTensor(), split=None)
    samples   = test_ds.samples
    if limit:
        import random
        random.seed(42)
        samples = random.sample(samples, min(limit, len(samples)))

    print(f"\n  GPT-5 Evaluation")
    print(f"  Model:  {GPT5_MODEL}")
    print(f"  Images: {len(samples):,}")
    print(f"  Note:   detail=low to minimise cost\n")

    true_labels, pred_labels, confidences = [], [], []
    errors = 0

    for i, (img_path, true_label) in enumerate(samples):
        try:
            result = query_gpt5(client, img_path)
            pred   = int(result["range"])
            conf   = float(result.get("confidence", 0.5))

            # Clamp to valid range
            pred = max(0, min(NUM_CARB_RANGES - 1, pred))

            true_labels.append(true_label)
            pred_labels.append(pred)
            confidences.append(conf)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  !! API error: {e}")
            true_labels.append(true_label)
            pred_labels.append(2)
            confidences.append(0.0)

        if (i + 1) % 25 == 0:
            acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
            print(f"  [{i+1:5d}/{len(samples):5d}]  running acc={acc:.3f}  errors={errors}")

        # Respect OpenAI rate limits
        time.sleep(0.5)

    labels = np.array(true_labels)
    preds  = np.array(pred_labels)

    acc       = float((labels == preds).mean())
    clin_acc  = float((np.abs(labels - preds) <= 1).mean())
    dangerous = float((np.abs(labels - preds) >= 2).mean())
    report    = classification_report(
        labels, preds,
        labels=list(range(NUM_CARB_RANGES)),
        target_names=RANGE_LABELS,
        output_dict=True, zero_division=0,
    )

    print(f"\n  {'-'*50}")
    print(f"  GPT-5 Results")
    print(f"  {'-'*50}")
    print(f"  Exact accuracy:           {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Clinical acc (+-1 range):  {clin_acc:.3f}  ({clin_acc*100:.1f}%)")
    print(f"  Dangerous preds (+-2+):    {dangerous:.3f}  ({dangerous*100:.1f}%)")
    print(f"  API errors:               {errors}")

    result = {
        "model":                                   f"GPT-5 ({GPT5_MODEL})",
        "images_evaluated":                        len(samples),
        "test_accuracy":                           round(acc, 4),
        "clinically_acceptable_accuracy_+-1_range": round(clin_acc, 4),
        "dangerous_predictions_+-2_ranges":         round(dangerous, 4),
        "api_errors":                              errors,
        "classification_report":                   report,
        "confusion_matrix":                        confusion_matrix(labels, preds).tolist(),
    }

    out_path = os.path.join(RESULTS_DIR, "gpt5_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  OK Results saved -> {out_path}")
