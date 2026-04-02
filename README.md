# AI Carbohydrate Counting for Type 1 Diabetes
**Jake Richardson-Price Digital Systems project**

---

## What is this project?

I wanted to see if AI could help with this. Specifically, I trained a image classification model on food photos and asked it to predict which "carb range" a meal falls into. I then compared how well it did against GPT-5, which can look at a photo and answer questions about it without any specialised training.

---

## How does it work?

### The ML model (ResNet-50)

I took ResNet-50 — a well-known image classification network — and fine-tuned it on the Food-101 dataset to predict which of 5 carbohydrate ranges a meal falls into. Rather than using a standard loss function, I wrote a custom one that combines three ideas:

- **Class weighting** — some carb ranges appear far more in the dataset than others, so I weighted the loss to stop the model just guessing the most common range
- **Focal loss** — borrowed from Lin et al. (2017), this pushes the model to focus harder on the examples it keeps getting wrong rather than coasting on the easy ones
- **Ordinal penalty** — being off by 2 ranges is much worse than being off by 1 (clinically speaking it could mean the wrong insulin dose), so I added an extra penalty that scales with how far off the prediction is

At evaluation time I also use **Test-Time Augmentation** — running each image through the model 8 times with slightly different crops and lighting, then averaging the predictions. It adds a couple of percent accuracy for free.

### The LLM comparison (GPT-5)

I sent the same test images to GPT-5 with a prompt explaining the task and asked it to classify each image into one of the 5 ranges. No training, no fine-tuning — just the model reasoning from the photo.

---

## Carbohydrate ranges

I split meals into 5 ranges based on carbohydrate content:

| Range | Carbs   | Example foods                             |
|-------|---------|-------------------------------------------|
| 0     | 0–20g   | Salads, eggs, grilled meat               |
| 1     | 21–40g  | Soups, light sandwiches                  |
| 2     | 41–60g  | Pasta, pizza, rice dishes                |
| 3     | 61–80g  | Large burgers, cake, big pasta portions  |
| 4     | 81g+    | Large desserts, big portions of starchy food |

Being off by ±1 range (roughly ±20g of carbs) is considered clinically acceptable — a small insulin adjustment can compensate. Being off by ±2 or more ranges is flagged as "dangerous" in my results, as that level of error could meaningfully affect blood glucose.

---

## Project structure

```
carb-counting-t1d/
├── config.py                  ← settings, and a lookup table of carb values per food
├── run.py                     ← run everything from here
├── requirements.txt
│
├── pipeline/
│   ├── dataset.py             ← downloads Food-101, sorts images into carb range folders
│   ├── train_direct.py        ← trains the ResNet-50 model
│   └── evaluate_direct.py     ← evaluates the trained model, saves figures
│
├── evaluation/
│   ├── llm_eval.py            ← sends test images to GPT-5 and saves the results
│   └── compare.py             ← generates the comparison charts for my dissertation
│
├── data/                      ← created automatically when you run --stage data
├── models/                    ← model checkpoints saved here during training
└── results/                   ← all output figures and JSON results end up here
```



---

## Running it

Everything runs through `run.py`:

```bash
# Run the whole pipeline start to finish
python run.py

# Or run one stage at a time
python run.py --stage data       # Download Food-101 and sort images into carb ranges
python run.py --stage train      # Train the ResNet-50 model (this takes a while)
python run.py --stage evaluate   # Evaluate the trained model on the test set
python run.py --stage llm --limit 25   # Quick GPT-5 test on 25 images (~£0.50)
python run.py --stage llm        # Full GPT-5 evaluation
python run.py --stage compare    # Generate the dissertation figures
```
## Dataset

The project uses **Food-101** — 101 food categories with 1,000 images each (101,000 total). It downloads automatically when you run `--stage data`, so there's nothing to set up manually. It's about 5GB and is free to use for academic research (Bossard et al., 2014).

The carbohydrate values in `config.py` were looked up manually on the USDA FoodData Central database. Each entry has a comment showing where the value came from.


## Output files

After running the full pipeline, the `results/` folder will contain:

```
results/
├── fig1_accuracy.png          ← main comparison chart (ResNet-50 vs GPT-5)
├── fig2_f1_per_class.png      ← F1 scores broken down by carb range
├── fig3_radar.png             ← radar chart comparing multiple metrics at once
├── cm_resnet50.png            ← confusion matrix for the ResNet-50 model
├── per_range_accuracy.png     ← bar chart showing accuracy per carb range
├── summary_table.txt          ← results table I copied into my dissertation
├── resnet50_results.json
├── gpt5_results.json
└── all_results.json
```

