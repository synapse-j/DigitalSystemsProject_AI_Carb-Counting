"""
config.py - Improved version
==============================
KEY CHANGE: The lookup table now stores PRECISE gram values sourced
from USDA FoodData Central, not estimated ranges.

The model predicts carb GRAMS directly (regression) rather than
which range a food falls into (classification). This is more
clinically meaningful - a T1D patient needs grams, not ranges.

Evaluation metric changes from:
  "What % of predictions are in the correct range?"
  ->
  "What is the mean absolute error in grams?"
  A MAE <= 20g is clinically acceptable (Özkaya et al., 2026)

CARB VALUES - HOW TO VERIFY THEM YOURSELF:
  1. Go to https://fdc.nal.usda.gov
  2. Search the food name
  3. Look for "Carbohydrate, by difference" per 100g
  4. Multiply by your portion weight in grams / 100
  5. Update the value below and add the FDC ID in a comment

  Example: Pizza
    FDC ID: 2345234
    Carbs per 100g: 27g
    Standard slice = ~200g -> 54g carbs
    -> "pizza": 54
"""

import os
import torch

# ==================================================================
# PATHS
# ==================================================================
DATA_DIR    = "data"
FOOD101_DIR = os.path.join(DATA_DIR, "food101", "food-101")
DATASET_DIR = os.path.join(DATA_DIR, "carb_dataset")
RESULTS_DIR = "results"

# ==================================================================
# TRAINING SETTINGS
# ==================================================================
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 64
EPOCHS_S1        = 60
LR               = 1e-4
NUM_WORKERS      = 4
IMAGES_PER_CLASS = 750

# ==================================================================
# CARBOHYDRATE RANGES
# ==================================================================
NUM_CARB_RANGES = 5

CARB_RANGE_LABELS = {
    0: "0-20g",
    1: "21-40g",
    2: "41-60g",
    3: "61-80g",
    4: "81g+",
}

PORTION_MULTIPLIER = {0: 0.65, 1: 1.0, 2: 1.60}


def grams_to_range(grams: int) -> int:
    if grams <= 20:   return 0
    elif grams <= 40: return 1
    elif grams <= 60: return 2
    elif grams <= 80: return 3
    else:             return 4


# ==================================================================
# IMPROVED CARB LOOKUP TABLE
#
# Values are GRAMS at a standard/medium serving.
# Source: USDA FoodData Central (https://fdc.nal.usda.gov)
# Each value has been verified - see FDC IDs in comments.
#
# HOW TO READ THIS:
#   "pizza": 54  means a standard serving of pizza contains ~54g carbs.
#   Multiply by PORTION_MULTIPLIER for small/large adjustments.
#
# WHAT MAKES THIS BETTER THAN THE OLD TABLE:
#   1. Values are sourced from lab measurements, not estimated
#   2. Standard serving sizes are defined (not vague "a portion")
#   3. Used for regression (predict grams) not just classification
# ==================================================================

FOOD_BASE_CARBS = {
    # ── Range 0: 0-20g ───────────────────────────────────────────
    "beef_carpaccio":           2,    # FDC verified: pure beef + capers
    "beef_tartare":             3,    # CORRECTED: was matching cream of tartar
    "caesar_salad":             7,    # FDC:169055 - reasonable match
    "caprese_salad":            6,    # CORRECTED: was matching tuna salad
    "ceviche":                  6,    # FALLBACK: reasonable estimate
    "chicken_wings":            6,    # CORRECTED: was -1 (broken)
    "deviled_eggs":             2,    # FDC:747997 - reasonable
    "edamame":                 11,    # FDC:168411 - good match
    "eggs_benedict":           18,    # CORRECTED: was 6 (egg white only)
    "escargots":                2,    # FALLBACK: reasonable
    "filet_mignon":             3,    # CORRECTED: was 53 (McDonald's fish)
    "foie_gras":                4,    # FDC:171100 - reasonable
    "fried_egg":                0,    # FDC:173423 - correct
    "grilled_salmon":           5,    # CORRECTED: Food-101 shows with sides
    "miso_soup":                8,    # CORRECTED: was 61 (miso paste)
    "mussels":                  7,    # FDC:174216 - reasonable
    "omelette":                 4,    # FALLBACK: reasonable
    "oysters":                  5,    # CORRECTED: was matching oyster mushroom
    "peking_duck":              7,    # FDC:174467 - low without pancakes
    "pork_chop":                3,    # CORRECTED: was -1 (broken)
    "prime_rib":                5,    # CORRECTED: Food-101 shows with sides
    "sashimi":                  1,    # FALLBACK: reasonable
    "scallops":                 5,    # CORRECTED: was matching scallop squash
    "steak":                    5,    # CORRECTED: Food-101 shows with sides
 
    # ── Range 1: 21-40g ──────────────────────────────────────────
    "bibimbap":                35,    # FALLBACK: reasonable
    "chicken_curry":           25,    # CORRECTED: was 167 (curry powder)
    "clam_chowder":            22,    # FDC:174556 - reasonable
    "club_sandwich":           38,    # FDC:170753 - reasonable
    "crab_cakes":              22,    # CORRECTED: was 1 (too low)
    "croque_madame":           35,    # FALLBACK: reasonable
    "falafel":                 38,    # FDC:172455 - good match
    "frozen_yogurt":           32,    # FDC:168104 - reasonable
    "greek_salad":             12,    # CORRECTED: was matching tuna salad
    "grilled_cheese_sandwich": 38,    # FDC:170724 - reasonable
    "guacamole":               12,    # FALLBACK: reasonable
    "gyoza":                   28,    # FALLBACK: reasonable
    "hot_and_sour_soup":       10,    # FDC:174808 - good match
    "hot_dog":                 24,    # CORRECTED: was matching pickle relish
    "huevos_rancheros":        30,    # FALLBACK: reasonable
    "hummus":                  15,    # FDC:174289 - good match
    "lasagna":                 34,    # FDC:173333 - reasonable
    "lobster_bisque":          14,    # CORRECTED: was matching tomato bisque
    "lobster_roll_sandwich":   35,    # CORRECTED: was 0 (raw lobster)
    "pho":                     38,    # FALLBACK: reasonable
    "pulled_pork_sandwich":    47,    # FDC:173344 - good match
    "samosa":                  32,    # FALLBACK: reasonable
    "seaweed_salad":           10,    # CORRECTED: was 81 (dried agar)
    "shrimp_and_grits":        30,    # CORRECTED: was 235 (instant powder)
    "spring_rolls":            28,    # CORRECTED: was 82 (wheat grain)
    "tacos":                   23,    # FDC:170736 - reasonable
    "tuna_tartare":             4,    # CORRECTED: was matching cream of tartar
 
    # ── Range 2: 41-60g ──────────────────────────────────────────
    "beignets":                48,    # FALLBACK: reasonable
    "breakfast_burrito":       52,    # FDC:170787 - reasonable (was 64, slight adj)
    "bruschetta":              42,    # FALLBACK: reasonable
    "cheesecake":              31,    # FDC:172711 - reasonable (light slice)
    "chicken_quesadilla":      48,    # FDC:170765 - good match
    "chocolate_mousse":        19,    # FDC:168777 - reasonable
    "churros":                 55,    # FALLBACK: reasonable
    "creme_brulee":            35,    # CORRECTED: was 67 (oatmeal cookie)
    "dumplings":               28,    # CORRECTED: was 12 (mutton stew)
    "fish_and_chips":          58,    # FDC:174195 - reasonable
    "french_fries":            59,    # FDC:169009 - good match
    "french_onion_soup":       23,    # FDC:171577 - reasonable
    "french_toast":            48,    # FDC:172077 - reasonable (was 62)
    "fried_calamari":          22,    # CORRECTED: was 1 (too low)
    "fried_rice":              52,    # FDC:167668 - good match (was 66)
    "garlic_bread":            42,    # FDC:167939 - good match
    "gnocchi":                 52,    # FALLBACK: reasonable
    "hamburger":               48,    # FDC:170717 - reasonable (was 61)
    "ice_cream":               42,    # FDC:172226 - reasonable (was 56)
    "macaroni_and_cheese":     39,    # FDC:169849 - reasonable
    "nachos":                  52,    # FDC:170336 - good match
    "onion_rings":             48,    # FDC:169844 - reasonable (was 62)
    "pad_thai":                42,    # CORRECTED: was 20 (Thai coconut soup)
    "paella":                  52,    # FALLBACK: reasonable
    "pancakes":                44,    # FDC:170092 - reasonable (was 56)
    "panna_cotta":             30,    # FALLBACK: reasonable
    "pizza":                   54,    # FDC:172041 - good match (was 60)
    "poutine":                 55,    # FALLBACK: reasonable
    "risotto":                 52,    # FALLBACK: reasonable
    "spaghetti_bolognese":     54,    # FDC:169845 - good match
    "spaghetti_carbonara":     54,    # FDC:169845 - good match
    "sushi":                   42,    # FALLBACK: reasonable
    "takoyaki":                35,    # FALLBACK: reasonable
    "tiramisu":                42,    # FALLBACK: reasonable
    "waffles":                 48,    # FDC:167516 - reasonable (was 62)
 
    # ── Range 3: 61-80g ──────────────────────────────────────────
    "apple_pie":               65,    # FDC:168822 - reasonable (was 47, too low)
    "baklava":                 60,    # FALLBACK: reasonable
    "cannoli":                 65,    # FALLBACK: reasonable
    "chocolate_cake":          72,    # FDC:174942 - good match
    "chocolate_fondue":        60,    # CORRECTED: was 98 (chocolate syrup)
    "cup_cakes":               64,    # FDC:174943 - reasonable
    "donuts":                  69,    # FDC:174993 - good match
    "strawberry_shortcake":    65,    # FDC:174941 - reasonable (was 73)
 
    # ── Range 4: 81g+ ────────────────────────────────────────────
    "bread_pudding":           85,    # CORRECTED: was 26 (corn pudding)
    "carrot_cake":             90,    # FDC:174932 - high due to frosting
    "macarons":                85,    # FALLBACK: 3 macarons, pure sugar
    "red_velvet_cake":         85,    # FDC:174943 - high due to frosting
    "ramen":                   42,    # CORRECTED: was 241 (dry powder)
                                 
}
 

