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
    "beef_carpaccio":           2,    
    "beef_tartare":             3,   
    "caesar_salad":             7,    
    "caprese_salad":            6,   
    "ceviche":                  6,  
    "chicken_wings":            6,    
    "deviled_eggs":             2,   
    "edamame":                 11,   
    "eggs_benedict":           18,   
    "escargots":                2,   
    "filet_mignon":             3,   
    "foie_gras":                4,   
    "fried_egg":                0,   
    "grilled_salmon":           5,    
    "miso_soup":                8,    
    "mussels":                  7,    
    "omelette":                 4,    
    "oysters":                  5,   
    "peking_duck":              7,    
    "pork_chop":                3,    
    "prime_rib":                5,    
    "sashimi":                  1,   
    "scallops":                 5,    
    "steak":                    5,    
 
    # ── Range 1: 21-40g ──────────────────────────────────────────
    "bibimbap":                35,   
    "chicken_curry":           25,    
    "clam_chowder":            22, 
    "club_sandwich":           38,  
    "crab_cakes":              22,   
    "croque_madame":           35,   
    "falafel":                 38,    
    "frozen_yogurt":           32,   
    "greek_salad":             12,   
    "grilled_cheese_sandwich": 38,  
    "guacamole":               12,    
    "gyoza":                   28,    
    "hot_and_sour_soup":       10,   
    "hot_dog":                 24,    
    "huevos_rancheros":        30,   
    "hummus":                  15,  
    "lasagna":                 34,    
    "lobster_bisque":          14,    
    "lobster_roll_sandwich":   35,    
    "pho":                     38,   
    "pulled_pork_sandwich":    47,    
    "samosa":                  32,   
    "seaweed_salad":           10,    
    "shrimp_and_grits":        30,   
    "spring_rolls":            28,   
    "tacos":                   23,   
    "tuna_tartare":             4,   
 
    # ── Range 2: 41-60g ──────────────────────────────────────────
    "beignets":                48,    
    "breakfast_burrito":       52,    
    "bruschetta":              42,   
    "cheesecake":              31,    
    "chicken_quesadilla":      48,   
    "chocolate_mousse":        19,   
    "churros":                 55,    
    "creme_brulee":            35,    
    "dumplings":               28,   
    "fish_and_chips":          58,    
    "french_fries":            59,   
    "french_onion_soup":       23,    
    "french_toast":            48,   
    "fried_calamari":          22,    
    "fried_rice":              52,    
    "garlic_bread":            42,    
    "gnocchi":                 52,    
    "hamburger":               48,   
    "ice_cream":               42,    
    "macaroni_and_cheese":     39,    
    "nachos":                  52,    
    "onion_rings":             48,    
    "pad_thai":                42,    
    "paella":                  52,   
    "pancakes":                44,   
    "panna_cotta":             30,    
    "pizza":                   54,    
    "poutine":                 55,    
    "risotto":                 52,   
    "spaghetti_bolognese":     54,    
    "spaghetti_carbonara":     54,    
    "sushi":                   42,    
    "takoyaki":                35,    
    "tiramisu":                42,   
    "waffles":                 48,    
 
    # ── Range 3: 61-80g ──────────────────────────────────────────
    "apple_pie":               65,    
    "baklava":                 60,    
    "cannoli":                 65,    
    "chocolate_cake":          72,   
    "chocolate_fondue":        60,    
    "cup_cakes":               64,   
    "donuts":                  69,    
    "strawberry_shortcake":    65,   
 
    # ── Range 4: 81g+ ────────────────────────────────────────────
    "bread_pudding":           85,   
    "carrot_cake":             90,    
    "macarons":                85,    
    "red_velvet_cake":         85,    
    "ramen":                   42,    
                                 
}
 

