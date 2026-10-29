# Substitution pipeline (protein / carbs)

Where this lives: `BTP_2/substitution/`

Inputs it needs (edit paths in `config.py` if yours differ):
- `dataset/gastrograph/nodes_our_recipeflavor.csv` and `edges_our_recipeflavor.csv`  (from data.zip)
- `dataset/raw/RecipeDB_full/compressed_data.csv.gz`
- `EDA/results/nutrient_audit/ingredient_nutrients_final.csv`  (made by the audit notebook, Step 5 cell)
- `dataset/external/usda/...` (already there)

Install once:  `pip install scipy`

Run order (from inside BTP_2/substitution):
1. `python build_master.py`   -> outputs/ingredient_master.csv
2. `python evaluate.py`       -> outputs/review_ingredient_level.csv, review_recipe_level.csv, evaluation_summary.txt

Library files (imported, not run): retrieval.py, ranking.py, recipes.py
Setup note: download the USDA FoodData Central datasets (SR Legacy and Foundation Foods, CSV) and unzip them into dataset/external/usda/ before running the audit notebook. They are not in the repo.
