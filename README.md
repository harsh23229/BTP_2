# Intelligent Ingredient Substitution

An ingredient substitution system that combines nutritional analysis, recipe context, and molecular flavour similarity to identify and rank potential alternatives for food ingredients.

The system extends the existing GastroGraph architecture by integrating trusted nutritional profiles and FlavorDB molecular similarity. It supports multiple nutritional objectives while considering how ingredients relate to one another in actual recipes.

## 1. Project Overview

Replacing an ingredient is not simply about finding another ingredient with similar nutritional properties. A suitable alternative should also make sense in the context of the recipe and, where molecular information is available, have relevant flavour characteristics.

This project combines three complementary sources of information:

- **Nutritional data:** Evaluates ingredients based on protein, carbohydrates, calories, and fat.
- **GastroGraph:** Captures ingredient relationships and recipe context to retrieve contextually relevant alternatives.
- **FlavorDB:** Provides molecular profiles that help estimate flavour-related similarity between ingredients.

These signals are used to retrieve, filter, and rank potential ingredient substitutions.

## 2. Data Sources

The project uses food and ingredient datasets developed by CoSyLab, along with nutritional reference data.

| Dataset / Source | Purpose |
|---|---|
| RecipeDB | Recipe information, ingredient composition, and ingredient-level nutritional analysis |
| FlavorDB | Food–molecule associations and molecular flavour information |
| GastroGraph | Ingredient relationships and recipe-context information |
| USDA FoodData Central | Trusted nutritional reference profiles for selected ingredients |

### Nutritional Data

The substitution pipeline considers four nutritional measures:

- **Protein (g/100g)**
- **Carbohydrates (g/100g)**
- **Calories (kcal/100g)**
- **Fat (g/100g)**

Nutritional values are compared on a common per-100g basis to ensure consistent comparisons between ingredients.

The nutritional master is built from `ingredient_nutrients_final.csv`, which contains:

| Source / Tier | Records |
|---|---:|
| USDA / Trusted | 251 |
| RecipeDB / Unverified | 514 |

Only explicitly USDA-sourced, trusted records are eligible for the current substitution ranking pipeline.

RecipeDB-derived nutritional profiles are retained for comparison and auditing, but are not automatically treated as trusted ranking data.

## 3. Nutritional Data Validation

Before nutritional profiles are used, the pipeline applies validation checks.

The `build_master.py` module:

1. Accepts only records marked `source=usda` and `tier=trusted`.
2. Requires all four nutrient values to be finite and non-negative.
3. Checks that protein + carbohydrates + fat does not exceed 100.5 g per 100g.
4. Checks that calories fall within the range of 0–950 kcal per 100g.
5. Requires a non-empty USDA food description.
6. Verifies that the description exists in the installed USDA food taxonomy files.

RecipeDB comparison differences are retained in `outputs/nutrient_source_review.csv` when the relevant audit file is available.

**Important:** RecipeDB-derived nutrient comparisons are diagnostic. They do not independently invalidate a manually selected USDA mapping. The validation checks record consistency and taxonomy existence; semantic correctness of every manual USDA ingredient mapping still requires human review.

## 4. Nutritional Substitution Objectives

The system supports five substitution objectives.

| Objective | Requirement |
|---|---|
| `more_protein` | At least 3g more protein per 100g; carbohydrate increase limited to 2g |
| `less_carbs` | At least 3g fewer carbohydrates per 100g; protein decrease limited to 2g |
| `less_calories` | At least 10 kcal fewer per 100g; protein decrease limited to 2g |
| `less_fat` | At least 1g less fat per 100g; protein decrease limited to 2g |
| `plant_swap` | Existing equal-protein plant-protein substitution objective |

These thresholds act as hard nutritional constraints. Candidates that fail the relevant requirements are excluded before similarity-based ranking.

The calorie and fat thresholds are initial operational settings and can be adjusted after evaluating candidate coverage and domain requirements.

## 5. GastroGraph-Based Contextual Retrieval

GastroGraph represents relationships between ingredients through their occurrence and context in recipes.

Instead of considering ingredients independently, the system uses graph relationships to identify candidates that are relevant to the target ingredient's recipe context.

This helps retrieve alternatives based on how ingredients are used in cooking.

Context similarity is one of the main retrieval signals. Optional embedding-based similarity can provide additional evidence about ingredient relationships.

## 6. FlavorDB Molecular Similarity

FlavorDB adds a molecular perspective to ingredient substitution.

Each food ingredient can be associated with a set of flavour-related molecules. The system compares these molecular profiles to estimate similarity between ingredient pairs.

### IDF-Weighted Jaccard Similarity

The pipeline uses a precomputed IDF-weighted Jaccard similarity score to measure molecular overlap between ingredient pairs.

**Formula:**

```text
Similarity(A, B) =
    Sum of IDF(m) for molecules shared by A and B
    ------------------------------------------------
    Sum of IDF(m) for all molecules in A or B
```

**Where:**

- **A and B:** Molecular profiles of two ingredients.
- **Shared molecules:** Molecules present in both A and B (intersection).
- **All molecules:** Unique molecules present in either A or B (union).
- **IDF(m):** Inverse Document Frequency weight assigned to molecule m.

**How IDF weighting works:**

- Common molecules receive lower weights.
- Rare or distinctive molecules receive higher weights.
- Shared distinctive molecules contribute more to the similarity score than shared common molecules.

This makes the similarity measure more informative than treating every shared molecule equally.

### FlavorDB Input Files

The pipeline reads precomputed results from:

- `EDA/results/flavour/molecular_similarity.csv`
- `EDA/results/flavour/molecule_audit_entity_stats.csv`

These files are generated by `EDA/eda_03_flavour.ipynb`.

Ingredient names are matched using normalized exact-name matching. Fuzzy matches are not silently accepted.

### Missing Molecular Data

The pipeline distinguishes between two situations:

- **Similarity = 0.0:** Both ingredients have usable molecular profiles, but no shared molecular evidence is found.
- **Similarity = NaN:** One or both ingredients have missing, unmapped, or insufficient molecular profiles.

Identical molecular profiles retain their calculated score and are flagged for cautious interpretation.

FlavorDB similarity is an additional retrieval signal, not a hard filter. Molecular similarity alone does not establish that two ingredients are valid culinary substitutes.

## 7. Candidate Retrieval and Ranking

The substitution process follows these steps:

1. The user selects an ingredient and a nutritional objective.
2. GastroGraph retrieves contextually related ingredient candidates.
3. Nutritional validation checks whether candidates satisfy the selected objective and its constraints.
4. FlavorDB contributes molecular similarity wherever usable profiles are available.
5. Optional embeddings contribute additional similarity evidence.
6. Eligible candidates are ranked using the available retrieval signals.
7. The system returns ranked alternatives with nutritional differences and supporting explanations.

The overall architecture combines nutritional constraints with contextual and molecular evidence rather than relying on a single similarity measure.

## 8. Project Structure

```text
BTP_2/
│
├── EDA/
│   ├── eda_03_flavour.ipynb
│   ├── results/
│   │   └── flavour/
│   │       ├── molecular_similarity.csv
│   │       └── molecule_audit_entity_stats.csv
│   │
│   └── eda_06_nutrient_audit.ipynb
│
└── substitution/
    ├── build_master.py
    ├── config.py
    ├── retrieval.py
    ├── ranking.py
    ├── substitute.py
    ├── cook.py
    ├── evaluate.py
    ├── gold_eval.py
    ├── recipes.py
    │
    └── outputs/
        ├── ingredient_master.csv
        ├── nutrient_source_review.csv
        ├── review_ingredient_level.csv
        ├── review_recipe_level.csv
        ├── evaluation_summary.txt
        ├── gold_results_per_query.csv
        └── gold_summary.txt
```

*The structure above highlights the main modules and outputs; other project data and supporting files are omitted for clarity.*

## 9. Installation and Execution

Run all commands from the `BTP_2/substitution/` directory.

### Requirements

The pipeline primarily uses:

- Python 3
- pandas
- NumPy
- SciPy

Additional packages are required for optional embedding functionality.

Install the dependencies used by the project:

```bash
pip install pandas numpy scipy
```

### Step 1: Build the Nutritional Master

```bash
python3 build_master.py
```

This creates:

- `outputs/ingredient_master.csv`
- `outputs/nutrient_source_review.csv` (when the RecipeDB audit file is available)

### Step 2: Evaluate Substitution Candidates

```bash
python3 evaluate.py
```

This evaluates ingredient-level and recipe-level substitution candidates and produces:

- `outputs/review_ingredient_level.csv`
- `outputs/review_recipe_level.csv`
- `outputs/evaluation_summary.txt`

The evaluation independently checks nutritional constraints and raises an assertion if any returned candidate violates a hard constraint.

### Step 3: Optional Gold-Standard Evaluation

```bash
python3 gold_eval.py
```

This evaluates retrieval performance against the project's gold-standard substitution pairs and generates:

- `outputs/gold_results_per_query.csv`
- `outputs/gold_summary.txt`

### Step 4: Interactive Recipe Substitution

```bash
python3 cook.py
```

This allows users to select a recipe, choose an ingredient, specify a substitution objective, and inspect ranked alternatives.

### Direct Ingredient-Level Examples

```bash
# Tofu – less fat
python3 substitute.py --ingredient tofu --objective less_fat

# Butter – less fat
python3 substitute.py --ingredient butter --objective less_fat
```

These examples demonstrate ingredient-level retrieval, while `cook.py` provides the interactive recipe-level workflow.

## 10. Safety Checks and Reliability

The pipeline includes several safeguards:

- Only trusted nutritional profiles are eligible for ranking.
- All four nutritional values must be finite and valid.
- Missing target or guard nutrient values cause candidates to be rejected.
- Hard nutritional constraints are applied before similarity ranking.
- RecipeDB-derived profiles are not automatically promoted to trusted nutritional data.
- Missing FlavorDB evidence is represented as unavailable rather than being treated as a meaningful zero score.
- `evaluate.py` independently rechecks returned candidates for nutritional constraint violations.

## 11. Limitations and Future Improvements

The current implementation has several limitations:

- Trusted nutritional coverage is limited to manually mapped USDA ingredient profiles.
- FlavorDB molecular information is available for only a subset of ingredients.
- Ingredient name matching uses normalized exact matching, so unmatched names may reduce molecular coverage.
- Molecular similarity does not guarantee equivalent taste or culinary functionality.
- Recipe context and embedding similarity can retrieve relevant candidates, but do not independently establish that a replacement will work in every recipe.
- USDA mappings for ambiguous ingredient names still require manual semantic review.

Future improvements can focus on expanding validated nutritional coverage, improving ingredient-name mapping, increasing molecular profile coverage, and evaluating substitutions with culinary and domain-specific validation.

## 12. Reproducibility and Data Requirements

The full pipeline depends on local project datasets and supporting files, including:

- GastroGraph files
- Compressed RecipeDB data
- USDA SR/Foundation food taxonomy files
- FlavorDB similarity and molecular audit outputs
- Optional embedding files

The complete build and evaluation should be run in the configured BTP_2 project environment. Syntax checks and synthetic tests alone do not establish that the full pipeline will run without the required local datasets.
