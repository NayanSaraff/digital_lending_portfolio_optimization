# Step-by-Step Execution

Follow these steps to reproduce the project outputs in order.

## 1. Set up the environment

1. Create or activate your Python environment.
2. Install the dependencies from `modeling/requirements.txt`.
3. Make sure you can open and run `.ipynb` notebooks.

## 2. Generate the data

1. Open the notebooks in `notebooks/`.
2. Run them in sequence so each stage builds on the previous one.
3. Confirm the CSV files are created or updated in `data/` and `outputs/`.

Recommended notebook order:

1. `01_generate_customers.ipynb`
2. `02_generate_loans.ipynb`
3. `03_generate_outcomes.ipynb`
4. `04_generate_repayments.ipynb`
5. `05_generate_signals.ipynb`
6. `04_eda_charts.ipynb`
7. `04_eda_depth.ipynb`

## 3. Review charts and summaries

1. Open the images in `outputs/charts/`.
2. Review the summary tables in `outputs/`.

## 4. Train or reload models

1. Review `modeling/train_models.py`.
2. Run the script to regenerate the model files if needed.
3. Check the saved models in `outputs/models/`.

## 5. Final submission check

1. Ensure `README.md` and this file are present at the top level of the submission package.
2. Verify the package includes `data/`, `notebooks/`, `outputs/charts/`, and `outputs/models/`.
3. Share the zip file with reviewers so they can inspect the files locally.