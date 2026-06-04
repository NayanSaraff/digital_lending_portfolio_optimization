# LendItEazy Project Review

This submission package is a self-contained review bundle for the LendItEazy synthetic lending-risk project. It is designed so a reviewer can quickly understand what the project does, where the data comes from, how the notebooks are ordered, what the charts and models show, and how to rerun the workflow if needed.

## Project Summary

LendItEazy is a lending portfolio risk analysis project built around a synthetic credit dataset. The workflow generates borrower, loan, repayment, and behavioral signal tables, then uses those tables to produce exploratory charts and train predictive risk models.

The project is useful for understanding three things:

1. How borrower and loan attributes can be transformed into structured risk data.
2. How delinquency and behavioral patterns can be visualized as early warning signals.
3. How simple machine learning models can support loan default prediction and credit policy decisions.

## What Is Included

- `data/` - core CSV tables used across the project.
	- borrower profiles
	- loan details
	- loan outcomes
	- repayment behavior
	- behavioral signals
- `notebooks/` - notebook-based workflow that generates the data and analysis outputs.
- `outputs/charts/` - exported visual charts used for presentation and review.
- `outputs/models/` - trained model files produced by the modeling workflow.
- `outputs/` - merged and summary CSV outputs for downstream analysis.
- `modeling/` - training script and dependencies for the model-building step.

## How The Project Works

The pipeline follows a simple sequence:

1. Generate customer profiles.
2. Generate loan accounts tied to those customers.
3. Generate loan outcomes, including default behavior.
4. Generate repayment behavior over time.
5. Generate behavioral signals that act as early-warning features.
6. Create EDA charts and summary outputs.
7. Train prediction models from the engineered features.

The notebooks and CSVs are arranged so that each stage can be inspected independently, but they also connect into a full analytical flow from raw synthetic data to risk decision support.

## Recommended Review Order

1. Read [STEP_BY_STEP_EXECUTION.md](STEP_BY_STEP_EXECUTION.md) for the exact run order.
2. Open the notebooks in `notebooks/` to see how each dataset is produced.
3. Review the charts in `outputs/charts/` to understand the strongest risk patterns.
4. Inspect `outputs/models/` to see the trained model artifacts.
5. Review the CSVs in `data/` and `outputs/` if you want to validate joins, summaries, or model inputs.

## Important Project Themes

This project focuses on credit risk behavior, not general data science experimentation. The main analytical themes are:

- borrower segment risk by employment type
- risk differences by product type
- acquisition-channel quality
- city-tier risk differences
- repayment deterioration before default
- behavioral signal aggregation for early warning

The generated charts in `outputs/charts/` are meant to make these themes easy to review visually.

## Output Interpretation

The most useful files for quick understanding are:

- `outputs/risk_scored_portfolio.csv` - consolidated portfolio view with risk scoring outputs.
- `outputs/merged_stage4.csv` - merged stage-level analytical dataset.
- `outputs/stage4_key_numbers_summary.csv` - compact summary of the main metrics.
- `outputs/charts/` - the visual evidence behind the report conclusions.
- `outputs/models/` - the model artifacts used for prediction or comparison.

If you want a fast review, start with the charts and the summary CSV, then move to the notebooks if you need to understand how the results were produced.

## Reproducibility

If you want to regenerate the modeling step, use `modeling/train_models.py` after installing the packages listed in `modeling/requirements.txt`. The notebooks are intended to be run from the project root so relative paths resolve correctly.

## Submission Notes

This package is meant for handoff and review. It includes the minimum set of source notebooks, generated data, chart outputs, model outputs, and instructions required for a reviewer to understand and reproduce the project locally.