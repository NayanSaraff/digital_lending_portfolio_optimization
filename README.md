# LendItEazy Financial Services — Synthetic Data Generation
### Digital Lending Portfolio Optimization | IIT Guwahati Summer Analytics

---

## Project Overview

This repository contains the synthetic data generation pipeline for **LendItEazy Financial Services Pvt Ltd** — a fictional Hyderabad-based NBFC created as part of the IIT Guwahati Summer Analytics case study on Digital Lending Portfolio Optimization.

The problem statement required participants to design and generate their own synthetic dataset reflecting a realistic digital lending ecosystem. This pipeline produces five interlinked tables spanning the complete borrower lifecycle — from customer onboarding and loan origination to repayment behavior and portfolio outcomes — enabling end-to-end credit risk analysis, early warning signal detection, pricing optimization, and leadership-level performance monitoring.

All data is synthetically generated with controlled distributions and intentional correlations to support meaningful risk segmentation and ML model development.

---

## Repository Structure

```
LendItEazy/
│
├── notebooks/                        # Data generation notebooks (run in order)
│   ├── 01_generate_customers.ipynb   # Generates customer_profile.csv
│   ├── 02_generate_loans.ipynb       # Generates loan_details.csv
│   ├── 03_generate_outcomes.ipynb    # Generates loan_outcome.csv
│   ├── 04_generate_repayments.ipynb  # Generates repayment_behavior.csv
│   └── 05_generate_signals.ipynb     # Generates behavioral_signals.csv
│
├── data/                             # Output CSV files
│   ├── customer_profile.csv
│   ├── loan_details.csv
│   ├── loan_outcome.csv
│   ├── repayment_behavior.csv
│   └── behavioral_signals.csv
│
├── outputs/                          # Analysis outputs
│   ├── risk_scored_portfolio.csv     # ML-scored portfolio (Stage 5)
│   ├── merged_stage4.csv             # EDA merged table (Stage 4)
│   ├── stage4_key_numbers_summary.csv
│   └── charts/                       # 25 EDA charts + ML visualizations
│
└── modeling/                         # ML model training pipeline
    ├── train_models.py
    ├── requirements.txt
    └── README.md
```

---

## Dataset Overview

| Table | Rows | Description |
|---|---|---|
| `customer_profile.csv` | 10,000 | One row per customer — demographics, employment, income, credit score |
| `loan_details.csv` | 12,000 | One row per loan — product, amount, tenure, interest rate, risk grade |
| `loan_outcome.csv` | 12,000 | One row per loan — default flag, recovery, write-off status |
| `repayment_behavior.csv` | 218,000+ | Monthly EMI records — DPD, payment status, bounce flag |
| `behavioral_signals.csv` | 120,000 | Monthly behavioral signals — stress score, inflow drops, volatility |

**Portfolio Summary:**
- Total disbursed: ₹281.3 crore
- Overall default rate: 20.5%
- Analysis period: January 2022 — December 2023

---

## Data Schema

### customer_profile.csv
```
customer_id, age, gender, city_tier, city_name, employment_type,
monthly_income, credit_score, existing_loans, years_employed,
education_level, acquisition_channel, onboarding_date
```

### loan_details.csv
```
loan_id, customer_id, product_type, loan_amount, tenure_months,
interest_rate, emi_amount, risk_grade, origination_date,
processing_fee, loan_purpose, disbursement_date, co_borrower_flag
```

### loan_outcome.csv
```
loan_id, customer_id, product_type, default_flag, days_to_default,
default_month, total_amount_paid, total_amount_pending,
recovery_amount, closure_date, write_off_flag
```

### repayment_behavior.csv
```
repayment_id, loan_id, customer_id, month_number, due_date,
payment_date, amount_due, amount_paid, dpd, payment_status,
bounce_flag
```

### behavioral_signals.csv
```
signal_id, customer_id, month, monthly_inflow, monthly_outflow,
closing_balance, balance_volatility, inflow_drop_flag,
spending_shock_flag, missed_bill_payments, upi_transaction_count,
new_loan_flag, stress_score
```

---

## Table Relationships

```
customer_profile [1] ──────── [*] loan_details
                                        │
                              ┌─────────┴──────────┐
                              │                    │
                             [1]                  [1]
                              │                    │
                             [*]                  [*]
                         loan_outcome    repayment_behavior

customer_profile [1] ──────── [*] behavioral_signals

loan_details [1] ──────────── [1] risk_scored_portfolio
```

**Primary Keys:** `customer_id` (customer_profile), `loan_id` (loan_details, loan_outcome)
**Foreign Keys:** All child tables link to parent tables via `customer_id` or `loan_id`

---

## Prerequisites

**Python 3.8 or higher**

Install required packages:

```bash
pip install numpy pandas faker openpyxl scikit-learn matplotlib seaborn
```

For the ML modeling pipeline:

```bash
pip install -r modeling/requirements.txt
```

---

## Run Order

Execute notebooks sequentially from the project root. Each notebook depends on outputs from the previous step.

```bash
# Step 1 — Generate customer profiles (required by all subsequent notebooks)
jupyter nbconvert --to notebook --execute notebooks/01_generate_customers.ipynb

# Step 2 — Generate loan details
jupyter nbconvert --to notebook --execute notebooks/02_generate_loans.ipynb

# Step 3 — Generate loan outcomes (default flags, recovery, write-offs)
jupyter nbconvert --to notebook --execute notebooks/03_generate_outcomes.ipynb

# Step 4 — Expand loans into monthly repayment records
jupyter nbconvert --to notebook --execute notebooks/04_generate_repayments.ipynb

# Step 5 — Generate monthly behavioral signals per customer
jupyter nbconvert --to notebook --execute notebooks/05_generate_signals.ipynb
```

To run interactively:

```bash
jupyter notebook
```

---

## Key Parameters

Each notebook exposes configuration parameters at the top of the file.

| Parameter | Notebook | Description |
|---|---|---|
| `N` | `01_generate_customers` | Total number of customers |
| `TOTAL_LOANS` | `02_generate_loans` | Total number of loans |
| `SEED` | All notebooks | Random seed for reproducible output |
| `BASE_RATE` | `03_generate_outcomes` | Base default probability before multipliers |
| Employment split probabilities | `01_generate_customers` | Controls segment distribution |
| Product assignment ranges | `02_generate_loans` | Loan amount, tenure, and rate ranges per product |
| DPD behavior probabilities | `04_generate_repayments` | Controls payment stress patterns |
| Signal thresholds | `05_generate_signals` | Stress score and volatility rules |

**To vary the generated data:** Change `SEED` at the top of any notebook. All distributions and correlations remain consistent — only the specific records change.

---

## Designed Correlations

The dataset was engineered with the following intentional risk correlations to enable meaningful segmentation analysis:

| Segment | Default Rate | Key Driver |
|---|---|---|
| Gig Worker | ~33% | Irregular income, high EMI-to-income ratio |
| Salaried | ~11% | Stable income, lower leverage |
| BNPL product | ~28% | First-time borrowers, no credit threshold |
| Social Media channel | ~31% | Unqualified borrower acquisition |
| Referral channel | ~9% | Pre-qualified borrower network |
| Tier 3 cities | ~26% | Income volatility, underpriced for risk |
| Tier 1 cities | ~15% | More stable employment and income |

Early warning signals are embedded such that 74% of eventual defaulters exhibit two or more stress indicators within months 1–3 of loan origination — approximately 60 days before actual default.

---

## Validation Checks

Run the following after each notebook to verify data integrity:

```python
import pandas as pd

cp = pd.read_csv('data/customer_profile.csv')
ld = pd.read_csv('data/loan_details.csv')
lo = pd.read_csv('data/loan_outcome.csv')
rb = pd.read_csv('data/repayment_behavior.csv')

# Row count checks
print(f"Customers       : {len(cp):,}")      # Expected: 10,000
print(f"Loans           : {len(ld):,}")      # Expected: 12,000
print(f"Outcomes        : {len(lo):,}")      # Expected: 12,000
print(f"Repayments      : {len(rb):,}")      # Expected: 218,000+

# Default rate check
print(f"Default rate    : {lo['default_flag'].mean()*100:.1f}%")  # Expected: ~20%

# Referential integrity
orphan_loans = set(ld['loan_id']) - set(lo['loan_id'])
print(f"Orphan loans    : {len(orphan_loans)}")  # Expected: 0

# DPD pattern check
rb_merged = rb.merge(lo[['loan_id','default_flag']], on='loan_id')
dpd_check = rb_merged.groupby('default_flag')['dpd'].mean()
print(f"Avg DPD defaulters    : {dpd_check[1]:.1f}")   # Expected: higher
print(f"Avg DPD non-defaulters: {dpd_check[0]:.1f}")   # Expected: lower
```

---

## Analysis Pipeline

After generating all CSVs, the project follows this analytical workflow:

```
Stage 3 — Data Generation       (this pipeline)
      ↓
Stage 4 — EDA and Segmentation  (notebooks/04_eda_charts.ipynb)
      ↓
Stage 5 — ML Credit Risk Model  (modeling/train_models.py)
      ↓
Stage 6 — Power BI Dashboard    (5-page CRO dashboard)
      ↓
Stage 7 — CRO Report            (6-page credit policy recommendation)
```

---

## Project Context

**Competition:** IIT Guwahati Summer Analytics — Digital Lending Portfolio Optimization

**Problem Statement:** Design a data-driven approach enabling a digital lender to understand customer risk holistically, detect early signs of financial stress, optimize growth strategies without compromising portfolio quality, and support leadership with forward-looking insights.

**Key Deliverables:**
- Credit Policy Recommendation Report (6 pages, addressed to fictional CRO)
- Power BI Dashboard (5 pages, CRO-level monitoring views)

**Standard Applied:** Work should lead with strategic recommendations substantiated by data — not data outputs in search of a conclusion.

---

## Author

**Analysis Period:** January 2022 — December 2023
**Generated:** May 2025
**Classification:** Academic / Competition Submission
