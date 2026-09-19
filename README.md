# Fraud Detection in Financial Transactions Using Ensemble Classification

**MSc Data Science and Artificial Intelligence | Sheffield Hallam University**  
**Researcher:** Aguan Chalamalla  
**Project type:** Applied machine-learning research using anonymised secondary data and supplementary non-real simulated data

> **Project status:** The completed research deliverable is an executed machine-learning notebook, its analysis and evaluation, and reproducibility/model-export evidence. A Flask web/API interface is a **future implementation**; it has not been established as a tested or deployed deliverable. The notebook contains experimental Flask-app generation code, but generating code is not evidence of a completed or validated application.

## Overview

Credit-card fraud is a rare-event classification problem. In the cleaned analytical dataset, fraudulent transactions make up only **0.1667%** of the records. A classifier can therefore obtain very high overall accuracy by predicting almost every transaction as legitimate, even if it misses fraud.

This project evaluates baseline and imbalance-aware ensemble methods using a **leakage-controlled train/validation/test design**. Models are selected using validation **Average Precision (AP, reported as AUPRC)**, and the operating threshold is also selected on validation data. An untouched test set is reserved for final evaluation. Additional experiments examine calibration, limited time-ordered sensitivity, permutation importance and interpretability using a separate synthetic transaction simulator.

**Research question:** How effectively can ensemble classifiers detect fraudulent transactions in a highly imbalanced financial dataset, and which model/imbalance-handling strategy provides a useful balance of precision, recall and AUPRC?

## Key findings

| Measure | Result |
|---|---:|
| Selected model | **Weighted XGBoost** |
| Validation AP / AUPRC (model-selection criterion) | **0.8847** |
| Validation-selected decision threshold (F2) | **0.33606732** |
| Untouched-test AP / AUPRC | **0.8195** |
| Untouched-test ROC-AUC | **0.9734** |
| Untouched-test precision | **79.17%** |
| Untouched-test recall | **80.00%** |
| Test confusion matrix | **TN 56,631 · FP 20 · FN 19 · TP 76** |

The selected model detected **76 of 95** fraud cases in the untouched test set, missed **19**, and produced **20** false-positive alerts. Lowering the threshold from 0.50 to the validation-selected value did **not** recover additional fraud cases on this test set; it increased false positives. The validation-selected threshold is retained to avoid post-hoc tuning on test labels.

These measurements describe performance on this historical benchmark; they **do not** establish readiness for production banking use.

## Dataset

**Primary dataset:** [Credit Card Fraud Detection — Kaggle / ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Dataset characteristic | Value |
|---|---|
| Source transactions | 284,807 |
| Source fraud cases | 492 (approximately 0.172%) |
| Exact duplicate rows removed | 1,081 |
| Analytical transactions | 283,726 |
| Analytical fraud cases | 473 (approximately 0.1667%) |
| Observation period | Approximately two days in September 2013 |
| Predictors | `Time`, `Amount`, `V1`–`V28` |
| Target | `Class`: `0` = non-fraud; `1` = fraud |

`V1`–`V28` are already PCA-transformed and anonymised in the source. The original meanings of these variables are not provided; the study does **not** assign invented business meanings to components such as `V14` or attempt to re-identify cardholders.

**Data-use and ethics documentation:** [BERD metadata record](https://berd-platform.de/records/qcqqe-g6q16) and [Open Data Commons Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/). Public accessibility should not be confused with unrestricted public-domain status. Review the relevant source terms before reusing or redistributing the dataset. The dataset is not included in this README.

## Methodology and workflow

```text
Approved secondary data + licence evidence
                |
       Data quality checks
   (schema, labels, missingness,
        exact duplicates)
                |
   EDA and descriptive analysis
                |
   Stratified split: 60 / 20 / 20
                |
      Training-only modelling
  (class weights / balanced methods /
     SMOTE inside the pipeline)
                |
   Compare validation AP / AUPRC
                |
  Select model and F2 threshold
         on validation only
                |
       Untouched test set
                |
   Error counts + PR/ROC curves
    + calibration + cross-validation
                |
 Limited time-ordered sensitivity
                |
  Permutation importance on real data
                |
 Separate synthetic simulator + SHAP
                |
   Results, figures, model/export
       and reproducibility evidence
                |
  [FUTURE] Validated Flask web/API
```

**Leakage prevention:** The source data is audited and split before resampling. SMOTE, when used, is confined to the training pipeline rather than being applied to the whole dataset or to the validation/test subsets. The final test set is not used to select the model or threshold.

**Time-ordered analysis:** A supplementary ordered split achieved AP/AUPRC of approximately **0.7845**. Because the real dataset spans only about **48 hours**, this is a *limited ordering-sensitivity check*, **not** evidence of long-term concept drift or operational temporal generalisation.

## Models compared

The executed notebook compares these seven approaches:

| Model | Role / reasoning |
|---|---|
| Dummy prior | Trivial baseline to establish a minimum comparison |
| Weighted Logistic Regression | Linear, class-weighted baseline |
| Random Forest with balanced class treatment | Bagging-style tree ensemble |
| Balanced Random Forest | Imbalance-aware forest using balanced samples |
| EasyEnsemble | Ensemble of learners trained on balanced subsets |
| SMOTE + Logistic Regression | Data-level resampling versus class-weighted approaches |
| Weighted XGBoost | Boosted trees with positive-class weighting |

**Weighted XGBoost** had the highest validation AUPRC among these executed approaches. Optional stacking, SMOTEENN and extensive tuning paths should **not** be described as completed experiments unless separately run and documented.

### Evaluation

The primary selection metric is **Average Precision / AUPRC** because precision-recall analysis focuses on performance for the rare fraud class. The notebook also reports precision, recall, F1, F2, ROC-AUC, balanced accuracy, MCC, confusion matrices and probability-calibration diagnostics. Overall accuracy is **not** the primary decision criterion.

- **Precision:** Of the flagged transactions, how many were actually fraudulent?
- **Recall:** Of the actual frauds, how many were detected?
- **False negative:** A fraudulent transaction incorrectly cleared.
- **False positive:** A legitimate transaction incorrectly flagged.
- **F2:** A precision-recall summary that places more weight on recall; used for validation-only threshold selection.

## Explainability and supplementary simulation

For the real benchmark, permutation importance evaluates which anonymised components affect AP/AUPRC. **V14** was the strongest measured component, but its original business meaning is unknown.

The approved **non-real** simulator is a separate interpretability experiment. It generates synthetic customers, terminals and transaction scenarios, including abnormal amounts, stolen-card-like behaviour and compromised terminals. It provides interpretable features such as `AMOUNT_ZSCORE` and `DISTANCE_TO_TERMINAL`; permutation importance and SHAP then help explain the synthetic model's behaviour.

Simulator reference: [Reproducible Machine Learning for Credit Card Fraud Detection — simulated dataset](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html).

**Important:** Synthetic-feature findings do not describe the original PCA inputs. High performance on scripted synthetic scenarios is not external validation of the real-data model.

## Run the notebook on Kaggle

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and create a notebook, or import this repository's **`creditcard-fraud-aguan.ipynb`**. If you upload it under a different name, open that `.ipynb` instead.
2. Select **Add Input** and attach the dataset [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. Ensure the notebook environment has the required Python packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `joblib`; `shap` is needed for optional SHAP analysis.
4. Review the configuration cell. Its recorded default settings include:

   ```python
   SEED = 42
   FAST_MODE = True
   RUN_CROSS_VALIDATION = True
   RUN_STACKING = False
   RUN_TUNING = False
   RUN_SHAP = True
   RUN_SIMULATOR = True
   DROP_EXACT_DUPLICATES = True
   ```

5. Use **Run All**. The loader searches recursively under `/kaggle/input` for `creditcard.csv`. It also accepts `creditcard.csv` in the current folder or at `data/creditcard.csv` when run locally.
6. Inspect the notebook outputs and download the required evidence from `/kaggle/working`.

Running the full notebook may take time. Some optional procedures can be disabled in the configuration cell. Reproduced metrics may differ if the data, versions, random-state handling or configuration changes; consult the saved environment metadata for the original run.

### Optional local notebook setup

Python and Jupyter must be installed. For an isolated environment:

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost shap joblib jupyter
jupyter notebook creditcard-fraud-aguan.ipynb
```

Download `creditcard.csv` separately from Kaggle and place it in the repository root or `data/` before running locally. The notebook uses `outputs/` outside Kaggle instead of `/kaggle/working`.

## Outputs and repository contents

The following names are taken from the notebook. The notebook itself must be added to the GitHub repository; generated outputs will exist **only after running it**.

```text
README.md                              # This project overview
creditcard-fraud-aguan.ipynb           # Executed research notebook (add this file)
    fraud_flask_app/                    # Generated experimental Flask code; future validation needed
CreditCard-fraud-aguan.ipynb
```

This is an **expected layout**, not a claim that every listed file has already been committed to GitHub. Review the generated manifest for the exact output files from your run. The notebook's generated Flask files are experimental code-generation output; **application implementation, formal endpoint testing, user evaluation and operational deployment remain future work** in the assessed project.

Do **not** commit private files, credentials, signed ethics paperwork with personal details, or unreviewed raw datasets/model artefacts. If sharing model files, review source licence terms, file size, model provenance and the security implications of loading `joblib`/pickle artefacts.

## Scope, ethics and limitations

- **Ethics:** Approved UREC1 route; no human participants, surveys, interviews, informal user acceptance testing or re-identification.
- **Evidence:** Report aggregated performance, diagnostic plots and non-identifying results. Handle retained materials according to Sheffield Hallam University requirements.
- **Dataset limitations:** Historical 2013 transactions, approximately two days of coverage, anonymised features and no demonstrated transfer to modern banking operations.
- **Threshold limitation:** The validation-selected threshold did not improve fraud recall over 0.50 on the untouched test set.
- **Simulation limitation:** Artificial scenarios cannot establish performance on real customers or other financial institutions.
- **Deployment limitation:** No tested, production-ready Flask system is claimed.

### Future work

Modern longitudinal and external datasets; cost-sensitive threshold selection; calibration and drift monitoring; broader tuning; validated API/web implementation; security and automated endpoint tests; and, **only following suitable ethics approval**, human-centred interface evaluation.

## Academic integrity and acknowledgement

This is an MSc research project by **Aguan Chalamalla**, Sheffield Hallam University. Thanks to supervisor **Thomas Pickard** for guidance. Dataset and simulator creators retain credit for their resources. Cite the dataset, supporting publications and source licences in any derivative research. AI-assisted drafting or coding support must be disclosed in line with the University's applicable rules; the author remains responsible for checking code, results, references and claims.

**Use restriction:** This repository is an academic research demonstration, not financial advice and not an autonomous fraud-blocking service. No production suitability, customer-level validation or universal fraud-detection guarantee is claimed.
