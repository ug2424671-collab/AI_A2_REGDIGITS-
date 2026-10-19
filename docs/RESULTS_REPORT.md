# Musanze Harvest Intelligence Service — Results Report

**Student:** Kiir Deng Kur

**Registration number:** 2424671

**Institutional username:** ug2424671@ines.ac.rw

**Verification code:** AI-A2-4671

**Repository:** https://github.com/ug2424671-collab/AI_A2_REGDIGITS-

## Purpose and scope

This classroom prototype demonstrates regression, binary classification, and clustering on
synthetic potato-farm data. Its predictions are not validated for real farms and must not be used
to make decisions about finance, insurance, or public services. A human agronomist must review
any result.

## Data and validation

The deterministic data generator created 281 raw rows. Validation found 14 missing values in
each of `soil_moisture_pct` and `fertilizer_kg`, three exact duplicate rows, and no range
violations. Cleaning removed the duplicates, leaving 278 rows. Missing feature values are imputed
inside the model pipelines, fitted using training data.

The stratified train/test split is 80/20. The test set contains 56 rows. The CSV fingerprint
recreated by the pipeline is:

```text
6ebb71d3b63014bcb27206abd81b4c8662c77a4053a9e7d3987209730defc5db
```

Target-derived columns are excluded from model inputs. The regression and classification models
were selected using five-fold cross-validation on the training set; the test set was reserved for
final evaluation.

## Regression

The selected model is Ridge regression. On the held-out test set, it achieved:

| Metric | Result |
|---|---:|
| MAE | 0.7150 tons |
| RMSE | 0.9183 tons |
| R² | 0.8724 |
| Training five-fold CV RMSE | 0.9268 tons |

**Interpretation:** The MAE means that, on this synthetic test set, absolute prediction error
averaged about 0.72 tons per farm. RMSE penalizes larger errors more strongly and was about
0.92 tons. The R² indicates that the model explained about 87% of the test-set variation in
yield. These values describe this generated dataset only; they do not establish real-world
accuracy.

## High-risk classification

The selected classifier is a random forest. “High risk” means yield per hectare below the
dataset median. On the held-out test set:

| Metric | Result |
|---|---:|
| Precision | 0.8966 |
| Recall | 0.9286 |
| F1 | 0.9123 |
| ROC AUC | 0.9732 |

Confusion matrix (rows = actual class [not high risk, high risk]; columns = predicted class
[not high risk, high risk]):

|  | Predicted not high risk | Predicted high risk |
|---|---:|---:|
| Actual not high risk | 25 | 3 |
| Actual high risk | 2 | 26 |

The model correctly classified 51 of 56 test records. In this evaluation, three records were
false alarms and two high-risk records were missed. A high score on synthetic data should not be
interpreted as evidence that the classifier is suitable for real decisions.

## Clustering

K-means selected **K=2** from the evaluated range K=2–6. Its silhouette score was **0.1598**.
That modest score indicates weak separation, so the clusters are exploratory groupings, not
clear natural categories. One group is primarily farms with cooperative support; the other is
primarily farms without it. The clustering fit uses input features only; yield and risk values in
the clustering report are post-hoc descriptions and were not used to form clusters.

## Reproduction and generated files

Run from the project root:

```bash
.venv/bin/python run_all.py --student-id ug2424671@ines.ac.rw --seed 4671
```

The pipeline writes the dataset fingerprint, validation and model metrics, test predictions,
three plots, saved models, and `models/model_info.json`. The main evidence files are
`artifacts/regression_metrics.json`, `artifacts/classification_metrics.json`,
`artifacts/clustering_report.json`, and `artifacts/plots/`.

## Limitations

The data is synthetic and generated from a known formula. Model selection and evaluation are
performed on data from that same generator, so the metrics are optimistic indicators for this
classroom exercise, not independent evidence of performance on real farms. The clusters are
weakly separated. Predictions require human review and are not a substitute for agronomic
expertise.
