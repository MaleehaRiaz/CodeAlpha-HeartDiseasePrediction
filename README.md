# Disease Prediction from Medical Data Project
I built a heart disease risk prediction model with model explainability, where I used the following:
Dataset: Heart Disease Dataset (Cleveland database), sourced from Kaggle, with 13 clinical features (age, sex, chest pain type, blood pressure, cholesterol, etc.) and a binary target for the presence of heart disease.

# Overview
The raw dataset has 1025 rows, but 723 of them are duplicates left over from resampling. This is fixed early on by removing duplicates, leaving 302 unique patients; all results below are based on this cleaned data.

# Notebook Covers:
 1. Data Cleaning: load the dataset, remove duplicate rows, check that values (age, blood pressure, cholesterol, etc.) fall in plausible clinical ranges, then split into train/test sets.
 2. Statistical Testing: t-tests on numeric features (age, blood pressure, cholesterol, max heart rate, ST depression), chi-square tests on categorical features (sex, chest pain type, ECG results, etc.), and a one-way ANOVA on max heart rate across chest pain types, plus a correlation heatmap.
 3. Feature Enigneering: 4 new features built from the raw columns (a stress-risk score, heart-rate-reserve, an age/heart rate ratio, and a high-risk chest pain flag), ranked using mutual information. SMOTE is applied to the training data only to balance the classes.
 4. Model Training: Logistic Regression, Random Forest, SVM, and XGBoost, each tuned with GridSearchCV, then compared using repeated stratified cross-validation (50 folds total) with 95% confidence intervals, plus a paired t-test between the top 2 models.
 5. Model Evaluation: Confusion matrix, ROC curves, precision-recall curves, and calibration curves for all 4 models, plus a full classification report and sensitivity/specfiicty breakdown for the best model.
 6. SHAP Explainability: Global feature importance, per-patient (local) explanations for individual predictions, and dependence plots for the top features.
 7. Fairness Audit: Check whether the model performs consistently across sex and age groups.
 8. Inference Pipeline: Saves the trained model, scaler, and metadata together into one bundle, with a reusable function to predict on a new patient and explain the prediction.
 9. Streamlit App (app.py): An interactive interface where you enter patient values and get a prediction with a SHAP explanation.
 10. Error Analysis: Looks specifically at which patients the model got wrong, how confident it was on those errors, and examines the single worst misclassification in detail.
 11. Threshold Tuning: Compares the default 0.5 classification cutoff against F1- and F2-optimized thresholds, since missing a real disease case is more costly than a false alarm.

# Results:
BEST model: XGBoost (after hyperparameter tuning)

 |Metric	             | Test Set (61 patients, held out)	       | Cross-validation (50 folds)
 |---|                |---|                                     |---|
 |Accuracy	           |            0.738                        |       0.848 ± 0.012
 |Precision           |           	0.758	                       |       0.843 ± 0.016
 |Recall              |          	 0.758	                       |       0.861 ± 0.018
 |F1	                 |            0.758	                       |       0.850 ± 0.012
 |ROC-AUC             |           	0.849	                       |       0.924 ± 0.008

1. XGBoost was not statistically significantly better than Logistic Regression (paired t-test on F1 across 50 folds, p = 0.499), so both are reported rather than treating the CV ranking as conclusive.

2. Fairness Check: Recall gap between sexes is 0.146 (female recall 0.846, male recall 0.700), comparable performance. The gap between age groups is larger: recall 0.857 for younger patients vs. 0.583 for older patients (split at the median age), a gap of 0.274. Each subgroup only has 15030 patients in the test set, so this is reported as a limitation worth flagging, not a statistically confirmed result.

3. Error Analysis: 45 of 61 test patients were classified correctly. Of the 16 errors, 8 were false negatives (missed disease), and 8 were false positives. The single worst error was a false positive that the model was 99.9% confident about.

4. Threshold Tuning: At the default 0.5 cutoff, there were 8 false negatives and 8 false positives. Shifting to the F1-optimal threshold (0.11) reduced false negatives to 6 (at the cost of 1 more false positive). The F2-optimal threshold (0.06), which places greater weight on recall, reduced false negatives to 5 but increased false positives to 11. 

# Limitations
Only 302 unique patients after removing duplicates, so results, especially the subgroup fairness numbers, are based on 15-30 patients per group.
