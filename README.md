# Multiple-Linear-Regression-for-housing
 
# USA Housing — A Linear Regression Study

**Disclaimer:** The analyses and outputs contained herein are provided for educational and illustrative purposes only. The author does not warrant the accuracy, provenance, or completeness of third‑party datasets; results are not generalizable and should not be relied upon for professional, academic, legal, or clinical decisions. Use the Materials at your own risk.

## Introduction

This project implements a custom linear regression model to predict housing prices using the USA Housing dataset. The script utilizes a gradient descent approach with feature scaling to determine how factors like area income, house age, and population impact real estate value. It includes an automated logging system that captures console output and error tracebacks with timestamps for debugging and performance tracking.

For each run, the script organizes output into dedicated directories:
- **logs/logs_test:** Stores session history including user inputs and model metrics.
- **images:** Contains visual evaluations including residual plots and loss convergence curves.

# FINAL OUTPUT OF MODEL
```text
Model Features:
Area Population
Avg. Area House Age
Avg. Area Income
Avg. Area Number of Rooms

mse= 0.08202714761373452
Model Correlation (R): 0.9581
R-Squared: 0.9180
```
### Residual Plot
![Residual Plot](images/Residual_plot_2026-01-14_19-10-21.png)
### Loss Curve

![Loss Curve](/images/loss_curve_2026-01-14_19-10-21.png)


Dataset used for this study:
- https://www.kaggle.com/datasets/aariyan101/usa-housingcsv

## Conclusion

This study evaluated the performance of a custom-built linear regression model on multi-feature housing data.

- **Model Performance**
  - The model achieved a high correlation (R ≈ 0.958) and an R-Squared value of 0.918, indicating that the selected features explain approximately 92% of the variance in housing prices.
  - The use of `StandardScaler` was critical in ensuring the gradient descent algorithm converged efficiently.
  - The Loss Curve demonstrates stable convergence, with the MSE reaching its floor within the allotted epochs.

- **Key Takeaways and Recommendations**
  - **Feature Importance:** "Avg. Area Income" and "House Age" remain dominant predictors in this dataset.
  - **Robustness:** The integration of `sys.excepthook` ensures that any runtime failures (e.g., missing data or improper dimensions) are recorded in the logs with full tracebacks.
  - **Visual Verification:** Residual plots should be checked for "homoscedasticity" (consistent variance) to ensure the linear model's assumptions are not violated.
  - **Scaling:** Always ensure target variables are inverse-transformed after prediction to report metrics in original currency units (Price).
