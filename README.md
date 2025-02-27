# Classification-and-Regression-LDA-QDA-Ridge-Regression-Regularization-Analysis

# Classification and Regression with LDA, QDA, and Ridge Regression

## Project Overview
This project focuses on implementing **Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), Ordinary Least Squares (OLS) Regression, and Ridge Regression** to analyze classification and regression performance. The models are evaluated based on accuracy, test errors, and regularization effects. The project includes **gradient descent-based optimization** and **non-linear feature mappings** to improve generalization.

## Key Features
- **LDA vs. QDA**: Compare classification accuracy and decision boundaries.
- **OLS vs. Ridge Regression**: Evaluate regularization impact on model performance.
- **Optimal λ Selection**: Identify the best regularization parameter for minimizing test error.
- **Gradient Descent for Ridge Regression**: Implement efficient optimization for large datasets.
- **Polynomial Feature Expansion**: Analyze model complexity trade-offs in regression.

## Results Summary

### **Classification (LDA & QDA)**
- **LDA Accuracy**: **97.0%**
- **QDA Accuracy**: **96.0%**
- **Key Insights**:
  - LDA assumes equal class covariance, leading to linear boundaries.
  - QDA allows class-specific covariance, enabling better performance on non-linearly separable data.

### **Regression (OLS & Ridge)**
- **MSE Without Intercept**: **106,775.36**
- **MSE With Intercept**: **3,707.84**
- **Ridge Regression (Optimal λ = 0.06)**
  - **Minimum Test Error**: **2851.33**
  - Ridge Regression stabilizes weights and prevents overfitting.

### **Non-Linear Regression**
- **Best Polynomial Degree (p)**:
  - **p = 1** (for λ = 0) → Best linear model.
  - **p = 4** (for λ = 0.06) → Best trade-off for non-linear regression.


