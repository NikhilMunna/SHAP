import shap
import xgboost
import pandas as pd
shap.initjs()

# Load Diabetes dataset
X, y = shap.datasets.diabetes()

# Shape
X.shape, y.shape

# Distribution of target variable
pd.Series(y).plot('hist')


# Train using XGBoost Regressor model
XGB_model = xgboost.XGBRegressor()
XGB_model.fit(X, y)


# Create Tree explainer
explainer = shap.TreeExplainer(XGB_model)

# Extract SHAP values to explain the model predictions
shap_values = explainer.shap_values(X)


# Plot Feature Importance
shap.summary_plot(shap_values, X, plot_type="bar")


# Plot Feature Importance - 'Dot' type
shap.summary_plot(shap_values, X, plot_type='dot')


# Visualize the explanation of first prediction
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])


# Visualize the training set using SHAP predictions
shap.force_plot(explainer.expected_value, shap_values, X)



