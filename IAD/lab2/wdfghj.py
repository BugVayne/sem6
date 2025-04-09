import xgboost as xgb

# Print the XGBoost version
print("XGBoost version:", xgb.__version__)

# Confirm GPU support
print("GPU is available:", xgb.config_context().get("use_gpu") is not None)