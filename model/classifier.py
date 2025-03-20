import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# assume fused_features is the fused feature matrix from attention fusion 
# definition to be changed based on the actual implementation
fused_features = np.hstack([tabular_data_scaled, image_features])

X_train, X_test, y_train, y_test = train_test_split(fused_features, np.random.randint(0, 2, size=100), test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1
)

# 7. Train the Model
xgb_model.fit(X_train, y_train)

# 8. Make Predictions
y_pred = xgb_model.predict(X_test)

# 9. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
