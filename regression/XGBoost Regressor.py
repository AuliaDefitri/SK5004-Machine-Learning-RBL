import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
data = pd.read_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/Features_Target.csv')

# 2. Pisahkan fitur dan target
X = data.drop(columns=['filename', 'Average_D'])
y = data['Average_D']

# 3. Bagi data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardisasi fitur (meskipun XGBoost tidak wajib, tetap dilakukan untuk konsistensi)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. XGBoost dengan hyperparameter tuning
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

random_search.fit(X_train_scaled, y_train)

# 6. Evaluasi model terbaik
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE dalam persen

print("📊 Hasil Evaluasi Model:")
print(f"1. RMSE: {rmse:.2f}")
print(f"2. R² Score: {r2:.2f}")
print(f"3. MAE: {mae:.2f}")
print(f"4. MAPE: {mape:.2f}%")

# 7. Visualisasi: Plot nilai aktual vs prediksi
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='mediumseagreen', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Average D')
plt.ylabel('Predicted Average D')
plt.title('XGBoost - Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()
