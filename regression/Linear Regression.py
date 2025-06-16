import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
data = pd.read_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/Features_target.csv')

# 2. Pisahkan fitur dan target
X = data.drop(columns=['filename','Average_D'])
y = data['Average_D']

# 3. Bagi data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# 6. Evaluasi model
y_pred = lr.predict(X_test_scaled)

# Hitung metrik
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE dalam persen

print("📊 Hasil Evaluasi Model:")
print(f"1. RMSE: {rmse:.2f}")
print(f"2. R² Score: {r2:.2f}")
print(f"3. MAE: {mae:.2f}")
print(f"4. MAPE: {mape:.2f}%")

# 7. Visualisasi
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='mediumseagreen', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Average D')
plt.ylabel('Predicted Average D')
plt.title('Linear Regression - Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Tampilkan hasil prediksi vs aktual
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
})

print(results_df.head(12).to_string(
    index=False,
    formatters={
        'Actual': '{:.2f}'.format,
        'Predicted': '{:.2f}'.format,
    }
))
