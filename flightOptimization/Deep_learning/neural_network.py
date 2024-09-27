# advanced_model_evaluation_with_neural_network.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Step 1: Load the dataset
df = pd.read_csv(r'Stored_data/filtered_flight_data.csv')
df2 = pd.read_csv(r'Stored_data/airports.csv')
df3 = pd.read_csv(r'Stored_data/flightheading.csv')
df4 = pd.read_csv(r'Stored_data/filtered_flight_data.csv')

# Feature Engineering
# (similar to earlier code: feature engineering, handling missing values, etc.)
df['DEP_DIFF'] = df['DEP_TIME'] - df['CRS_DEP_TIME']
df['ARR_DIFF'] = df['ARR_TIME'] - df['CRS_DEP_TIME']
df['DEP_HOUR'] = df['DEP_TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
df['ARR_HOUR'] = df['ARR_TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
df['IS_DELAYED'] = df['ARR_DELAY'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encoding for origin and destination airports
df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'], drop_first=True)

# Label encoding for cancellation code
df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].fillna('None')
df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].astype('category').cat.codes

# Step 2: Split Data into Training and Test Sets
X = df.drop(columns=['ARR_DELAY', 'ARR_DELAY_NEW'])
y = df['ARR_DELAY_NEW']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Define Models to Evaluate (Linear, Ridge, Lasso, RandomForest, XGBoost)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Step 4: Train and Evaluate Models (including XGBoost)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - Mean Squared Error: {mse}, R²: {r2}')

# Step 5: Implement a Neural Network using TensorFlow/Keras

# Build the Neural Network model
def create_neural_network(input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression (no activation function for regression)
    
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Create model
nn_model = create_neural_network(X_train.shape[1])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the Neural Network
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

# Step 6: Evaluate the Neural Network
y_pred_nn = nn_model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
print(f'Neural Network - Mean Squared Error: {mse_nn}, R²: {r2_nn}')

# Step 7: Fine-tune Neural Network (if needed)
# You can adjust layers, learning rates, etc., based on performance

# Optional: Visualize Loss and RMSE during training
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Step 8: Retrain Best Performing Model (Optional: If including LLM features)
# Repeat earlier steps if including LLM-based textual features
