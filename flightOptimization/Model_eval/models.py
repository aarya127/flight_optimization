import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv(r'Stored_data/filtered_flight_data.csv')
df2 = pd.read_csv(r'Stored_data/airports.csv')
df3 = pd.read_csv(r'Stored_data/flightheading.csv')
df4 = pd.read_csv(r'Stored_data/filtered_flight_data.csv')

# Feature engineering from earlier steps
# Same as before: feature engineering and handling missing values

# Assuming feature engineering code from earlier (fill missing, new features, etc.)

# Feature Engineering steps already covered
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

# Step 3: Define Models to Evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Step 4: Train and Evaluate Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - Mean Squared Error: {mse}, R²: {r2}')

# Step 5: Fine-tune with XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_

y_pred_xgb = best_xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f'XGBoost (Best Model) - Mean Squared Error: {mse_xgb}, R²: {r2_xgb}')

# Step 6: Use Hugging Face LLM for Text Feature Extraction (if applicable)
# Assuming there's some textual data for the open-source LLM to use

# For demonstration, we'll use a pretrained sentiment model (for cancellation reason text feature)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Assuming 'CANCELLATION_REASON' contains text data on why the flight was cancelled
# Example: df['CANCELLATION_REASON'] = ['weather', 'technical issues', ...]  (This needs to be in the dataset)

if 'CANCELLATION_REASON' in df.columns:
    def extract_text_features(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        features = torch.mean(outputs.logits, dim=1).detach().numpy()  # Use the logits as features
        return features

    df['CANCELLATION_REASON_FEATURES'] = df['CANCELLATION_REASON'].apply(lambda x: extract_text_features(x))

    # You can then append these features into your X data
    # X = pd.concat([X, pd.DataFrame(df['CANCELLATION_REASON_FEATURES'].tolist())], axis=1)

# Step 7: Retrain Best Performing Model with Text Features (if available)
if 'CANCELLATION_REASON_FEATURES' in df.columns:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    best_xgb_model.fit(X_train, y_train)
    y_pred_final = best_xgb_model.predict(X_test)
    mse_final = mean_squared_error(y_test, y_pred_final)
    r2_final = r2_score(y_test, y_pred_final)
    print(f'Final XGBoost (with LLM text features) - Mean Squared Error: {mse_final}, R²: {r2_final}')
