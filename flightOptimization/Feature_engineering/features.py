import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load and Inspect the Data
df = pd.read_csv(r'Stored_data/filtered_flight_data.csv')
df2 = pd.read_csv(r'Stored_data/airports.csv')
df3 = pd.read_csv(r'Stored_data/flightheading.csv')
df4 = pd.read_csv(r'Stored_data/filtered_flight_data.csv')

# Inspect the first few rows
print(df.head())

# Check for data types and missing values
df.info()

# Summary statistics
print(df.describe())

# Step 2: Handle Missing Data
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Handle missing values: For delay columns, replace NaNs with 0 (if delay = 0 when NaN)
delay_cols = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
df[delay_cols] = df[delay_cols].fillna(0)

# Drop columns with excessive missing values (if necessary)
df.dropna(axis=1, thresh=len(df)*0.8, inplace=True)

# Drop rows with missing values in critical columns (optional)
df.dropna(subset=['DEP_TIME', 'ARR_TIME'], inplace=True)

# Step 3: Feature Engineering
# Feature 1: Flight Duration Difference (Actual vs Scheduled)
df['DEP_DIFF'] = df['DEP_TIME'] - df['CRS_DEP_TIME']
df['ARR_DIFF'] = df['ARR_TIME'] - df['CRS_DEP_TIME']

# Feature 2: Hour of Departure and Arrival
df['DEP_HOUR'] = df['DEP_TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))
df['ARR_HOUR'] = df['ARR_TIME'].apply(lambda x: int(str(x).zfill(4)[:2]))

# Feature 3: Day of the Week
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek  # Monday = 0, Sunday = 6

# Feature 4: Delayed Flight Indicator
df['IS_DELAYED'] = df['ARR_DELAY'].apply(lambda x: 1 if x > 0 else 0)

# Step 4: Visualize the Data

# 4.1 Univariate Analysis: Flight Delays
sns.histplot(df['ARR_DELAY'], bins=50)
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.show()

# 4.2 Bivariate Analysis: Delays vs Day of the Week
sns.boxplot(x='DAY_OF_WEEK', y='ARR_DELAY', data=df)
plt.title('Arrival Delays by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Arrival Delay (minutes)')
plt.show()

# 4.3 Categorical Features: Cancellations by Reason
sns.countplot(x='CANCELLATION_CODE', data=df)
plt.title('Cancellations by Reason')
plt.xlabel('Cancellation Code')
plt.ylabel('Count')
plt.show()

# Step 5: Correlation Analysis
# Correlation heatmap
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 6: Handle Categorical Variables
# One-hot encoding for origin and destination airports
df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'], drop_first=True)

# Label encoding for cancellation code
df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].fillna('None')
df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].astype('category').cat.codes

# Step 7: Split Data for Modeling
# Select features and target variable
X = df.drop(columns=['ARR_DELAY', 'ARR_DELAY_NEW'])
y = df['ARR_DELAY_NEW']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 8: Modeling (Optional)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
