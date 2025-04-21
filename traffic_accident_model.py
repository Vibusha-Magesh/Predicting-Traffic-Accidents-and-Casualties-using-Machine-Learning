import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load dataset
df = pd.read_csv(r"C:\Users\Saranya\Desktop\Fourth Semester\Machine Learning\Project\US road.csv")

# All 17 features
all_features = [
    'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation',
    'Driving_experience', 'Types_of_Junction', 'Road_surface_type', 'Light_conditions',
    'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 'Pedestrian_movement',
    'Cause_of_accident', 'Road_surface_conditions', 'Number_of_vehicles_involved'
]

target_casualties = 'Number_of_casualties'  # Regression target
target_severity = 'Accident_severity'  # Classification target

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in all_features + [target_severity]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Regression: Predicting number of casualties
X_reg = df[all_features]  # Use all features
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)
y_reg = df[target_casualties]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Train regression model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_reg.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"Number of Casualties Prediction RMSE: {rmse:.2f}")

# Add predicted casualties to dataset for classification
df['Predicted_casualties'] = np.ceil(rf_reg.predict(X_reg_scaled))
features_with_casualties = all_features + ['Predicted_casualties']

# Classification: Predicting accident severity
X_cls = df[features_with_casualties]
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)
y_cls = df[target_severity]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)

# Train classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

search = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5, n_iter=5, random_state=42, n_jobs=-1)
search.fit(X_train_cls, y_train_cls)
rf_cls = search.best_estimator_

rf_cls.fit(X_train_cls, y_train_cls)
y_pred_cls = rf_cls.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"Accident Severity Prediction Accuracy: {accuracy:.2f}")

# 🔹 Select a random test sample
test_sample_index = np.random.randint(0, X_test_reg.shape[0])  # Random index from test set
test_sample = X_test_reg[test_sample_index].reshape(1, -1)

# Convert sample back to original feature space
test_sample_df = pd.DataFrame(scaler_reg.inverse_transform(test_sample), columns=all_features)

# Round small floating-point errors to zero
test_sample_df = test_sample_df.round(5)
test_sample_df[test_sample_df.abs() < 1e-10] = 0

# Predict Number of Casualties
predicted_casualties = np.ceil(rf_reg.predict(test_sample)[0])
test_sample_df['Predicted_casualties'] = predicted_casualties

# Prepare for classification
test_sample_cls = scaler_cls.transform(test_sample_df[features_with_casualties])

# Predict Severity
predicted_severity = rf_cls.predict(test_sample_cls)[0]

# 🔹 Display Features Used for Prediction
print("\n🔹 Features Used for Prediction:")
for feature, value in zip(test_sample_df.columns, test_sample_df.values.flatten()):
    # Decode categorical values if applicable
    if feature in label_encoders:
        value = label_encoders[feature].inverse_transform([int(value)])[0]
    
    print(f"{feature}: {value}")

# 🔹 Output Predictions
print("\n🔹 Predictions:")
print(f"Predicted Number of Casualties: {predicted_casualties:.0f}")

if target_severity in label_encoders:
    predicted_label = label_encoders[target_severity].inverse_transform([predicted_severity])[0]
    print(f"Predicted Accident Severity: {predicted_label}")
else:
    print("Error: 'Accident_severity' not encoded properly.")

# 🔹 Feature Importance Visualization
importance = pd.Series(rf_cls.feature_importances_, index=features_with_casualties)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance for Accident Severity")
plt.show()


