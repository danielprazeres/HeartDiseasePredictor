import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the absolute path to the dataset
data_path = os.path.join(script_dir, '../data/heart_disease_uci.csv')

# Try to load the dataset
try:
    heart_data = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
    print(heart_data.head())  # Display the first 5 rows of the dataset
except FileNotFoundError:
    print("Dataset not found. Please check the data directory.")
    exit()

# Print general information about the dataset
print("\nGeneral Information:")
print(heart_data.info())

# Describe the dataset to get basic statistics of numerical columns
print("\nBasic Statistics:")
print(heart_data.describe())

# Check for missing values in the dataset
print("\nMissing Values:")
print(heart_data.isnull().sum())

# Treat missing values
numeric_columns = heart_data.select_dtypes(include=['float64', 'int64']).columns
heart_data[numeric_columns] = heart_data[numeric_columns].fillna(heart_data[numeric_columns].mean())

for column in heart_data.select_dtypes(include=['object']).columns:
    heart_data[column].fillna(heart_data[column].mode()[0], inplace=True)

print("\nMissing Values After Treatment:")
print(heart_data.isnull().sum())

# Convert the target variable to binary (0 and 1)
heart_data['num'] = heart_data['num'].apply(lambda x: 1 if x > 0 else 0)

# Separate the target variable
y = heart_data['num']
X = heart_data.drop('num', axis=1)

# Standardize numerical columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Apply One-Hot Encoding
categorical_columns = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print("\nData After One-Hot Encoding:")
print(X.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Perform the grid search
grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")

# Get the best model
best_rf_model = grid_search.best_estimator_

# Evaluate the best model
best_rf_accuracy = best_rf_model.score(X_test, y_test)
print(f"Best Random Forest Model Accuracy: {best_rf_accuracy:.2f}")

# Additional evaluation metrics
best_rf_y_pred = best_rf_model.predict(X_test)
best_rf_conf_matrix = confusion_matrix(y_test, best_rf_y_pred)
print("\nBest Random Forest Confusion Matrix:")
print(best_rf_conf_matrix)
best_rf_class_report = classification_report(y_test, best_rf_y_pred)
print("\nBest Random Forest Classification Report:")
print(best_rf_class_report)

cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average Cross-Validation Score: {cv_scores.mean():.2f}")

rf_auc = roc_auc_score(y_test, best_rf_y_pred)
print(f"Random Forest AUC-ROC: {rf_auc:.2f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % rf_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the ROC curve
figures_dir = os.path.join(script_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
plt.savefig(os.path.join(figures_dir, 'roc_curve.png'))
plt.show()

# Additional plots (histograms, scatter plots)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Age distribution
axs[0, 0].hist(X['age'], bins=20)
axs[0, 0].set_title('Distribution of Age')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('Frequency')

# Cholesterol distribution
axs[0, 1].hist(X['chol'], bins=20)
axs[0, 1].set_title('Distribution of Serum Cholesterol')
axs[0, 1].set_xlabel('Cholesterol (mg/dl)')
axs[0, 1].set_ylabel('Frequency')

# Age vs Cholesterol
axs[1, 0].scatter(X['age'], X['chol'], alpha=0.5)
axs[1, 0].set_title('Age vs. Serum Cholesterol')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Cholesterol (mg/dl)')

# Age vs Max Heart Rate
axs[1, 1].scatter(X['age'], X['thalch'], alpha=0.5)
axs[1, 1].set_title('Age vs. Maximum Heart Rate Achieved')
axs[1, 1].set_xlabel('Age')
axs[1, 1].set_ylabel('Maximum Heart Rate')

# Save the additional plots
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'additional_plots.png'))
plt.show()

# Save the best model
model_path = os.path.join(script_dir, 'models', 'best_random_forest_model.pkl')
joblib.dump(best_rf_model, model_path)
print(f"Model saved to {model_path}")
