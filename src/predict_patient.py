import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

# Function to predict heart disease
def predict_heart_disease(patient_data):
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Construct the path to the saved model
    model_path = os.path.join(script_dir, 'models', 'best_random_forest_model.pkl')
    
    # Load the trained model
    model = joblib.load(model_path)
    
    # Create a DataFrame from patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Check if the columns match those used during training
    expected_columns = model.feature_names_in_
    patient_df = pd.get_dummies(patient_df).reindex(columns=expected_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(patient_df)
    probability = model.predict_proba(patient_df)[:, 1]
    
    has_heart_disease = prediction[0] == 1
    return has_heart_disease, probability[0]

# Patient data for testing
patient_data = {
    'age': 55,
    'trestbps': 140,
    'chol': 230,
    'fbs': 1,
    'thalch': 150,
    'exang': 0,
    'oldpeak': 2.0,
    'ca': 0,
    'sex_Male': 1,
    'dataset_Hungary': 0,
    'dataset_Switzerland': 0,
    'dataset_VA Long Beach': 0,
    'cp_atypical angina': 0,
    'cp_non-anginal': 1,
    'cp_typical angina': 0,
    'restecg_normal': 1,
    'restecg_st-t abnormality': 0,
    'slope_flat': 1,
    'slope_upsloping': 0,
    'thal_normal': 1,
    'thal_reversable defect': 0,
    'id': 1  # Add the 'id' column
}

# Get the prediction and probability
has_heart_disease, probability = predict_heart_disease(patient_data)
print(f"Prediction: {'Has heart disease' if has_heart_disease else 'No heart disease'}")
print(f"Probability: {probability:.2f}")

# Load data to generate comparison plots
data_path = os.path.join(os.path.dirname(__file__), '../data/heart_disease_uci.csv')
heart_data = pd.read_csv(data_path)

# Handle missing values
numeric_columns = heart_data.select_dtypes(include=['float64', 'int64']).columns
heart_data[numeric_columns] = heart_data[numeric_columns].fillna(heart_data[numeric_columns].mean())
for column in heart_data.select_dtypes(include=['object']).columns:
    heart_data[column].fillna(heart_data[column].mode()[0], inplace=True)

# Create plots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Age distribution
axs[0, 0].hist(heart_data['age'], bins=20, alpha=0.7, label='Dataset')
axs[0, 0].axvline(patient_data['age'], color='r', linestyle='dashed', linewidth=2, label='Patient')
axs[0, 0].set_title('Distribution of Age')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].legend()

# Cholesterol distribution
axs[0, 1].hist(heart_data['chol'], bins=20, alpha=0.7, label='Dataset')
axs[0, 1].axvline(patient_data['chol'], color='r', linestyle='dashed', linewidth=2, label='Patient')
axs[0, 1].set_title('Distribution of Serum Cholesterol')
axs[0, 1].set_xlabel('Cholesterol (mg/dl)')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].legend()

# Age vs Cholesterol
axs[1, 0].scatter(heart_data['age'], heart_data['chol'], alpha=0.5, label='Dataset')
axs[1, 0].scatter(patient_data['age'], patient_data['chol'], color='r', label='Patient')
axs[1, 0].set_title('Age vs. Serum Cholesterol')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Cholesterol (mg/dl)')
axs[1, 0].legend()

# Age vs Maximum Heart Rate
axs[1, 1].scatter(heart_data['age'], heart_data['thalch'], alpha=0.5, label='Dataset')
axs[1, 1].scatter(patient_data['age'], patient_data['thalch'], color='r', label='Patient')
axs[1, 1].set_title('Age vs. Maximum Heart Rate Achieved')
axs[1, 1].set_xlabel('Age')
axs[1, 1].set_ylabel('Maximum Heart Rate')
axs[1, 1].legend()

# Save the plots
figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'patient_comparison.png'))
plt.close()  # Close the plot to free memory and avoid displaying it
