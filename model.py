import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv('dataset.csv')

# Feature engineering
df['soil_param'] = df[['Temperature', 'Humidity', 'Moisture']].mean(axis=1)
df['nutrient_param'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1)

# Drop original columns
df = df.drop(['Temperature', 'Humidity', 'Moisture', 
             'Nitrogen', 'Phosphorous', 'Potassium'], axis=1)

# Separate features and target
X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

# Encode categorical features
categorical_features = ['Soil Type', 'Crop Type']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out()) + ['soil_param', 'nutrient_param']
plt.figure(figsize=(10,6))
sns.barplot(x=rf.feature_importances_, y=feature_names)
plt.title('Feature Importance')
plt.show()

# Save model
joblib.dump(rf, 'fertilizer_classifier.pkl')

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')

# Save the label encoder for target variable
joblib.dump(le, 'label_encoder.pkl')