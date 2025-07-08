import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\laksh\Desktop\wine quality prediction system\winequality.csv")

# Convert quality to binary labels: 1 for Good (>=7), 0 for Bad (<7)
df['quality_label'] = df['quality'].apply( lambda x: 1 if x >= 7 else 0)

# Split into features and target
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Bad', 'Good'])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Save predictions
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv('wine_quality_classification.csv', index=False)

# Visualize confusion matrix
plt.figure(num='Wine Quality Classification',figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Wine Quality Classification Confusion Matrix')
plt.show()
