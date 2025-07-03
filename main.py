import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\laksh\Desktop\Beverage quality prediction system\winequality.csv')
X = df.drop('quality', axis=1)
y = df['quality']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Difference':y_test.values-y_pred
})

# Export to CSV
results.to_csv('actual_vs_predicted.csv', index=False)



