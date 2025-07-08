# 🍷 Wine Quality Classification System

## 📌 Overview
This machine learning project classifies wine samples as **Good** or **Bad** based on their physicochemical properties. The goal is to predict wine quality and gain insights into how chemical composition affects it. We use a supervised learning approach with a **Random Forest Classifier**.

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository  
- **File**: `winequality.csv`  
- **Wines**: Portuguese "Vinho Verde" red and white variants  
- **Samples**: ~6,500 records  
- **Target Variable**: Wine quality score (originally 3 to 9)

### 🔬 Features (Input Variables)
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

---

## 🧹 Data Preprocessing

1. **Target Transformation**:
   - The `quality` column is converted into a binary label:
     - `1` → Good wine (quality ≥ 7)
     - `0` → Bad wine (quality < 7)

2. **Feature Selection**:
   - Dropped original `quality` column
   - Used all remaining columns as features

3. **Train-Test Split**:
   - 80% training, 20% testing
   - Random state fixed for reproducibility

4. **Scaling**:
   - *Not applied*: Random Forest is robust to feature scaling.
   - For future models like Logistic Regression or SVM, scaling may be considered.

---

## ⚙️ Model Details

- **Algorithm**: `RandomForestClassifier` (from scikit-learn)
- **Goal**: Classify wine as Good or Bad
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## 🧪 How to Run

### 1. Clone the Repository

git clone https://github.com/yourusername/wine-quality-classifier.git  
cd wine-quality-classifier

### 2. Install Requirements

pip install -r requirements.txt

### 3. Add the Dataset

Place the `winequality.csv` file in the project root directory.

### 4. Run the Classifier

python main.py

---

## 📁 Project Structure

wine-quality-classifier/
│
├── main.py                          # Main ML script  
├── winequality.csv                  # Dataset file  
├── wine_quality_classification.csv  # Output predictions (auto-generated)  
├── requirements.txt                 # Project dependencies  
└── README.md                        # Project documentation

---

## 📦 Requirements

Install all Python dependencies:

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn

(You can also generate this using: `pip freeze > requirements.txt`)

---

## 📈 Output

- **Console Output**:
  - Accuracy Score
  - Classification Report
- **File**:
  - `wine_quality_classification.csv` → Contains actual and predicted labels
- **Visualization**:
  - Confusion Matrix plotted using `matplotlib` and `seaborn`

---

## 🚀 Future Improvements

- Use advanced classifiers like XGBoost or SVM  
- Hyperparameter tuning with GridSearchCV  
- Implement multiclass classification (predict exact quality score)  
- Build a Streamlit or Flask web app interface  

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repository and submit a pull request.

---

By using this project, you can learn how physicochemical data can be leveraged for real-world classification tasks in the food and beverage industry.
