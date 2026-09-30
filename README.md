# heart-attack-prediction
End-to-end machine learning project for heart attack prediction using Python and Scikit-learn.

# ❤️ Heart Attack Prediction Using Machine Learning

An end-to-end machine learning project that explores and predicts heart attack occurrence using demographic, lifestyle, and health-related information from a healthcare dataset containing over 246,000 records.

The project covers the complete machine learning workflow, including data cleaning, exploratory data analysis, feature encoding, preprocessing, model development, evaluation, hyperparameter tuning, and feature importance analysis.

> **Note:** This project is for educational and analytical purposes and is not intended for clinical diagnosis or medical decision-making.

---

## 📌 Project Overview

Heart attack prediction is a binary classification problem where identifying individuals at higher risk can be challenging, particularly when the target variable is highly imbalanced.

In this project, I used a healthcare dataset containing **246,022 records and 40 original columns** to investigate patterns associated with heart attack occurrence.

The workflow included:

- Data inspection and quality checks
- Duplicate detection and removal
- Exploratory Data Analysis (EDA)
- Categorical and binary variable encoding
- Feature engineering and preprocessing
- Stratified train-test splitting
- Machine learning model development
- Model comparison
- Hyperparameter tuning using `RandomizedSearchCV`
- Model evaluation using multiple classification metrics
- Feature importance and coefficient analysis

---

## 🎯 Project Objectives

The main objectives of this project were to:

- Explore and understand the healthcare dataset through exploratory data analysis.
- Clean and preprocess the data for machine learning.
- Build multiple classification models for heart attack prediction.
- Compare model performance using appropriate evaluation metrics.
- Improve model performance through hyperparameter tuning.
- Identify influential features associated with heart attack prediction.
- Determine a suitable model based on the characteristics of the imbalanced classification problem.

---

## 📊 Dataset

The original dataset contains:

- **246,022 records**
- **40 columns**
- Demographic information
- Lifestyle information
- Medical history
- Health-related characteristics

Examples of variables include:

- State
- Sex
- General Health
- Physical Health Days
- Mental Health Days
- Physical Activities
- Sleep Hours
- BMI
- Smoking Status
- Diabetes
- Angina
- Stroke
- Asthma
- COPD
- Arthritis
- Kidney Disease
- Alcohol Consumption
- Vaccination History
- Age Category
- Chest Scan
- Other health-related variables

### Target Variable

The target variable is:

`HadHeartAttack`

It indicates whether an individual has experienced a heart attack.

### Class Distribution

The target variable is imbalanced:

| Class | Proportion |
|---|---:|
| No heart attack | 94.54% |
| Heart attack | 5.46% |

Because of this imbalance, accuracy alone is not sufficient for evaluating model performance.

---

## 🧹 Data Cleaning & Preprocessing

The dataset was examined for:

- Missing values
- Duplicate records
- Data types
- Categorical variables
- Binary variables
- Feature distributions

The original dataset contained **9 duplicate rows**, which were identified and removed.

After duplicate removal, the dataset contained:

**246,013 records**

### Feature Encoding

Several variables were converted into numerical representations.

Binary Yes/No variables were encoded as:

- Yes → 1
- No → 0

Other categorical variables were transformed using appropriate mappings or one-hot encoding.

Examples include:

- Sex
- General Health
- Age Category
- Last Checkup Time
- Removed Teeth
- Smoking Status

Nominal categorical variables were converted using `pandas.get_dummies()`.

After preprocessing and encoding, the dataset contained:

- **101 input features**
- **1 target variable**

---

## 🔎 Exploratory Data Analysis

Exploratory analysis was performed to investigate relationships between health characteristics and heart attack occurrence.

Visualisations included analysis of:

- Sex and angina
- BMI distribution
- Smoking status and heart attack occurrence
- Sleep hours and heart attack status
- Heart attack class distribution
- Feature correlations

The analysis also highlighted the importance of considering the imbalance in the target variable when interpreting model performance.

---

## 🤖 Machine Learning Models

Five classification algorithms were developed and compared:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Linear Support Vector Classifier (LinearSVC)**
5. **K-Nearest Neighbors (KNN)**

Scikit-learn pipelines were used for preprocessing and model training.

Where appropriate, `StandardScaler` was used before model training.

Class weighting was also considered for models such as Logistic Regression, Random Forest, Decision Tree and LinearSVC to help address the imbalanced target variable.

---
## **📏 Train-Test Split**

The dataset was divided using an **80/20 stratified train-test split**.

```text
Training records: 196,810  
Testing records: 49,203

---

## 📈 **Baseline Model Performance**


The initial models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC AUC**
- **Confusion Matrix**

## **Baseline Results**

| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8339 | 0.2162 | 0.7778 | 0.3384 | 0.8888 |
| Random Forest | 0.9451 | 0.4967 | 0.4239 | 0.4574 | 0.8812 |
| Decision Tree | 0.9159 | 0.2558 | 0.2825 | 0.2685 | 0.6175 |
| LinearSVC | 0.8419 | 0.2233 | 0.7648 | 0.3457 | 0.8887 |
| KNN | 0.9442 | 0.4603 | 0.1273 | 0.1994 | 0.7325 |

These results demonstrate why multiple evaluation metrics are important for this dataset. Models with high accuracy did not necessarily achieve strong recall or F1-score for the minority class.

---

## ⚙️ **Hyperparameter Tuning**

Hyperparameter optimisation was performed using:

`RandomizedSearchCV`

with:

- **3-fold cross-validation**
- **F1-score** as the scoring metric
- **random_state = 42**

Hyperparameter tuning was performed for:

- **Logistic Regression**
- **LinearSVC**
- **Random Forest**

### **Tuned Logistic Regression**

**Best parameters:**

```text
C = 100
penalty = l2
class_weight = None
---

Tuned LinearSVC

Best parameters:

C = 10
class_weight = balanced
tol = 0.0001
max_iter = 5000
Tuned Random Forest

Best parameters:

n_estimators = 200
max_depth = 20
min_samples_split = 2
min_samples_leaf = 1
max_features = sqrt
class_weight = balanced
🏆 Model Selection

Based on the tuned models evaluated in this project:

Random Forest achieved the highest F1-score among the tuned models at 0.4442.
Logistic Regression achieved the highest accuracy at 94.82%.
LinearSVC achieved the highest recall at 76.48% among the tuned models.

The Tuned Random Forest was selected as the final model because F1-score provided a useful balance between precision and recall for this imbalanced classification problem.

However, the choice of model depends on the objective. If the priority were specifically to identify as many positive heart attack cases as possible, recall would become particularly important, making the LinearSVC results relevant to that consideration.

📊 Model Visualisations
Model Comparison

ROC Curves

Precision-Recall Curves

Confusion Matrix

Feature Importance

🔍 Feature Analysis

Feature analysis was performed using the tuned Random Forest model and Logistic Regression coefficients.

The Random Forest feature-importance analysis examined the top 20 features contributing to the model's predictions.

The Logistic Regression coefficient analysis also provided insight into features with stronger positive and negative relationships within the fitted model.

Some influential variables identified during the analysis included:

HadAngina
AgeCategory
ChestScan
HadStroke
SmokerStatus
HadDiabetes
LastCheckupTime
Other demographic and health-related variables

These model-based relationships should be interpreted as predictive associations within this dataset and should not be treated as evidence of causation.

🧰 Technologies Used
Programming & Data Analysis
Python
Pandas
NumPy
Data Visualisation
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Logistic Regression
Decision Tree
Random Forest
LinearSVC
K-Nearest Neighbors
RandomizedSearchCV
Development Environment
Jupyter Notebook
💡 Key Takeaways

This project provided practical experience across the complete machine learning workflow.

The main lessons from the project include:

Accuracy can be misleading when the target variable is imbalanced.
Precision, recall and F1-score provide additional insight into minority-class performance.
Stratified splitting helps maintain class proportions between training and testing data.
Scikit-learn pipelines provide a structured approach to preprocessing and model training.
Hyperparameter tuning can change model performance and the relative trade-offs between metrics.
Different models may be preferable depending on the objective of the prediction task.
Feature importance and model coefficients can help improve interpretation of machine learning models.
📚 Project Reflection

This project strengthened my understanding of the complete machine learning workflow, from data cleaning and exploratory data analysis to model development, evaluation and hyperparameter tuning.

One of the biggest lessons was that high accuracy alone can be misleading when working with an imbalanced dataset. Evaluating models using Precision, Recall, F1-score and ROC AUC provided a more meaningful assessment of classification performance.

I also gained practical experience with:

Scikit-learn pipelines
Classification modelling
Model comparison
Feature encoding
RandomizedSearchCV
Cross-validation
Feature importance
Model evaluation
Working with an imbalanced healthcare dataset

Overall, the project improved my ability to apply machine learning techniques to a real-world dataset and make modelling decisions based on evidence from multiple evaluation metrics.

👩🏽‍💻 Author

Lola

Data Scientist | Data Analyst | Python | SQL | Machine Learning

GitHub: @Lolalingo

---
## 🚀 **How to Run the Project**

### 1. Clone the repository

```
```

```
git clone https://github.com/Lolalingo/heart-attack-prediction.git
```

### 2. Navigate to the project directory

```
```

```
cd heart-attack-prediction
```

### 3. Install the required libraries

```
```

```
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```
```

```
jupyter notebook
```

Open:

```
```

```
ML_project.ipynb
```
