from ast import mod
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO)

def load_data():
    """Load the breast cancer data set from Scikit-Learn."""
    cancer_data = load_breast_cancer()
    df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
    df['target'] = cancer_data['target']
    return df, cancer_data

def visualize_data(df):
    """Visualize the breast cancer data set by displaying certain characteristics."""
    logging.info("First 5 entries of the dataset:\n%s", df.head())
    num_features = df.shape[1] - 1 # Subtract 1 to exclude the target column
    num_entries = df.shape[0]
    benign_count = df['target'].value_counts().get(0, 0)
    malignant_count = df['target'].value_counts().get(1, 0)
    logging.info("Total number of features: %d", num_features)
    logging.info("Total number of entries: %d", num_entries)
    logging.info("Number of benign entries: %d", benign_count)
    logging.info("Number of malignant entries: %d", malignant_count)

def train_model(x, y):
    """Train the logistic regression model on the cancer data set."""
    model = LogisticRegression(solver='liblinear')
    model.fit(x, y)
    return model

def evaluate_model(model, x, y):
    """Evaluate the model accuracy."""
    acc = model.score(x, y)
    logging.info(f"Model accuracy percentage: {acc:.2f}")
    
def feature_significance(model, feature_names):
    """Get and print the top and bottom 5 features based on weight coefficient."""
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Weight Coefficient': coefficients})  
    feature_importance['Absolute Coefficient'] = feature_importance['Weight Coefficient'].abs()
    feature_importance_sorted = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)  
    
    logging.info("Top 5 most significant features\n%s:", feature_importance_sorted.head(5))
    logging.info("Bottom 5 least significant features\n%s:", feature_importance_sorted.tail(5)) 
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_sorted['Feature'], feature_importance_sorted['Absolute Coefficient'])
    plt.title('Feature Importance Based on Absolute Weight Coefficient')
    plt.xlabel('Feature Name')
    plt.ylabel('Absolute Coefficient')
    plt.xticks(rotation=90)
    plt.show()
    
def generate_confusion_matrix(model, x, y):
    """Generate and display the confusion matrix of the Logistic Regression model."""
    predictions = model.predict(x)
    cm = confusion_matrix(y, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix of Breast Cancer Cell Diagnosis') 
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Actual Diagnosis')
    plt.show()     
    
def main():
    df, cancer_data = load_data()
    visualize_data(df)
    x = df[cancer_data.feature_names].values
    y = df['target'].values
    
    model = train_model(x, y)
    evaluate_model(model, x, y)
    feature_significance(model, cancer_data.feature_names)
    generate_confusion_matrix(model, x, y)

if __name__ == "__main__":
    main()        