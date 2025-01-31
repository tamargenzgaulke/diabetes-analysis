# Diabetes Prediction with Logistic Regression and Random Forest

This repository contains a machine learning project focused on predicting diabetes outcomes using a publicly available dataset. The project explores the use of two classification models: Logistic Regression and Random Forest, and demonstrates their performance in predicting whether a patient has diabetes or not.

## Dataset

The dataset used is the **Pima Indians Diabetes Database**, which can be accessed from the following link:

[Dataset Source](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

### Features
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure
- **SkinThickness**: Skinfold thickness
- **Insulin**: 2-Hour serum insulin
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: 1 if the patient has diabetes, 0 if not (target variable)

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/diabetes-prediction.git
    cd diabetes-prediction
    ```

2. Run the Python script:

    ```bash
    python diabetes_prediction.py
    ```

3. Check the output for model evaluation metrics, including accuracy, confusion matrix, and classification report.

## Project Overview

### Data Exploration

We begin by loading the dataset, checking for missing values, and visualizing the distribution of the target variable and the correlations between features.

```python
import pandas as pd
df = pd.read_csv(dataset_url, names=columns)
```

### Data Preprocessing

The dataset is split into training and testing sets, followed by data normalization using **StandardScaler**.

### Models

- **Logistic Regression**: A linear model used to predict the outcome based on a combination of the input features.
- **Random Forest**: An ensemble learning method that builds multiple decision trees and combines their predictions.

### Evaluation

Both models are evaluated based on accuracy, confusion matrix, and classification report to assess their performance in predicting diabetes.

## Results

### Logistic Regression
- **Accuracy**: X%
- **Confusion Matrix**:  
  ![Confusion Matrix LR](path_to_your_image)
- **Classification Report**:  
  ![Classification Report LR](path_to_your_image)

### Random Forest
- **Accuracy**: X%
- **Confusion Matrix**:  
  ![Confusion Matrix RF](path_to_your_image)
- **Classification Report**:  
  ![Classification Report RF](path_to_your_image)

## Visualizations

Here are some visualizations generated during the analysis:

- **Distribution of Diabetes Cases**:  
  ![Distribution of Diabetes Cases](![image](https://github.com/user-attachments/assets/493289b0-2f55-4681-b2b6-91a81c5563ca))
- **Correlation Heatmap**:  
  ![Correlation Heatmap](![image](https://github.com/user-attachments/assets/9a57498a-980b-4719-9428-ae3ecb3718c2)
)

## Conclusion

This project demonstrates the application of two powerful machine learning algorithms, Logistic Regression and Random Forest, to solve a real-world problem in healthcare. Both models show competitive performance, with Random Forest generally outperforming Logistic Regression.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
