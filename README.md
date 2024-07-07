# Customer Churn Prediction Project

## Overview
This project aims to develop a machine learning model to predict customer churn in a telecommunications company. The model is designed to help the company identify potential churners and implement proactive measures to retain customers.

## Table of Contents
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
- [Model Selection and Tuning](#model-selection-and-tuning)
- [Documentation](#documentation)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

## Project Structure
The project is organized into the following main components:

 1. `data/`
   - Contains the dataset used in the project.
  
 2. `notebooks/`
   - Includes the Jupyter notebook used for data exploration, preprocessing, modeling, and evaluation.
  
 3. `models/`
   - Stores trained machine learning models.
   
 4. `encoders/`
   - Saves encoders or transformers used during preprocessing.
   
 5. `README.md`
   - The main documentation file providing an overview of the project.

 6. `requirements.txt`
   - Lists all project dependencies for easy installation.

 7. `images/`
   - Contains images, charts, or diagrams used in documentation.


## Dependencies
To run this project, you need to have the following dependencies installed. You can install them using `pip`:

pandas: Data manipulation and analysis.

scikit-learn: Machine learning tools and utilities.

matplotlib: Data visualization.

seaborn: Statistical data visualization.

joblib: Joblib is used for parallelizing code, particularly during model training.

numpy: Mathematical functions for numerical operations.

jupyter: Jupyter notebooks for interactive data exploration.

## Installation

Follow these steps to set up and run the project on your local machine.

### Clone the Repository

git clone https://github.com/aaserwaa/Customer-Churn-Analysis.git
cd your-project

## Usage

Run the cells in the notebook to execute the machine learning pipeline. This includes data preprocessing, model training, and evaluation.

### Interpret Results

Review the results and metrics presented in the notebook. This includes insights into the model's performance, strengths, weaknesses, and implications on business objectives.

### Model Persistence (Optional)

If you want to save the trained model for later use, follow the instructions in the "Model Persistence" section.


## Data Exploration and Preprocessing

### Data Overview

The dataset used in this project consists of 3 samples with 21 features. The target variable is Churn.

### Data Exploration

During the exploratory data analysis (EDA), key insights were uncovered:

Few missing values, skewed datasets among others as seen in the notebook

### Data Preprocessing

#### Handling Missing Values

We addressed missing values using imputation method.

#### Encoding Categorical Variables

Categorical variables were encoded using One-hot encoding.

#### Scaling Numerical Features

Numerical features were scaled using StandardScaler.

#### Handling Imbalanced Classes

To address imbalanced classes, we applied SMOTE.

#### Feature Engineering

We performed feature engineering by doing log transformations.

### Data Splitting

The dataset was split into training(80%) and testing sets(20%). Special attention was given to maintaining a representative distribution of classes in both sets.

## Model Selection and Tuning

### Model Selection

We considered multiple machine learning models for predicting customer churn. After initial evaluation, the following models were shortlisted:

1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)


### Hyperparameter Tuning

To optimize the performance of the models, a grid search was conducted for each model. Key hyperparameters were tuned to achieve better accuracy and generalization.

### Cross-Validation

Cross-validation was employed to assess the model's performance robustly. A 5-fold cross-validation strategy was used, and the average performance metrics were considered.

### Grid Search Results

The grid search results for each model are summarized below:

#### Decision Tree
Performing hyperparameter tuning for Decision Tree
Best parameters: 
{'classifier__criterion': 'gini', 'classifier__max_depth': 20, 'classifier__min_samples_split': 10}

Best accuracy: 0.880518170720833

#### Random Forest

Performing hyperparameter tuning for Random Forest
Best parameters: 
{'classifier__criterion': 'entropy', 'classifier__max_depth': 20, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 100}

Best accuracy: 0.8951423801046058

#### Support Vector Machine (SVM)

Performing hyperparameter tuning for SVM
Best parameters:
{'classifier__C': 10, 'classifier__kernel': 'rbf'}

Best accuracy: 0.8691102358088807

### Model Comparison

Random Forest seems to perform relatively well, showing a good balance between precision and recall, making it a more competitive model for predicting customer churn.

SVM also exhibits a balanced performance, but with a slightly lower precision for churn prediction.
Decision Tree, KNN, SGD, and Logistic Regression have their strengths and weaknesses

Based on these insights, Random Forest model seems to be the best model for predicting customer churn


## Documentation
 
- **Data Processing:** Covers exploration, preprocessing, and handling missing values.

- **Modeling:** Highlights model selection, tuning, and evaluation metrics.

- **Results:** Summarizes key outcomes and insights from the models.

### Code Comments

Comments were used in the notebook to explain complex or critical code sections.

### Future Work

-  Areas for improvement and enhancements were recommended .

## Conclusion

The project successfully addresses the challenge of predicting customer churn in a telecommunications company. Key insights and recommendations include:

- Utilizing a Random Forest model for its high accuracy and balanced recall.
- Identifying areas for improvement in precision for churned customers.
- Considering imbalances in the dataset and exploring methods to address them.

The documented process and results serve as a valuable reference for future analysis and improvements.

## Contributing

Contributions to this project are welcome. To contribute, please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the [MIT License](LICENSE).

Here is a link to my [published article on LinkedIn](https://www.linkedin.com/pulse/customer-churn-analysis-my-journey-predictive-analytics-serwaa-akoto-wnvke)


