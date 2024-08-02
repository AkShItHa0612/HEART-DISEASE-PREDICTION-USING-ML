---

# Heart Disease Prediction

## Overview

The **Heart Disease Prediction** project involves building a machine learning model to predict heart disease based on a dataset. This project uses various algorithms including Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests. The dataset is analyzed and pre-processed, followed by training and evaluating different models to determine their accuracy.

## Project Components

1. **Data Collection and Pre-processing:**
   - The dataset `heart_disease_data.csv` is loaded and analyzed. It contains 606 rows and 14 columns of patient data.
   - Key steps include loading the data into a pandas DataFrame, checking for missing values, and performing exploratory data analysis (EDA) using Seaborn and Matplotlib.

2. **Data Visualization:**
   - Visualizations include a count plot of the target variable and a heatmap to show correlations between features.

3. **Model Training:**
   - The dataset is split into training and testing sets.
   - Several machine learning models are trained, including:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree Classifier**
     - **Random Forest Classifier**

4. **Model Evaluation:**
   - Accuracy of each model is calculated for both training and testing datasets.
   - The model performance is compared to choose the best-performing algorithm.

5. **Predictive System:**
   - A function to predict heart disease based on user input data is implemented.

## Getting Started

To run this project, follow these steps:

1. **Clone the Repository:**
   - Use `git clone` to clone the repository to your local machine.

2. **Install Dependencies:**
   - Ensure you have Python installed.
   - Install required libraries using pip:
     ```bash
     pip install numpy pandas scikit-learn seaborn matplotlib
     ```

3. **Run the Code:**
   - Open the Jupyter Notebook or a Python environment.
   - Run the provided code cells in sequence.

## Data Analysis and Visualization

- **Data Overview:**
  ```python
  heart_data.head()
  heart_data.tail()
  heart_data.sample(5)
  heart_data.info()
  heart_data.describe()
  ```
  
- **Visualizations:**
  - Count plot of heart disease distribution:
    ```python
    sns.countplot(x='target', data=heart_data)
    plt.xlabel("Heart Condition")
    plt.ylabel("Count")
    plt.title("Distribution of Heart Disease in the Dataset")
    plt.show()
    ```
  - Heatmap of correlations:
    ```python
    sns.heatmap(heart_data.corr())
    plt.show()
    ```

## Model Training and Evaluation

- **Logistic Regression:**
  ```python
  model = LogisticRegression()
  model.fit(X_train, Y_train)
  ```
  - Accuracy on training and test data.

- **K-Nearest Neighbors (KNN):**
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  classifier = KNeighborsClassifier(n_neighbors=5)
  classifier.fit(X_train, Y_train)
  ```

- **Decision Tree Classifier:**
  ```python
  from sklearn.tree import DecisionTreeClassifier
  classifier = DecisionTreeClassifier(criterion='entropy')
  classifier.fit(X_train, Y_train)
  ```

- **Random Forest Classifier:**
  ```python
  from sklearn.ensemble import RandomForestClassifier
  classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
  classifier.fit(X_train, Y_train)
  ```

- **Accuracy Scores:**
  - The accuracy scores for each model are printed to assess their performance.

## Predictive System

- **Prediction Example:**
  ```python
  input_data = (44, 0, 2, 118, 242, 0, 1, 149, 0, 0.3, 1, 1, 2)
  input_data_as_numpy_array = np.asarray(input_data)
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
  prediction = model.predict(input_data_reshaped)
  ```

