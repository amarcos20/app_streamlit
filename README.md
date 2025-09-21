
# Student Intervention System

## Project Description

This notebook features a complete Machine Learning pipeline, including data visualization of a student dataset, allowing for the identification of factors leading to academic failure. Additionally, it provides comprehensive data treatment for the use of ML models, including Logistic Regression, SVM (with and without GridSearchCV), AdaBoostClassifier, KNN, Random Forest (with and without GridSearchCV), and Gradient Boosting.

The `ADABOOST CLASSIFIER` was selected as the best model after comparing evaluation metrics such as accuracy, precision, recall, and ROC AUC from the scikit-learn library.

This interactive web application, developed with Streamlit, aims to analyze and visualize student performance data and utilize a Machine Learning model to predict academic success. The application is based on the `student-data.csv` dataset and a previously trained and saved Machine Learning pipeline (preprocessing and model).

The tool allows for data exploration, individual student predictions, and analysis of evaluation metrics and the interpretability of the trained model.

Our app: `https://amarcos.streamlit.app`

## Features

*   **Home:** Overview of the application and its capabilities.
*   **Data Exploration:** Analyze the characteristics of the original dataset through statistical summaries, distributions, and visualizations of relationships between variables, including correlation with the target variable.
*   **Individual Prediction:** Enter a student's data to get an instant prediction of their probability of passing or failing the final exam, using the trained model. Allows selection of different models saved in the `artefacts` folder.
*   **Trained Model Analysis:** View the performance metrics (Accuracy, Precision, Recall, F1-Score, AUC ROC) and the confusion matrix of the main saved model (`best_model.joblib`) on the test set. Includes, where applicable, visualizations of feature importance or coefficients to understand model decisions.
*   **Model Evaluation and Comparison:** Allows selection of different types of Machine Learning algorithms, temporarily training them on processed training data, and evaluating their performance on the test set. Useful for comparing approaches.
*   **Documentation:** Detailed description of the dataset, features, model artifacts, and application sections.

## Prerequisites

To run this application, you need to have Python installed on your system. Using a virtual environment is recommended.

The necessary Python libraries are listed below. You can install them manually or via a `requirements.txt` file.

## Installation

1.  Clone or download the project files to your computer.
2.  Navigate to the project directory in your terminal.
3.  Install the necessary Python libraries. If you have a `requirements.txt` file provided with the project, run:
    ```bash
    pip install -r requirements.txt
    ```
    Otherwise, manually install the dependencies used in the code:
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly streamlit-option-menu joblib
    ```

## Data and Artifacts Configuration

This application expects the following files to exist in the correct directory:

*   `student-data.csv`: The original dataset (or an initial processed version) used for EDA. It should be in the project's root folder.
*   `artefacts/`: A subfolder in the project root containing the Machine Learning pipeline artifacts. At least the following are required:
    *   `artefacts/preprocessor.joblib`: The trained preprocessor object (e.g., StandardScaler, OneHotEncoder, ColumnTransformer).
    *   `artefacts/best_model.joblib`: The trained Machine Learning model (e.g., LogisticRegression, RandomForestClassifier). This is the main model analyzed and used by default.
    *   `artefacts/original_input_columns.joblib`: A list (or similar) containing the names of the original input columns that the preprocessor expects.
    *   `artefacts/processed_feature_names.joblib`: A list (or similar) containing the names of the features after preprocessing.
    *   *(Optional)* Other `.joblib` files of trained models that you want to make available in the "Individual Prediction" section.
*   `data/processed/`: A subfolder in the project root containing the processed data. The following are required:
    *   `data/processed/train_processed.csv`: Processed training data (DataFrame).
    *   `data/processed/test_processed.csv`: Processed testing data (DataFrame).
    *   Both processed CSV files must contain the target column (`passed_mapped`, by default) and the processed features.

**Note:** These files (`*.joblib` and `*_processed.csv`) must be generated from your model training script or notebook before running the Streamlit application. Ensure that the paths in the code (`artefacts/` and `data/processed/`) match your project's folder structure.

## How to Run

1.  Make sure you are in the project's root directory in your terminal.
2.  Execute the following command in the `bash`:
    ```bash
    streamlit run app.py
    ```

3.  The application will open in your default browser.

## Expected Folder Structure

```
.
├── artefacts/
│   ├── preprocessor.joblib
│   ├── best_model.joblib
│   ├── original_input_columns.joblib
│   ├── processed_feature_names.joblib
│   └── (outros_modelos).joblib  
├── data/
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
├── student-data.csv             # Dataset original
└── sua_aplicacao.py             # Este ficheiro Streamlit
```


## Academic Project

This application was developed as part of an academic project for the "Elements of Artificial Intelligence and Data Science" course by students:

Afonso Marcos (202404088)

Pedro Afonso (202404125)

Afonso Silva (202406661)

© 2025 Student Intervention System.
