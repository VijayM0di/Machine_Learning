# Machine Learning Model Comparison

This project implements and compares several machine learning models for a given task. The models include tree-based methods, time-series models, and neural networks.

## Models and Tuning

The following models have been implemented and tuned as part of this project:

| Model      | Tuning Method(s)                  | GPU Support Used? |
|------------|-----------------------------------|-------------------|
| XGBoost    | GridSearchCV, RandomizedSearchCV | Yes               |
| LightGBM   | GridSearchCV                      | Yes               |
| SARIMAX    | Manual grid search                | No                |
| Prophet    | Manual grid search (ParameterGrid)| No                |
| TabPFN     | Feature selection, scaling, device| Yes               |
| Keras      | Keras Tuner (Random, Hyperband, BO)| Yes               |

## Tech Stack

This project leverages a number of powerful libraries and frameworks from the Python ecosystem:

*   **Core Language:** Python 3.9+
*   **Data Manipulation & Analysis:** Pandas, NumPy
*   **Machine Learning & Modeling:**
    *   Scikit-learn (for `GridSearchCV`, `RandomizedSearchCV`, preprocessing, and metrics)
    *   XGBoost
    *   LightGBM
    *   Statsmodels (for `SARIMAX`)
    *   Prophet
    *   TabPFN
    *   TensorFlow & Keras (for deep learning models)
*   **Hyperparameter Tuning:** Keras Tuner
*   **Environment Management:** Conda

## Process Overview

The project follows a standard machine learning workflow to ensure robust and comparable results:

1.  **Data Loading and Preprocessing:** The dataset is loaded and cleaned. This includes handling missing values, converting data types, and preparing the data for modeling.
2.  **Exploratory Data Analysis (EDA):** Initial analysis is performed in Jupyter notebooks to understand data distributions, identify correlations, and uncover underlying patterns.
3.  **Feature Engineering:** Relevant features are selected and engineered to improve model performance. This includes techniques like scaling, which was applied for the TabPFN model.
4.  **Model Training and Hyperparameter Tuning:** Each of the models listed in the table above is trained on the preprocessed data. Hyperparameter tuning is performed using the specified methods (e.g., `GridSearchCV`, `Keras Tuner`) to find the optimal set of parameters for each model.
5.  **Evaluation and Comparison:** The tuned models are evaluated on a hold-out test set using relevant performance metrics. The results are then compiled to compare the effectiveness of each approach.

## Project Structure

A brief description of your project's file structure can go here. For example:

```
├── data/             # Raw and processed data
├── notebooks/        # Jupyter notebooks for exploration and analysis
├── scripts/          # Python scripts for training, evaluation, etc.
├── src/              # Source code for the project
└── requirements.txt  # Project dependencies
```

## Installation

To set up the environment for this project, please follow these steps:

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd ml
    ```

2.  It is recommended to use a virtual environment. If you are using `conda`:
    ```bash
    conda create --name ml_env python=3.9
    conda activate ml_env
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You should create a `requirements.txt` file containing all necessary libraries like pandas, scikit-learn, xgboost, lightgbm, etc.)*

## Usage

To run the model training and evaluation scripts:

```bash
python scripts/train_models.py
```

To run the comparison notebook:

```bash
jupyter notebook notebooks/model_comparison.ipynb
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request