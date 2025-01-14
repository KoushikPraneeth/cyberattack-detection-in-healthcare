# Cyber Detection Project

This project is designed to detect various types of cyberattacks, including **Malware**, **DDoS (Distributed Denial of Service)**, and **Phishing**, using machine learning algorithms. The project implements a pipeline for each type of cyberattack, leveraging multiple machine learning models to classify and predict attacks. The models are trained and evaluated on datasets specific to each type of attack, and the results are logged for performance analysis.

---

## Project Structure

The project is organized as follows:

```
koushikpraneeth-cyberattack-detection-in-healthcare.git/
├── README.md
├── algorithms/
│   ├── __init__.py
│   ├── dt_pipeline.py            # Decision Tree Pipeline
│   ├── knn_pipeline.py           # k-Nearest Neighbors Pipeline
│   ├── lr_pipeline.py            # Logistic Regression Pipeline
│   ├── nn_pipeline.py            # Neural Network Pipeline
│   ├── rf_pipeline.py            # Random Forest Pipeline
│   └── xgb_pipeline.py           # XGBoost Pipeline
├── data/
│   ├── DDOS_dataset.txt          # DDoS Dataset
│   ├── DDOS_test_set.csv         # DDoS Test Set
│   ├── DDOS_train_set.csv        # DDoS Training Set
│   ├── Malware dataset.csv       # Malware Dataset
│   ├── Malware_test_set.csv      # Malware Test Set
│   ├── Malware_train_set.csv     # Malware Training Set
│   ├── Phishing_test_set.csv     # Phishing Test Set
│   ├── Phishing_train_set.csv    # Phishing Training Set
│   └── phishing_cyberattacks.csv # Phishing Dataset
├── fusion.py                     # Stacking Ensemble for Model Fusion
├── main.py                       # Main Script to Run Pipelines
├── requirements.txt              # Python Dependencies
└── tree.txt                      # Directory Structure
```

---

## Key Features

1. **Multiple Machine Learning Models**:
   - The project implements **Decision Tree**, **Logistic Regression**, **k-Nearest Neighbors (KNN)**, **Neural Networks**, **Random Forest**, and **XGBoost** for cyberattack detection.
   - Each model is implemented as a pipeline, including data preprocessing, training, hyperparameter tuning, and evaluation.

2. **Dataset-Specific Pipelines**:
   - Separate pipelines are implemented for **Malware**, **DDoS**, and **Phishing** datasets.
   - Each pipeline handles dataset-specific preprocessing, such as encoding categorical variables, handling missing values, and feature selection.

3. **Hyperparameter Tuning**:
   - **GridSearchCV** is used for hyperparameter tuning across all models to ensure optimal performance.
   - The best hyperparameters are logged, and the best model is used for evaluation.

4. **Model Evaluation**:
   - Each model is evaluated using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
   - Evaluation results are printed for each model and dataset.

5. **Stacking Ensemble**:
   - The `fusion.py` script implements a **Stacking Ensemble** model, combining predictions from **Decision Tree**, **Random Forest**, and **XGBoost**.
   - A **Logistic Regression** meta-model is used to combine the predictions of the base models.

6. **Asynchronous Execution**:
   - The project uses **asyncio** to run pipelines asynchronously, improving efficiency and performance.

---

## Datasets

The project uses the following datasets:

1. **DDoS Dataset**:
   - Contains network traffic features such as `duration`, `protocol_type`, `src_bytes`, `dst_bytes`, and `attack_flag`.
   - The target variable is `attack_flag`, which indicates whether the traffic is normal (`0`) or an attack (`1`).

2. **Malware Dataset**:
   - Contains features related to malware behavior, such as `hash`, `classification`, and various system metrics.
   - The target variable is `classification`, which indicates whether the sample is `benign` (`0`) or `malware` (`1`).

3. **Phishing Dataset**:
   - Contains features related to phishing websites, such as `UrlLengthRT`, `PctExtResourceUrlsRT`, and `CLASS_LABEL`.
   - The target variable is `CLASS_LABEL`, which indicates whether the website is legitimate (`0`) or phishing (`1`).

---

## Algorithms

The following machine learning algorithms are implemented:

1. **Decision Tree (`dt_pipeline.py`)**:
   - A tree-based model that splits the data based on feature values.
   - Hyperparameters tuned: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

2. **Logistic Regression (`lr_pipeline.py`)**:
   - A linear model for binary classification.
   - Hyperparameters tuned: `penalty`, `C`, `solver`, `l1_ratio`.

3. **k-Nearest Neighbors (`knn_pipeline.py`)**:
   - A non-parametric model that classifies data points based on the majority class of their nearest neighbors.
   - Hyperparameters tuned: `n_neighbors`, `weights`, `algorithm`, `leaf_size`, `p`.

4. **Neural Networks (`nn_pipeline.py`)**:
   - A deep learning model with two hidden layers and a sigmoid output layer.
   - Uses **Keras** for model building and training.

5. **Random Forest (`rf_pipeline.py`)**:
   - An ensemble of decision trees that aggregates predictions to improve accuracy.
   - Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

6. **XGBoost (`xgb_pipeline.py`)**:
   - A gradient boosting algorithm that optimizes for speed and performance.
   - Hyperparameters tuned: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`.

---

## Usage

### 1. Install Dependencies
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

### 2. Run the Main Pipeline
To run the entire pipeline for all datasets (DDoS, Malware, and Phishing), execute the following command:
```bash
python main.py
```

### 3. Run the Stacking Ensemble
To run the stacking ensemble for a specific dataset, use the `fusion.py` script:
```bash
python fusion.py --source ddos       # For DDoS dataset
python fusion.py --source malware    # For Malware dataset
python fusion.py --source phishing   # For Phishing dataset
```

---

## Example Output

### Decision Tree Pipeline (DDoS Dataset)
```
*****Decision Tree Best Model Prediction Stats*****
Accuracy: 0.98
Precision: 0.97
Recall: 0.96
F1 Score: 0.96
```

### Stacking Ensemble (Malware Dataset)
```
*****Stacking Prediction Stats*****
Accuracy: 0.99
Precision: 0.98
Recall: 0.99
F1 Score: 0.98
```

---

## Performance Analysis

- **Hyperparameter Tuning**: GridSearchCV is used to find the best hyperparameters for each model, ensuring optimal performance.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score are used to evaluate model performance.
- **Asynchronous Execution**: The use of `asyncio` improves the efficiency of running multiple pipelines.

---

## Future Improvements

1. **Feature Engineering**:
   - Explore additional feature engineering techniques to improve model performance.
   
2. **Model Interpretability**:
   - Use tools like SHAP or LIME to explain model predictions.

3. **Real-Time Detection**:
   - Implement real-time detection capabilities for live network traffic.

4. **Dataset Expansion**:
   - Include more datasets for other types of cyberattacks, such as ransomware or SQL injection.

5. **Deployment**:
   - Deploy the models as a web service using Flask or FastAPI for real-world usage.


---

## Acknowledgments

- The datasets used in this project are publicly available and were preprocessed for machine learning tasks.
- Special thanks to the open-source community for providing libraries like Scikit-learn, XGBoost, and Keras.

---
