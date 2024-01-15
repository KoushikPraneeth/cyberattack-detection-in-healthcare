
# Cyber Detection Project

This project is designed for cyber detection, implementing various machine learning algorithms on different cyberattack datasets. The main file (`main.py`) orchestrates the entire pipeline for malware, DDOS, and phishing datasets present in the `data` folder. The implemented algorithms include Decision Tree, Logistic Regression, k-Nearest Neighbors, Neural Networks, XGBoost, and Random Forests.

## Project Structure

│ main.py

│ requirements.txt

│
├── algorithms

│ ├── dt_pipeline.py

│ ├── knn_pipeline.py

│ ├── lr_pipeline.py

│ ├── nn_pipeline.py

│ ├── rf_pipeline.py

│ ├── xgb_pipeline.py

│ └── init.py
│

└── data

├── DDOS_dataset.txt

├── Malware dataset.csv

└── phishing_cyberattacks.csv

## Algorithms

Each algorithm is implemented in a pipeline structure, including data ingestion, preprocessing, splitting, normalization, model training, and metrics logging. The following algorithms are implemented:

- Decision Tree (`algorithms/dt_pipeline.py`)
- k-Nearest Neighbors (`algorithms/knn_pipeline.py`)
- Logistic Regression (`algorithms/lr_pipeline.py`)
- Neural Networks (`algorithms/nn_pipeline.py`)
- Random Forests (`algorithms/rf_pipeline.py`)
- XGBoost (`algorithms/xgb_pipeline.py`)

## Usage

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the main file `main.py` to execute the entire pipeline for malware, DDOS, and phishing datasets.

```bash
python main.py
