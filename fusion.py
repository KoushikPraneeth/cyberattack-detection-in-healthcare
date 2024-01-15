import asyncio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import pandas as pd
from algorithms.xgb_pipeline import XGBPipeline
import argparse
from algorithms.rf_pipeline import RFPipeline
from algorithms.dt_pipeline import DTPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.model_selection import GridSearchCV


def evaluate(actual, pred):
    # Evaluate the model's performance using various metrics
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)

    print("*****Stacking Prediction Stats*****")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


async def ddos_processing():
    ddos_training_set = pd.read_csv('data/DDOS_train_set.csv')
    ddos_test_set = pd.read_csv('data/DDOS_test_set.csv')

    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])

    print("Pre-processing DDOS data...")
    ddos_training_set.columns = columns
    ddos_test_set.columns = columns

    is_attack = ddos_training_set.attack.map(lambda a: 0 if a == 'normal' else 1)
    test_attack = ddos_test_set.attack.map(lambda a: 0 if a == 'normal' else 1)
    ddos_training_set['attack_flag'] = is_attack
    ddos_test_set['attack_flag'] = test_attack

    features_to_encode = ['protocol_type', 'service', 'flag']
    encoded = pd.get_dummies(ddos_training_set[features_to_encode])

    # get numeric features, we won't worry about encoding these at this point
    numeric_features = ['duration', 'src_bytes', 'dst_bytes']

    # model to fit/test
    print("Splitting features and target for DDOS dataset")
    features = encoded.join(ddos_training_set[numeric_features])
    test_features = encoded.join(ddos_test_set[numeric_features])
    print(f"Len of test features before: {len(test_features)}")
    test_features.dropna(inplace=True)
    print(f"Len of test features after: {len(test_features)}")

    target = ddos_training_set['attack_flag']
    test_target = ddos_test_set['attack_flag']

    start_time = time.time()
    xgb_pipeline = XGBPipeline()
    best_xgb= await xgb_pipeline.predict(features, target)
    print(f"Time taken by XGB: {time.time() - start_time}")

    start_time = time.time()
    rf_pipeline = RFPipeline()
    best_rf = await rf_pipeline.predict(features, target)
    print(f"Time taken by Random Forest: {time.time() - start_time}")

    start_time = time.time()
    dt_pipeline = DTPipeline()
    best_dt = await dt_pipeline.predict(features, target)
    print(f"Time taken by Decision Tree: {time.time() - start_time}")

    base_models = [
        ('decision_tree', best_dt),
        ('xgboost', best_xgb),
        ('random_forest', best_rf),
    ]

    meta_model = LogisticRegression()
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1],
        'solver': ['liblinear', 'sag', 'saga'],
        'l1_ratio': [0.5]
    }
    lr_grid_search = GridSearchCV(meta_model, param_grid, cv=5, scoring='accuracy', verbose=2)
    lr_grid_search.fit(features, target)
    meta_model = lr_grid_search.best_estimator_

    start_time = time.time()
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(features, target)
    stacking_predictions = stacking_model.predict(test_features)
    evaluate(test_target, stacking_predictions)
    print(f"Time taken by Stacking Ensemble: {time.time() - start_time}")


async def malware_processing():
    print("Loading Malware data...")
    malware_train_data = pd.read_csv('data/Malware_train_set.csv')
    malware_test_data = pd.read_csv('data/Malware_test_set.csv')
    malware_train_data.dropna(inplace=True)
    malware_test_data.dropna(inplace=True)

    malware_train_data['classification'] = malware_train_data.classification.map({'benign': 0, 'malware': 1})
    malware_test_data['classification'] = malware_test_data.classification.map({'benign': 0, 'malware': 1})

    malware_train_data = malware_train_data.sample(frac=1).reset_index(drop=True)
    malware_test_data = malware_test_data.sample(frac=1).reset_index(drop=True)

    print("Splitting features and target for Malware dataset")
    features = malware_train_data.drop(
        ["hash", "classification", 'vm_truncate_count', 'shared_vm', 'exec_vm', 'nvcsw', 'maj_flt', 'utime'],
        axis=1)
    test_features = malware_test_data.drop(
        ["hash", "classification", 'vm_truncate_count', 'shared_vm', 'exec_vm', 'nvcsw', 'maj_flt', 'utime'],
        axis=1)

    target = malware_train_data["classification"]
    test_target = malware_test_data['classification']

    start_time = time.time()
    xgb_pipeline = XGBPipeline()
    best_xgb = await xgb_pipeline.predict(features, target)
    print(f"Time taken by XGB: {time.time() - start_time}")

    start_time = time.time()
    rf_pipeline = RFPipeline()
    best_rf = await rf_pipeline.predict(features, target)
    print(f"Time taken by Random Forest: {time.time() - start_time}")

    start_time = time.time()
    dt_pipeline = DTPipeline()
    best_dt = await dt_pipeline.predict(features, target)
    print(f"Time taken by Decision Tree: {time.time() - start_time}")

    base_models = [
            ('random_forest', best_rf),
            ('xgboost', best_xgb),
            ('decision_tree', best_dt),
    ]

    meta_model = LogisticRegression()

    start_time = time.time()
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(features, target)
    stacking_predictions = stacking_model.predict(test_features)
    evaluate(test_target, stacking_predictions)
    print(f"Time taken by Stacking Ensemble: {time.time() - start_time}")


async def phishing_preocessing():
    print("Loading Phishing data...")

    phishing_train_data = pd.read_csv('data/Phishing_train_set.csv')
    phishing_test_data = pd.read_csv('data/Phishing_test_set.csv')
    phishing_train_data.dropna(inplace=True)
    phishing_test_data.dropna(inplace=True)

    phishing_train_data.drop(columns='id', inplace=True)
    phishing_test_data.drop(columns='id', inplace=True)

    phishing_train_data = phishing_train_data.sample(frac=1).reset_index(drop=True)
    phishing_test_data = phishing_test_data.sample(frac=1).reset_index(drop=True)

    print("Splitting features and target for Phishing dataset")
    target = phishing_train_data['CLASS_LABEL']
    test_target = phishing_test_data['CLASS_LABEL']
    features = phishing_train_data.drop(['CLASS_LABEL', 'RelativeFormAction', 'DoubleSlashInPath', 'HttpsInHostname',
                                   'DomainInSubdomains', 'FakeLinkInStatusBar', 'RandomString', 'EmbeddedBrandName',
                                   'AtSymbol', 'ImagesOnlyInForm', 'NumHash', 'AbnormalFormAction', 'UrlLengthRT',
                                   'PctExtResourceUrlsRT', 'PopUpWindow', 'RightClickDisabled', 'IpAddress',
                                   'SubdomainLevelRT', 'TildeSymbol'], axis=1)
    test_features = phishing_test_data.drop(['CLASS_LABEL', 'RelativeFormAction', 'DoubleSlashInPath', 'HttpsInHostname',
                                   'DomainInSubdomains', 'FakeLinkInStatusBar', 'RandomString', 'EmbeddedBrandName',
                                   'AtSymbol', 'ImagesOnlyInForm', 'NumHash', 'AbnormalFormAction', 'UrlLengthRT',
                                   'PctExtResourceUrlsRT', 'PopUpWindow', 'RightClickDisabled', 'IpAddress',
                                   'SubdomainLevelRT', 'TildeSymbol'], axis=1)

    start_time = time.time()
    xgb_pipeline = XGBPipeline()
    best_xgb = await xgb_pipeline.predict(features, target)
    print(f"Time taken by XGB: {time.time() - start_time}")

    start_time = time.time()
    rf_pipeline = RFPipeline()
    best_rf = await rf_pipeline.predict(features, target)
    print(f"Time taken by Random Forest: {time.time() - start_time}")

    start_time = time.time()
    dt_pipeline = DTPipeline()
    best_dt = await dt_pipeline.predict(features, target)
    print(f"Time taken by Decision Tree: {time.time() - start_time}")

    base_models = [
            ('xgboost', best_xgb),
            ('random_forest', best_rf),
            ('decision_tree', best_dt),
    ]

    meta_model = LogisticRegression()

    start_time = time.time()
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(features, target)
    stacking_predictions = stacking_model.predict(test_features)
    evaluate(test_target, stacking_predictions)
    print(f"Time taken by Stacking Ensemble: {time.time() - start_time}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Process datasets with different pipelines.")
    parser.add_argument("--source", choices=["ddos", "malware", "phishing"], help="Specify the dataset to process.", required=True)
    args = parser.parse_args()

    dataset = args.source.lower()

    if dataset == "ddos":
        print("DDoS Pipeline Started")
        print("*" * 50)
        await ddos_processing()
        print("DDoS Pipeline Ended")
        print("*" * 50)
    elif dataset == "malware":
        print("Malware Pipeline Started")
        print("*" * 50)
        await malware_processing()
        print("Malware Pipeline Ended")
        print("*" * 50)
    elif dataset == "phishing":
        print("Phishing Pipeline Started")
        print("*" * 50)
        await phishing_preocessing()
        print("Phishing Pipeline Ended")
        print("*" * 50)
    else:
        print("Invalid dataset. Please choose 'ddos', 'malware', or 'phishing.'")

if __name__=="__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())