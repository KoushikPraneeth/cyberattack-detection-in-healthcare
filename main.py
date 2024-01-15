import pandas as pd
import asyncio

# Importing pipeline modules from algorithms directory
from algorithms.nn_pipeline import NNPipeline
from algorithms.xgb_pipeline import XGBPipeline
from algorithms.dt_pipeline import DTPipeline
from algorithms.lr_pipeline import LRPipeline
from algorithms.rf_pipeline import RFPipeline
from algorithms.knn_pipeline import KNNPipeline
import tracemalloc

# Starting memory tracing for performance analysis
tracemalloc.start()


# Function to process DDOS dataset
async def ddos_processing():
    print("Loading DDOS data...")
    ddos_data = pd.read_csv('data/DDOS_dataset.txt')
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])

    print("Pre-processing DDOS data...")
    ddos_data.columns = columns
    is_attack = ddos_data.attack.map(lambda a: 0 if a == 'normal' else 1)
    ddos_data['attack_flag'] = is_attack

    features_to_encode = ['protocol_type', 'service', 'flag']
    encoded = pd.get_dummies(ddos_data[features_to_encode])

    # get numeric features, we won't worry about encoding these at this point
    numeric_features = ['duration', 'src_bytes', 'dst_bytes']

    # model to fit/test
    print("Splitting features and target for DDOS dataset")
    features = encoded.join(ddos_data[numeric_features])
    target = ddos_data['attack_flag']

    print("Initiating XGB Pipeline for DDOS...")
    print("=" * 50)
    xgb_pipeline = XGBPipeline()
    await xgb_pipeline.predict(features, target)
    print("XGB Pipeline for DDOS Ended.")
    print("=" * 50)

    print("Initiating Random Forest Pipeline for DDOS...")
    print("=" * 50)
    rf_pipeline = RFPipeline()
    await rf_pipeline.predict(features, target)
    print("Random Forest Pipeline for DDOS Ended.")
    print("=" * 50)

    print("Initiating Neural Network Pipeline for DDOS...")
    print("=" * 50)
    nn_pipeline = NNPipeline()
    await nn_pipeline.predict(features, target)
    print("Neural Network Pipeline for DDOS Ended.")
    print("=" * 50)


# Function to process Malware dataset
async def malware_processing():
    print("Loading Malware data...")
    malware_data = pd.read_csv('data/Malware dataset.csv')
    malware_data['classification'] = malware_data.classification.map({'benign': 0, 'malware': 1})
    malware_data = malware_data.sample(frac=1).reset_index(drop=True)

    print("Splitting features and target for Malware dataset")
    features = malware_data.drop(
        ["hash", "classification", 'vm_truncate_count', 'shared_vm', 'exec_vm', 'nvcsw', 'maj_flt', 'utime'],
        axis=1)
    target = malware_data["classification"]

    print("Initiating XGB Pipeline for Malware...")
    print("=" * 50)
    xgb_pipeline = XGBPipeline()
    await xgb_pipeline.predict(features, target)
    print("XGB Pipeline for Malware Ended.")
    print("=" * 50)

    print("Initiating Random Forest Pipeline for Malware...")
    print("=" * 50)
    rf_pipeline = RFPipeline()
    await rf_pipeline.predict(features, target)
    print("Random Forest Pipeline for Malware Ended.")
    print("=" * 50)

    print("Initiating Neural Network Pipeline for Malware...")
    print("=" * 50)
    nn_pipeline = NNPipeline()
    await nn_pipeline.predict(features, target)
    print("Neural Network Pipeline for Malware Ended.")
    print("=" * 50)


# Function to process Phishing dataset
async def phishing_preocessing():
    print("Loading Phishing data...")

    phishing_data = pd.read_csv('data/phishing_cyberattacks.csv')
    phishing_data.drop(columns='id', inplace=True)
    phishing_data = phishing_data.sample(frac=1).reset_index(drop=True)

    print("Splitting features and target for Phishing dataset")
    target = phishing_data['CLASS_LABEL']
    features = phishing_data.drop(['CLASS_LABEL', 'RelativeFormAction', 'DoubleSlashInPath', 'HttpsInHostname',
                                   'DomainInSubdomains', 'FakeLinkInStatusBar', 'RandomString', 'EmbeddedBrandName',
                                   'AtSymbol', 'ImagesOnlyInForm', 'NumHash', 'AbnormalFormAction', 'UrlLengthRT',
                                   'PctExtResourceUrlsRT', 'PopUpWindow', 'RightClickDisabled', 'IpAddress',
                                   'SubdomainLevelRT', 'TildeSymbol'], axis=1)

    print("Initiating Decision Tree Pipeline for Phishing...")
    print("=" * 50)
    dt_pipeline = DTPipeline()
    await dt_pipeline.predict(features, target)
    print("Decision Tree Pipeline for Phishing Ended.")
    print("=" * 50)

    print("Initiating Logistic Regression Pipeline for Phishing...")
    print("=" * 50)
    lr_pipeline = LRPipeline()
    await lr_pipeline.predict(features, target)
    print("Logistic Regression Pipeline for Phishing Ended.")
    print("=" * 50)

    print("Initiating K-Nearest Neighbors Pipeline for Phishing...")
    print("=" * 50)
    knn_pipeline = KNNPipeline()
    await knn_pipeline.predict(features, target)
    print("K-Nearest Neighbors Pipeline for Phishing Ended.")
    print("=" * 50)


# Main function to run all pipelines
async def main():
    print("DDoS Pipeline Started")
    print("*" * 50)
    await ddos_processing()
    print("DDoS Pipeline Ended")
    print("*" * 50)

    print("Malware Pipeline Started")
    print("*" * 50)
    await malware_processing()
    print("Malware Pipeline Ended")
    print("*" * 50)

    print("Phishing Pipeline Started")
    print("*" * 50)
    await phishing_preocessing()
    print("Phishing Pipeline Ended")
    print("*" * 50)

# Run the main loop
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
