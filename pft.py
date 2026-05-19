#偏振特征模板计算
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib
import numpy as np
import os
import muller_utils
import yaml

def get_pft(cluster: MiniBatchKMeans, scaler: StandardScaler, data_list, mask_list, feature_type="pbp"):

    feature_all = muller_utils.data_mask_to_feature(data_list, mask_list, type=feature_type)

    feature_all = scaler.transform(feature_all)
    labels = cluster.predict(feature_all)

    # count the number of points in each cluster
    cluster_counts = np.bincount(labels, minlength=cluster.n_clusters)
    # return the idx that numbers are more than average numbers
    avg_count = np.mean(cluster_counts)
    
    pft = np.where(cluster_counts > avg_count)[0]
    
    return pft

import students_submission
import types
if hasattr(students_submission, "get_pft") and isinstance(getattr(students_submission, "get_pft"), types.FunctionType):
    get_pft = students_submission.get_pft

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))
    cluster: MiniBatchKMeans = joblib.load(os.path.join(cfg["checkpoint_dir"], "kmeans_model.pkl"))
    scaler: StandardScaler = joblib.load(os.path.join(cfg["checkpoint_dir"], "scaler.pkl"))

    data_list, mask_list = muller_utils.load_data_and_mask(cfg["pft_dir"], cfg["pft_dir"])

    pft = get_pft(cluster, scaler, data_list, mask_list, feature_type=cfg["feature_type"])

    print("聚类索引数量：", len(pft))
    joblib.dump(pft, os.path.join(cfg["checkpoint_dir"], "pft.pkl"))

