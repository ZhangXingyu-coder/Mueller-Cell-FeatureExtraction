import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from joblib import dump  # 推荐直接用joblib的dump/load
import yaml
import muller_utils

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))
    data_dir = cfg["pft_dir"]  # 假设数据目录在配置文件中指定

    data_list, mask_list = muller_utils.load_data_and_mask(data_dir, data_dir)

    feature_all = muller_utils.data_mask_to_feature(data_list, mask_list, type=cfg["feature_type"])

    scaler = StandardScaler()
    cluster = MiniBatchKMeans(n_clusters=cfg["n_clusters"], batch_size=cfg["batch_size"], random_state=cfg["random_state"])
    feature_all = scaler.fit_transform(feature_all)
    cluster.fit(feature_all)

    groups_mean = []
    groups_std = []
    for i in range(cfg["n_clusters"]):
        cluster_points = feature_all[cluster.labels_ == i]
        groups_mean.append(cluster_points.mean(axis=0))  # 质心坐标
        groups_std.append(cluster_points.std(axis=0))    # 方差

    groups_mean = np.array(groups_mean)
    groups_std = np.array(groups_std)
    # 保存聚类模型
    if not os.path.exists(cfg["checkpoint_dir"]):
        os.makedirs(cfg["checkpoint_dir"])
    dump(cluster, os.path.join(cfg["checkpoint_dir"], "kmeans_model.pkl"))
    dump(groups_std, os.path.join(cfg["checkpoint_dir"], "std.pkl"))
    dump(scaler, os.path.join(cfg["checkpoint_dir"], "scaler.pkl"))
    print("聚类模型已保存。")


