from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib
import numpy as np
import cv2
import os
import muller_utils
import yaml
import time
import shutil
import mat73

def calc_roi_mask(pbp_flat, mask, labels, pft, centers, stds):
    # pbp: (N, C), mask: (H, W)
    h, w = mask.shape
    roi_mask_flat = np.zeros(pbp_flat.shape[0], dtype=np.uint8)
    for i in pft:
        center = centers[i]
        dist = np.linalg.norm(pbp_flat - center, axis=1)
        roi_mask_flat |= (dist < stds[i]) & (labels == i)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[mask] = roi_mask_flat * 255
    return roi_mask



if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))
    cluster: MiniBatchKMeans = joblib.load(os.path.join(cfg["checkpoint_dir"], "kmeans_model.pkl"))
    scaler: StandardScaler = joblib.load(os.path.join(cfg["checkpoint_dir"], "scaler.pkl"))
    stds = joblib.load(os.path.join(cfg["checkpoint_dir"], "std.pkl"))
    pft = joblib.load(os.path.join(cfg["checkpoint_dir"], "pft.pkl"))

    k_std = cfg["k_std"]
    k_std = np.sqrt(k_std) * np.sqrt( np.sum(( stds )**2, axis=1)) 

    data_list, mask_list = muller_utils.load_data_and_mask(cfg["pred_dir"], cfg["pred_dir"])
    save_path = os.path.join(cfg["output_dir"], time.strftime("%Y%m%d_%H%M%S"))

    os.makedirs(save_path, exist_ok=True)
    for idx, (data, mask) in enumerate(zip(data_list, mask_list)):
        feature_flatten = muller_utils.data_mask_to_feature([data], [mask], type=cfg["feature_type"])
        feature_flatten = scaler.transform(feature_flatten)
        labels = cluster.predict(feature_flatten)
        roi_mask = calc_roi_mask(feature_flatten, mask, labels, pft, cluster.cluster_centers_, k_std)
        img_path = os.path.join(save_path, f"pred_mask_{idx}.png")
        cv2.imwrite(img_path, roi_mask)
        print(f"Saved: {img_path}")

    # copy config.yaml to save path
    config_path = os.path.join(save_path, "config.yaml")
    shutil.copy("config.yaml", config_path)   
    shutil.copytree("ckpt", os.path.join(save_path, "ckpt"))

    

