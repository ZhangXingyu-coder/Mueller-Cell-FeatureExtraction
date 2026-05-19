from scipy.ndimage import gaussian_filter
import numpy as np
import scipy.io as scio
import h5py
import muller_features as mf
import os
import cv2

def mm2mm44(data):
    temp=[]
    for para in mf.parameters:
        temp.append(data[para])

    MM=np.stack(temp,axis=2)
    MM[:,:,0]=1.

    MM_=np.zeros(MM.shape)
    for i in range(1,16):
        MM_[:,:,i]=gaussian_filter(MM[:,:,i],3)
    MM_[:,:,0]=1.

    MM=MM_.copy()
    MM_=0.

    mf.clear_all_MMD()

    MM_44 = MM.reshape(MM.shape[0],MM.shape[1],4,4)
    return MM_44

def load_mat_file(path):
    try:
        # 先尝试用 scipy.io.loadmat 读取（适用于v7.2及以下）
        return scio.loadmat(path)
    except NotImplementedError:
        # 如果是 v7.3 (HDF5) 格式，使用 h5py
        mat = {}
        with h5py.File(path, 'r') as f:
            for key in mf.parameters:
                if key in f.keys():
                    arr = f[key][()]
                    # h5py 读出来是列主序，需转置
                    arr = np.array(arr, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr.T
                    mat[key] = arr
        return mat

def mm2pbp(data):
    is_img = (data.ndim == 4)
    if not is_img:
        data = data[None, ...]
    MMpbp=np.zeros((data.shape[0],data.shape[1],len(mf.function_list))).astype('float32')
    for i, f in enumerate(mf.function_list):
        MMpbp[:,:,i]=f(data)
        
    for i in range(MMpbp.shape[2]):
        if np.isnan(MMpbp[:,:,i]).sum()>0:
            nanmask=np.isnan(MMpbp[:,:,i])
            temp_pbp=MMpbp[:,:,i].copy()
            temp_pbp[nanmask]=0.
            MMpbp[:,:,i][nanmask]=gaussian_filter(temp_pbp,20)[nanmask]
    MMpbp=np.real(MMpbp)
    if not is_img:
        MMpbp = MMpbp[0]
    return MMpbp

def save_file(file_path, data):
    if not isinstance(data, dict):
        tmp = {}
        for i, f_name in enumerate(mf.function_list_names):
            tmp[f_name] = data[...,i].astype('float32')
        data = tmp
    if file_path.endswith('.mat'):
        scio.savemat(file_path, data)
    elif file_path.endswith('.exr'):
        import pyexr
        pyexr.write(file_path, data)

def load_data_and_mask(data_dir, mask_dir, type='walk', mask_name='mask.png'):
    data_list = []
    mask_list = []
    data_path_list = []
    mask_path_list = []
    if type == 'walk':
        for root, dirs, files in os.walk(data_dir):
            if "FinalMM.mat" in files:
                data_path_list.append(os.path.join(root, "FinalMM.mat"))
                mask_path_list.append(os.path.join(mask_dir, mask_name))

    elif type == 'listdir':
        for fname in os.listdir(data_dir):
            if not fname.endswith('.mat'):
                continue
            mask_name = os.path.splitext(fname)[0] + ".png"  # 假设mask和数据同名但为png
            data_path_list.append(os.path.join(data_dir, fname))
            mask_path_list.append(os.path.join(mask_dir, mask_name))

    for data_path, mask_path in zip(data_path_list, mask_path_list):
        if not os.path.exists(data_path) or not os.path.exists(mask_path):
            print(f"文件不存在: {data_path} 或 {mask_path}")
            continue 
        data = load_mat_file(data_path)
        data= mm2mm44(data)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0)  # 二值化，非0为mask
        data_list.append(data)
        mask_list.append(mask)

    return data_list, mask_list

def data_mask_to_feature(data_list, mask_list, type="pbp"):
    feature_all = []
    for data, mask in zip(data_list, mask_list):
        # 确保mask和特征 shape一致
        if mask.shape != data.shape[:2]:
            print("mask和特征 shape不一致，跳过")
            continue
        if type == "pbp":
            feature = mm2pbp(data[mask])
        elif type == "muller":
            feature = data[mask].reshape(-1, 16)
        else:
            raise ValueError("Unsupported feature type. Use 'pbp' or 'muller'.")
        feature_all.append(feature)
    feature_all = np.concatenate(feature_all, axis=0)
    return feature_all

if __name__ == "__main__":
    # test mm2pbp function
    # with h5py.File('ROI1/m11.mat', 'r') as f:
    #     print(f.keys())
    for root, dirs, files in os.walk('data/贴壁细胞样本'):
        for file in files:
            if file.endswith('.mat'):
                print(f"Processing {file}...")
                mat_path = os.path.join(root, file)
                data = load_mat_file(mat_path)
                save_file(mat_path, data)

    # MM = mm2mm44(MM)  # 转换为MM44
    # MMpbp = mm2pbp(MM)  # 转换为PBP
    
    # MM = mm2mm44(MM)
    # MMpbp = mm2pbp(MM)
    # save_file('exr/FinalMMpbp.exr', MMpbp)  # 保存处理后的数据
