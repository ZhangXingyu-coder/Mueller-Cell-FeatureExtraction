# Mueller Matrix Toolkit for cell feature extraction
基于无监督学习的活细胞动态偏振特征提取

A lightweight Python toolkit for Mueller matrix decomposition and polarization parameter extraction, designed for label-free living-cell imaging.

## ✨ Features
- Lu-Chipman decomposition of 4x4 Mueller matrix
- Extraction of polarization parameters (DoP, depolarization, retardance) of living cells
- Visualization utilities for cell dynamics and structural changes
- Example workflow on experimental data of apoptotic cells

## 👤 Author
[张星宇] 
MSc student, Biomedical Engineering, Tsinghua University
📧 xingyu-z22@mais.tsinghua.edu.cn

[魏金福] 
MSc student, Artificial Intelligence, Tsinghua University



### 🔧配置环境（建议使用虚拟环境）
0. 下载代码
    ```
    git clone https://github.com/ZhangXingyu-coder/Mueller-Cell-FeatureExtraction.git
    cd Mueller-Cell-FeatureExtraction
    ```
1. 创建虚拟环境
    ```
    python -m venv .mueller
    ```
2. 激活环境
    ```
    source .mueller/bin/activate
    ```
3. 安装依赖库
    ```
    pip install -r requirement.txt
    ```
---- 
### 数据准备 (具体路径名称可从config.py中修改)
1. 超像素计算数据。

    将采集的缪勒矩阵文件存储到data/文件夹中([下载连接](https://drive.google.com/file/d/1ah6lLDIRuEgeVQYy-wEM9buNtnjSs91F/view?usp=share_link))
    
2. 绘制Mask

    使用Mask_GUI对FinalMM.mat文件绘制mask，并保存到同目录下命名为mask.png

------
### 程序运行
1. 运行Mask GUI (标注mask用)
    ```
    python mask_gui.py 
    ```
2. 像素预测 (配置信息见 [Config](#config-configyaml) )
    ```
    sh main.sh
    ```
    notes: 计算超像素和特征模板部分耗时较长。可仅运行预测部分(``` python predict.py```)
---------
### 自定义代码修改
students_submission.py

    1. get_mask() 用于修改mask_gui.py中的的计算mask方法。
    2. feature_list 用于自定义修改程序使用的偏振特征。
    3. get_pft() 用于修改计算pft方法。

### Config (config.yaml) 
config.yaml中的参数说明
| 参数名           | 类型    | 默认值/示例                        | 说明                                                         |
|------------------|---------|------------------------------------|--------------------------------------------------------------|
| n_clusters       | int     | 1024                               | 超像素聚类的类别数                                           |
| batch_size       | int     | 1000                               | 聚类时每批处理的样本数                                       |
| random_state     | int     | 42                                 | 随机种子，保证实验可复现                                     |
| data_dir         | str     | data/贴壁细胞样本                  | 混合缪勒矩阵图像和mask图的文件夹路径                         |
| checkpoint_dir   | str     | ./ckpt                             | 聚类模型和标准化器的保存路径                                 |
| feature_type     | str     | pbp                                | 特征类型，支持 pbp 或 muller                                 |
| pft_dir          | str     | data/贴壁细胞样本/HeLa/ROI1        | 偏振特征模板（单类细胞）的缪勒矩阵图像和mask路径             |
| pred_dir         | str     | data/贴壁细胞样本/HeLa/ROI2        | 预测用的缪勒矩阵图像和mask路径                               |
| k_std            | float   | 1.0                                | 距离质心的标准差倍数，决定细胞像素判定阈值                   |
| output_dir       | str     | ./output                           | 保存识别出的细胞像素点的输出路径                             |
