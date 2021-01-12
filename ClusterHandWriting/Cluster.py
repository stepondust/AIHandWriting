'''
Description: 自己动手写聚类（一）——初步搭建 k-means 聚类框架
Url: https://blog.csdn.net/qq_44009891/article/details/106214080
Author: stepondust
Date: 2020-05-21
'''
import random, json, os
import pandas as pd
import numpy as np
# from tqdm._tqdm import trange
from tqdm.std import trange


def DataProcess(dataset):
    '''
    msg: 数据预处理
    param {
        dataset:pandas.DataFrame 数据集
    } 
    return: None
    '''
    dataset_len = len(dataset)
    for i in range(dataset_len):
        dataset.loc[i, "Region"] = dataset.loc[i, "Region"].strip()  # 处理 Region 这一列的对齐问题
        for j in range(2, 20):
            value = dataset.iloc[i, j]
            if type(value) == str:
                dataset.iloc[i, j] = float(value.replace(",", "."))  # 处理小数点用逗号表示的问题
    dataset = dataset.fillna(dataset.mean())  # 使用均值插补的方法处理缺失值问题
    dataset.to_csv("countries_of_the_world.csv", index=0)  # 保存为 CSV 文件，index=0 表示不保留行索引
    # dataset.to_csv("countries of the world.csv", index=0)  # 保存为 CSV 文件，index=0 表示不保留行索引


def oneHot(raw_dataset):
    '''
    msg: 使用 One-Hot 处理离散的无序属性
    param {
        raw_dataset:pandas.DataFrame 数据集
    } 
    return: None
    '''
    dataset_len = len(raw_dataset)
    regions = sorted(set(raw_dataset["Region"]))  # 使用集合得到 Region 属性的 11 个属性值即对应 11 个地区
    dataset = raw_dataset.drop(columns="Region")  # 舍弃 Region 属性列
    for region in regions:
        # 如果国家位于某个地区，那么在该地区对应的属性列就填 1，否则就填 0
        region_column = [float(raw_dataset.loc[i, "Region"] == region) for i in range(dataset_len)]
        dataset[region] = region_column  # 增加 11 个地区对应的属性列
    return dataset


def initClusters(dataset, clusters_num, seed):
    '''
    msg: 初始化聚类簇
    param {
        dataset:pandas.DataFrame 数据集
        clusters_num:int 簇的数量
        seed:int 随机数种子
    } 
    return {
        mean_vector_dict:dict 簇均值向量字典
    }
    '''
    dataset_len = len(dataset)
    dataset["Class"] = [-1 for i in range(dataset_len)]  # 给数据集再添加一个属性 “Class”
    random.seed(seed)  # 设置随机数种子
    init_clusters_index = sorted([random.randint(0, dataset_len - 1) for i in range(clusters_num)])  # 从数据集当中随机选择 clusters_num 个样本
    mean_vector_dict = {}  # 创建均值向量字典
    for i in range(clusters_num):
        mean_vector_dict[i] = dataset.iloc[init_clusters_index[i], 1:-1]  # 把这 clusters_num 个样本作为初始均值向量
        dataset.loc[init_clusters_index[i], "Class"] = i  # 在数据集中对应的 “Class” 属性列上填入自己的簇类别号
    return mean_vector_dict


def vectorDist(vector_X, vector_Y, p):
    vector_X = np.array(vector_X)
    vector_Y = np.array(vector_Y)
    return sum((vector_X - vector_Y) ** p) ** (1 / p)


def vectorAverage(cluster):
    return cluster.iloc[:, 1:-1].mean()


def ifEqual(pandas_X, pandas_Y):
    return pandas_X.equals(pandas_Y)


def kMeansClusters(dataset, mean_vector_dict):
    '''
    msg: 实现 k-means 聚类
    param {
        dataset:pandas.DataFrame 数据集
        mean_vector_dict:dict 簇均值向量字典
    } 
    return: None
    '''
    dataset_len = len(dataset)
    bar = trange(100)  # 使用 tqdm 第三方库，调用 tqdm.std.trange 方法给循环加个进度条
    for _ in bar:  # 使用 _ 表示进行占位，因为在这里我们只是循环而没有用到循环变量
        bar.set_description("The clustering is runing") # 给进度条加个描述
        for i in range(dataset_len):
            dist_dict = {}
            for cluster_id in mean_vector_dict:
                dist_dict[cluster_id] = vectorDist(dataset.iloc[i, 1:-1], mean_vector_dict[cluster_id], p=2)  # 计算样本 xi 与各均值向量的距离
            dist_sorted = sorted(dist_dict.items(), key=lambda item: item[1])  # 对样本 xi 与各均值向量的距离进行排序
            dataset.loc[i, "Class"] = dist_sorted[0][0]  # 根据距离最近的均值向量确定 xi 的簇类别并在 “Class” 属性列上填入对应簇类别号，即将 xi 划入相应的簇
        flag = 0
        for cluster_id in mean_vector_dict:
            cluster = dataset[dataset["Class"] == cluster_id]  # 得到簇内的所有样本
            cluster_mean_vector = vectorAverage(cluster)  # 根据簇内的所有样本计算新的均值向量
            if not ifEqual(mean_vector_dict[cluster_id], cluster_mean_vector):  # 判断新的均值向量是否和当前均值向量相同
                mean_vector_dict[cluster_id] = cluster_mean_vector  # 不相同，将新的均值向量替换当前均值向量
            else:
                flag += 1  # 保持当前均值向量不变，并进行计数
        if flag == len(mean_vector_dict):  # 判断是否所有簇的均值向量均未更新
            bar.close()  # 所有簇的均值向量均未更新，关闭进度条
            print("The mean vectors are no longer changing, the clustering is over.")
            return  # 直接退出循环
    print("Reach the maximum number of iterations, the clustering is over.")


def getClusters(dataset, clusters_num):
    '''
    msg: 提取聚类结果
    param {
        dataset:pandas.DataFrame 数据集
        clusters_num:int 簇的数量
    } 
    return {
        clusters_dict:dict 键值对的值为 pandas.DataFrame 类型
        cluster_indexs_dict:dict 键值对的值为 list 类型
        cluster_countries_dict:dict 键值对的值为 list 类型
    }
    '''
    clusters_dict = {}
    cluster_indexs_dict = {}
    cluster_countries_dict = {}
    for cluster_id in range(clusters_num):
        clusters_dict[cluster_id] = dataset[dataset["Class"] == cluster_id]
        cluster_indexs_dict[cluster_id] = list(clusters_dict[cluster_id].index)
        cluster_countries_dict[cluster_id] = list(dataset.loc[cluster_indexs_dict[cluster_id], "Country"])
    return clusters_dict, cluster_indexs_dict, cluster_countries_dict


def distanceMatrix(dataset, path):
    '''
    msg: 以字典形式构建数据集的距离矩阵
    param {
        dataset:pandas.DataFrame 数据集
        path:str 存放距离矩阵的文件，建议格式为 .json
    } 
    return{
        matrix_dict:dict 字典形式的距离矩阵
    }
    '''
    if not os.path.exists(path):
        dataset_len = len(dataset)
        matrix_dict = {}
        for i in range(dataset_len):
            for j in range(i + 1, dataset_len):
                matrix_dict[str((i, j))] = vectorDist(dataset.iloc[i, 1:-1], dataset.iloc[j, 1:-1], p=2)
        with open(path, 'w+') as f:
            json.dump(matrix_dict, f)
    else:
        with open(path, 'r+') as f:
            matrix_dict = json.load(f)
    return matrix_dict


def silhouetteCoefficient(dataset, clusters_num, clusters_dict, cluster_indexs_dict, dist_matrix):
    '''
    msg: 计算数据集中所有样本的轮廓系数的平均值
    param {
        dataset:pandas.DataFrame 数据集
        clusters_num:int 簇的数量
        clusters_dict:dict 键值对的值为 pandas.DataFrame 类型
        cluster_indexs_dict:dict 键值对的值为 list 类型
        dist_matrix:dict 字典形式的距离矩阵
    } 
    return {
        silhouette_coefficient:float 数据集中所有样本的轮廓系数的平均值
    }
    '''
    dataset_len = len(dataset)
    a = np.array([0 for i in range(dataset_len)], dtype=np.float64)
    b = np.array([0 for i in range(dataset_len)], dtype=np.float64)
    s = np.array([0 for i in range(dataset_len)], dtype=np.float64)

    for cluster_id in range(clusters_num):

        cluster_len = len(cluster_indexs_dict[cluster_id])
        clusters_copy_remove = cluster_indexs_dict.copy()
        clusters_copy_remove.pop(cluster_id)

        for i in cluster_indexs_dict[cluster_id]:
            cluster_copy_remove = cluster_indexs_dict[cluster_id].copy()
            cluster_copy_remove.remove(i)
            for j in cluster_copy_remove:
                a[i] += dist_matrix[str((min(i, j), max(i, j)))]
            a[i] = a[i] / cluster_len - 1

            bi = []
            for key in clusters_copy_remove:
                xb = 0
                for k in clusters_copy_remove[key]:
                    xb += dist_matrix[str((min(i, k), max(i, k)))]
                xb = xb / len(clusters_copy_remove[key])
                bi.append(xb)
            if len(bi) != 0:
                b[i] = min(bi)

            s[i] = ((b[i] - a[i]) / max(a[i], b[i]))

    silhouette_coefficient = np.average(s)
    return silhouette_coefficient


if __name__ == "__main__":
    clusters_num = 2
    seed = 1

    raw_dataset = pd.read_csv("../datasets/countries of the world.csv")
    
    dataset = oneHot(raw_dataset)

    mean_vector_dict = initClusters(dataset, clusters_num, seed)

    print(f"Set the number of clusters to {clusters_num} and the random seed to {seed}, start clustering.")
    kMeansClusters(dataset, mean_vector_dict)
    clusters_dict, cluster_indexs_dict, cluster_countries_dict = getClusters(dataset, clusters_num)
    print("The result of clusters:\n", cluster_countries_dict)

    dist_matrix = distanceMatrix(dataset, "matrix.json")

    silhouette_coefficient = silhouetteCoefficient(dataset, clusters_num, clusters_dict, cluster_indexs_dict, dist_matrix)
    print(f"The average of the silhouette coefficients of all samples in the dataset is {silhouette_coefficient}")