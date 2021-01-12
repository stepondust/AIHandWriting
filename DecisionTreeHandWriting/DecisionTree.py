'''
Description: 自己动手写决策树（一）——初步搭建决策树框架
Url: https://blog.csdn.net/qq_44009891/article/details/105787894
Author: stepondust
Date: 2020-05-02
'''
import numpy as np
import pandas as pd
import math, sys, os


# 数据预处理
def dataProcess(source_fpath, target_fpath):
    with open(source_fpath) as source_f:
        sample_list = []
        for line in source_f:
            content = line.strip().split(",")
            sample_list.append(np.array(content))
        csvdf = pd.DataFrame(sample_list)
        csvdf.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
        csvdf.to_csv(target_fpath, index=0)  # 保存为 CSV 文件时不保留行索引


# 给定数据集，计算并返回其信息熵
def informationEntropy(dataset):
    entropysum = 0
    category_list = list(dataset["Class"])
    for category in set(dataset["Class"]):
        pk = category_list.count(category) / len(dataset)
        entropysum += pk * math.log(pk, 2)
    return (-1) * entropysum


# 给定数据集和离散类型属性，计算并返回根据该属性划分数据集得到的信息增益
def informationDiscreteGain(dataset, attribute):
    entropy = informationEntropy(dataset)
    entropysum = 0
    attribute_value_list = list(dataset[attribute])
    for attribute_value in set(dataset[attribute]):
        weight = attribute_value_list.count(attribute_value) / len(dataset)
        entropysum += weight * informationEntropy(dataset[dataset[attribute] == attribute_value])
    return entropy - entropysum


# 给定数据集和连续类型属性，计算根据该属性划分数据集得到的信息增益，并返回在该属性上的划分点以及信息增益
def informationContinuousGain(dataset, attribute):
    entropy = informationEntropy(dataset)
    attribute_value_list = sorted(set(dataset[attribute]))
    if len(attribute_value_list) == 1:
        thresholds = [attribute_value_list[0]]
    else:
        thresholds = [(attribute_value_list[i] + attribute_value_list[i + 1]) / 2 for i in range(len(attribute_value_list) - 1)] # 候选划分点集合
    
    threshold_entropysum_dict = {}
    for threshold in thresholds:
        lessthreshold = dataset[dataset[attribute] <= threshold]
        morethreshold = dataset[dataset[attribute] > threshold]
        lessweight = len(lessthreshold) / len(dataset)
        moreweight = len(morethreshold) / len(dataset)
        entropysum = lessweight * informationEntropy(lessthreshold) + moreweight * informationEntropy(morethreshold)
        threshold_entropysum_dict[threshold] = entropysum
        
    threshold_entropysum_sorted = sorted(threshold_entropysum_dict.items(), key=lambda item: item[1])
    minentropysum_threshold = threshold_entropysum_sorted[0][0]
    minentropysum = threshold_entropysum_sorted[0][1]
    return minentropysum_threshold, entropy - minentropysum


# 计算数据集中类数量最多的类
def maxNumOutcome(dataset):
    category_list = list(dataset["Class"])
    category_dict = {}
    for category in set(dataset["Class"]):
        category_dict[category] = category_list.count(category)
    category_sorted = sorted(category_dict.items(), key=lambda item: item[1], reverse=True)
    return category_sorted[0][0]


# 递归生成决策树结点，返回决策树模型字典
def treeNodeGenerate(dataset, attribute_list):
    if len(set(dataset["Class"])) == 1:
        node = list(set(dataset["Class"]))[0]
    elif len(attribute_list) == 0 or sum([len(set(dataset[attribute])) - 1 for attribute in attribute_list]) == 0:
        node = maxNumOutcome(dataset)
    else:
        attribute_gain_dict = {}
        for attribute in attribute_list:
            threshold, attribute_gain = informationContinuousGain(dataset, attribute)
            attribute_gain_dict[attribute] = threshold, attribute_gain
        attribute_gain_sorted = sorted(attribute_gain_dict.items(), key=lambda item: item[1][1], reverse=True)
        maxgain_attribute = attribute_gain_sorted[0][0]
        maxgain_threshold = attribute_gain_sorted[0][1][0]

        son_node_attribute_list = attribute_list.copy()
        son_node_attribute_list.remove(maxgain_attribute)

        left_node_dataset = dataset[dataset[maxgain_attribute] <= maxgain_threshold]
        if len(left_node_dataset) == 0:
            leftnode = maxNumOutcome(dataset)
        else:
            leftnode = treeNodeGenerate(left_node_dataset, son_node_attribute_list)
        
        right_node_dataset = dataset[dataset[maxgain_attribute] > maxgain_threshold]
        if len(right_node_dataset) == 0:
            rightnode = maxNumOutcome(dataset)
        else:
            rightnode = treeNodeGenerate(right_node_dataset, son_node_attribute_list)
        
        if leftnode == rightnode:
            node = leftnode
        else:
            node = {}
            node[(maxgain_attribute, maxgain_threshold)] = {"<=":leftnode, ">":rightnode}

    return node


# 预测一条数据的结果
def predictOne(tree_train_model, testdata):
    if type(tree_train_model) == str:
        predict_value = tree_train_model
    elif type(tree_train_model) == dict:
        key = list(tree_train_model)[0]
        if testdata[key[0]] <= key[1]:
            son_tree_train_model = tree_train_model[key]["<="]
        else:
            son_tree_train_model = tree_train_model[key][">"]
        predict_value = predictOne(son_tree_train_model, testdata)
    return predict_value


# 进行模型预测，返回预测结果列表
def predict(tree_train_model, testdataset):
    predict_list = []
    for i in range(len(testdataset)):
        predict_value = predictOne(tree_train_model, testdataset.loc[i])
        predict_list.append((testdataset.loc[i]["Class"], predict_value))
    return predict_list


# 对预测模型进行精确度评估
def predictAccuracy(predict_list):
    predict_true_num = 0
    for bigram in predict_list:
        if bigram[0] == bigram[1]:
            predict_true_num += 1
    accuracy = predict_true_num / len(predict_list)
    return accuracy


# 将数据集划分为训练集和测试集
def subdatasetPartitioning(dataset):

    # 打乱索引
    index = [i for i in range(len(dataset))]
    np.random.seed(1)
    np.random.shuffle(index)

    # 以 8:2 划分训练集和测试集
    traindatasetlen = int(len(dataset) * 0.8)
    traindataset = dataset.loc[index[:traindatasetlen]]
    testdataset = dataset.loc[index[traindatasetlen:]]

    return traindataset, testdataset


# 类似于分层抽样，每个类别划分同样个数的样本给训练集
def datasetPartitioning(dataset):

    traindataset_list = []
    testdataset_list = []
    for i in range(3):
        subdataset = dataset.loc[i * 50 : (i + 1) * 50 - 1]
        subdataset = subdataset.reset_index()  # 重置索引
        subtraindataset, subtestdataset = subdatasetPartitioning(subdataset)
        traindataset_list.append(subtraindataset)
        testdataset_list.append(subtestdataset)

    traindataset = pd.concat(traindataset_list, ignore_index=True)  # ignore_index=True 表示忽略原索引，类似重置索引的效果
    testdataset = pd.concat(testdataset_list, ignore_index=True)

    return traindataset, testdataset


if __name__ == "__main__":

    # 数据预处理
    source_fpath = "../datasets/iris.data"
    target_fpath = "iris.csv"
    dataProcess(source_fpath, target_fpath)
    
    # 读取数据
    dataset = pd.read_csv("iris.csv")

    # 将数据集以 8:2 划分为训练集和测试集（每一类别抽 40 个作训练集）
    traindataset, testdataset =  datasetPartitioning(dataset)

    # 使用训练集进行模型训练
    attribute_list = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    tree_train_model = treeNodeGenerate(traindataset, attribute_list)
    print("The Dict of Trained Model:")
    print(tree_train_model, "\n")

    # 使用训练好的模型在测试集上进行测试得到预测结果
    predict_list = predict(tree_train_model, testdataset)
    print("The List of Predicting Outcomes (Actual Label, Predicted Value) :")
    print(predict_list, "\n")

    # 对预测结果进行评估
    print("The Accuracy of Model Prediction: ", predictAccuracy(predict_list))
