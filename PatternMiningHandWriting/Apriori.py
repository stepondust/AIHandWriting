'''
Description: 自己动手写模式挖掘（一）——初步搭建 Apriori 频繁模式挖掘框架
Url: https://blog.csdn.net/qq_44009891/article/details/106423085
Author: stepondust
Date: 2020-05-31
'''
import numpy as np
import sys


def findFrequentOneItemsets(dataset, min_sup):
    goods_list = [i for i in dataset.flatten() if i != '']
    goods_set = set(goods_list)
    goods_dict = {}
    for key in goods_set:
        num = goods_list.count(key)
        if num < min_sup:
            continue
        else:
            goods_dict[(key, )] = num
    # print(goods_dict)
    frequent_one_itemsets = goods_dict
    return frequent_one_itemsets


def kMinusOneSubset(superset):
    sub_sorted = sorted(superset)
    subset = set([tuple(sub_sorted[:i] + sub_sorted[i + 1:]) for i in range(len(sub_sorted))])
    return subset


def aprioriGen(frequent_k_minus_one_itemsets):
    k_minus_one_list_sorted = sorted(frequent_k_minus_one_itemsets.keys())
    # k_minus_one_list_sorted = [('bottled water', 'other vegetables'), ('other vegetables', 'soda'), ('bottled water', 'soda')]
    # print(k_minus_one_list_sorted)
    k_set = set()
    for i in k_minus_one_list_sorted:
        temp_set = set([tuple(set(i + j)) for j in k_minus_one_list_sorted if i < j])
        # print(temp_set)
        # print([kMinusOneSubset(j) for j in temp_set])
        # print([kMinusOneSubset(j) - set(k_minus_one_list_sorted) for j in temp_set])
        # print("\n")
        k_set.update([tuple(sorted(j)) for j in temp_set if len(kMinusOneSubset(j) - set(k_minus_one_list_sorted)) == 0])
        # print(k_set)
    # print(k_set)
    return k_set


def candidateItemsets(C, t):
    candidate = []
    for i in C:
        if t.issuperset(i):
            candidate.append(tuple(sorted(i)))
    return candidate


def apriori(dataset, frequent_one_itemsets, min_sup):
    L = [frequent_one_itemsets]
    k = 1
    while(len(L[k - 1]) != 0):
        C = aprioriGen(L[k - 1])
        if len(C) == 0:
            L.append(C)
            break
        candidate_list = []
        for t in dataset:
            Ct = candidateItemsets(C, set(t))
            candidate_list.extend(Ct)
        candidate_set = set(candidate_list)
        candidate_dict = {}
        for key in candidate_set:
            num = candidate_list.count(key)
            if num < min_sup:
                continue
            else:
                candidate_dict[key] = num
        # print(candidate_dict)
        L.append(candidate_dict)
        k += 1
    del L[-1]
    # print(L)
    frequent_itemsets = {}
    for i in L:
        frequent_itemsets.update(i)
    return frequent_itemsets


def allProperSubset(superset):
    n = len(superset)
    subset = []
    for i in range(1, 2 ** n - 1):  # 子集个数, 每循环一次一个子集
        combo = []
        for j in range(n):  # 用来判断二进制下标为 j 的位置数是否为 1
            if (i >> j) % 2:
                combo.append(superset[j])
        subset.append(tuple(combo))
    return subset


def PowerSetsBinary(items):
    N = len(items)
    for i in range(1, 2 ** N - 1):  # 子集个数, 每循环一次一个子集
        combo = []
        for j in range(N):  # 用来判断二进制下标为 j 的位置数是否为 1
            if(i >> j) % 2:
                combo.append(items[j])
        print(combo)
# PowerSetsBinary([1, 2, 3])


def associationRules(frequent_itemsets, min_conf):
    rules_list = []
    for frequent_itemset in frequent_itemsets:
        if len(frequent_itemset) == 1:
            continue
        else:
            proper_subsets = allProperSubset(frequent_itemset)
            frequent_itemset_support = frequent_itemsets[frequent_itemset]
            for proper_subset in proper_subsets:
                if frequent_itemset_support / frequent_itemsets[proper_subset] >= min_conf:
                    # print(f"{proper_subset} -> {tuple(sorted(set(frequent_itemset) - set(proper_subset)))}")
                    rules_list.append((frequent_itemset, proper_subset, tuple(sorted(set(frequent_itemset) - set(proper_subset)))))
    return rules_list


def lift(frequent_itemsets, rules_list, total_num):
    for rule in rules_list:
        rule_lift = frequent_itemsets[rule[0]] * total_num / (frequent_itemsets[rule[1]] * frequent_itemsets[rule[2]])
        # print(f"{str(rule[1]):-<40}-> {str(rule[2]): <22}: lift={rule_lift}")
        print(f"{str(rule[1]): <40} -> {str(rule[2]): <22}: lift={rule_lift}")


if __name__ == "__main__":
    min_sup = 200
    min_conf = 0.3
    
    # 读取数据
    dataset = np.loadtxt("../datasets/groceries.csv", delimiter=",", usecols=range(1, 33), skiprows=1, dtype=str)

    # 发现频繁 1 项集
    frequent_one_itemsets = findFrequentOneItemsets(dataset, min_sup)

    # 获得所有的频繁项集
    frequent_itemsets = apriori(dataset, frequent_one_itemsets, min_sup)

    # 由频繁项集产生强关联规则
    rules_list = associationRules(frequent_itemsets, min_conf)

    # 使用提升度评判关联规则
    lift(frequent_itemsets, rules_list, len(dataset))