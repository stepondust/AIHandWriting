'''
Description: 初步搭建 Apriori 频繁模式挖掘框架
Author: stepondust
Date: 2020-05-31
'''
import numpy as np


def findFrequentOneItemsets(dataset, min_sup):
    '''
    msg: 发现所有满足最小支持度计数阈值的频繁 1 项集
    param {
        dataset:numpy.ndarray Groceries 事务数据集
        min_sup:int 最小支持度计数阈值
    } 
    return{
        frequent_one_itemsets:dict 字典形式的频繁 1 项集的集合，每个键值对的键为包含商品的元组，值为商品对应的支持度计数
    }
    '''
    goods_list = [i for i in dataset.flatten() if i != ''] # 将 numpy.ndarray 类型的数据集平铺并去除所有空字符串 ''，得到所有商品的列表
    goods_set = set(goods_list)
    frequent_one_itemsets = {}
    for key in goods_set:
        num = goods_list.count(key) # 每件商品的支持度计数
        if num < min_sup:
            continue # 当前商品的支持度计数小于最小支持度计数阈值，放弃该商品并搜索下一件商品
        else:
            frequent_one_itemsets[(key, )] = num # 当前商品的支持度计数大于或等于最小支持度计数阈值，将该商品加入频繁 1 项集的集合
    return frequent_one_itemsets


def kMinusOneSubset(superset):
    '''
    msg: 得到 k 项集的所有 k - 1 项子集
    param {
        superset:tuple k 项集
    } 
    return{
        k_minus_one_subset:set k 项集的所有 k - 1 项子集
    }
    '''
    sub_sorted = sorted(superset)
    k_minus_one_subset = set([tuple(sub_sorted[:i] + sub_sorted[i + 1:]) for i in range(len(sub_sorted))])
    return k_minus_one_subset


def aprioriGen(frequent_k_minus_one_itemsets):
    '''
    msg: 根据频繁（k - 1）项集的集合得到候选 k 项集的集合
    param {
        frequent_k_minus_one_itemsets:dict 频繁（k - 1）项集的集合
    } 
    return{
        candidate_k_set:set 候选 k 项集的集合
    }
    '''
    k_minus_one_list_sorted = sorted(frequent_k_minus_one_itemsets.keys())
    candidate_k_set = set()
    for i in k_minus_one_list_sorted:
        temp_set = set([
            tuple(set(i + j)) for j in k_minus_one_list_sorted if i < j
            ]) # 连接步：将频繁（k - 1）项集的集合与其自己做连接，得到候选的 k 项集的集合
        candidate_k_set.update([
            tuple(sorted(j)) for j in temp_set if len(kMinusOneSubset(j) - set(k_minus_one_list_sorted)) == 0
            ]) # 剪枝步：根据先验性质，删除非频繁的候选 k 项集
    return candidate_k_set


def candidateItemsets(candidate_k_set, t):
    '''
    msg: 得到事务 t 的候选子集，这些候选子集均是候选 k 项集的集合的元素
    param {
        candidate_k_set:set 候选 k 项集的集合
        t:numpy.ndarray Groceries 事务数据集中的事务
    } 
    return{
        candidate:list 以事务 t 的候选子集为元素的列表，这些候选子集均是候选 k 项集的集合的元素
    }
    '''
    candidate = []
    for i in candidate_k_set:
        if t.issuperset(i): # 判断事务 t 是否是候选 k 项集 i 的超集，即判断候选 k 项集 i 是否是事务 t 的子集
            candidate.append(tuple(sorted(i)))
    return candidate


def apriori(dataset, frequent_one_itemsets, min_sup):
    '''
    msg: 获得所有的频繁项集
    param {
        dataset:numpy.ndarray Groceries 事务数据集
        frequent_one_itemsets:dict 字典形式的频繁 1 项集的集合，每个键值对的键为包含商品的元组，值为商品对应的支持度计数
        min_sup:int 最小支持度计数阈值
    } 
    return{
        frequent_itemsets:dict 字典形式的频繁项集的集合
    }
    '''
    frequent_itemsets_list = [frequent_one_itemsets] # 创建一个以每一层的频繁项集的集合为元素的列表
    k = 1
    while(len(frequent_itemsets_list[k - 1]) != 0): # 判断上一层的（k - 1）项集的集合是否为空，为空则说明不能再找到频繁项集，退出循环，不为空则继续循环
        candidate_k_set = aprioriGen(frequent_itemsets_list[k - 1]) # 根据频繁（k - 1）项集的集合得到候选 k 项集的集合
        if len(candidate_k_set) == 0: # 判断候选 k 项集的集合是否为空，为空则说明不能再找到频繁项集，退出循环，不为空则继续
            frequent_itemsets_list.append({}) # 对应 del frequent_itemsets_list[-1] 语句，防止误删
            break
        candidate_itemsets_list = [] # 创建一个以各个事务的子集为元素的列表，它们均是候选的
        for t in dataset: # t 为事务数据集中的事务
            ct = candidateItemsets(candidate_k_set, set(t)) # 得到事务 t 的候选子集，这些候选子集均是候选 k 项集的集合的元素
            candidate_itemsets_list.extend(ct)
        candidate_set = set(candidate_itemsets_list)
        candidate_dict = {} # 创建字典形式的频繁 k 项集的集合
        for key in candidate_set:
            num = candidate_itemsets_list.count(key) # 每个候选 k 项集的支持度计数
            if num < min_sup:
                continue # 当前候选 k 项集不满足最小支持度计数阈值，放弃该候选 k 项集并搜索下一候选 k 项集
            else:
                candidate_dict[key] = num # 当前候选 k 项集满足最小支持度计数阈值，将该候选 k 项集加入频繁 k 项集的集合
        frequent_itemsets_list.append(candidate_dict)
        k += 1
    del frequent_itemsets_list[-1] # 上一层的的项集的集合为空，退出循环并且删除上一层的的项集
    frequent_itemsets = {} # 创建字典形式的频繁项集的集合
    for i in frequent_itemsets_list:
        frequent_itemsets.update(i)
    return frequent_itemsets


def allProperSubset(superset):
    '''
    msg: 获得集合的所有非空真子集
    param {
        superset:tuple 元组形式的集合
    } 
    return{
        proper_subsets:list 列表形式的集合，列表当中的每个元素均为元组形式的给定集合的非空真子集
    }
    '''
    n = len(superset)
    proper_subsets = []
    for i in range(1, 2 ** n - 1): # 根据子集个数，循环遍历所有非空真子集
        proper_subset = []
        for j in range(n):
            if (i >> j) % 2: # 判断二进制下标为 j 的位置数是否为 1
                proper_subset.append(superset[j])
        proper_subsets.append(tuple(proper_subset))
    return proper_subsets


def associationRules(frequent_itemsets, min_conf):
    '''
    msg: 由频繁项集产生强关联规则
    param {
        frequent_itemsets:dict 字典形式的频繁项集的集合
        min_conf:double 最小置信度阈值
    } 
    return{
        rules_list:list 以规则为元素的列表，其中规则以三元组 (频繁项集 Z，子集 S，子集 Z-S) 形式组织，对应规则 S⇒Z−S
    }
    '''
    rules_list = [] # 创建一个以规则为元素的列表，其中规则以三元组 (频繁项集 Z，子集 S，子集 Z-S) 形式组织，对应规则 S⇒Z−S
    for frequent_itemset in frequent_itemsets: # 遍历所有的频繁项集
        if len(frequent_itemset) == 1: # 判断是否是频繁 1 项集，因为使用频繁 1 项集产生的规则是无用的
            continue
        else:
            proper_subsets = allProperSubset(frequent_itemset) # 得到当前频繁项集的所有非空真子集
            frequent_itemset_support = frequent_itemsets[frequent_itemset] # 得到当前频繁项集对应的支持度计数
            for proper_subset in proper_subsets: # 遍历当前频繁项集的所有非空真子集并生成对应规则
                if frequent_itemset_support / frequent_itemsets[proper_subset] >= min_conf: # 判断当前规则是否满足最小置信度阈值
                    rules_list.append((
                        frequent_itemset, proper_subset, tuple(sorted(set(frequent_itemset) - set(proper_subset)))
                        )) # 当前规则满足最小置信度阈值，将其以三元组形式加入规则列表
    return rules_list


def lift(rule, total_num):
    '''
    msg: 使用提升度评判关联规则
    param {
        rule:tuple 以三元组 (频繁项集 Z，子集 S，子集 Z-S) 形式组织的规则
        total_num:int 事务数据集中所有事务的数量
    } 
    return{
        rule_lift:double 关联规则的提升度
    }
    '''
    rule_lift = frequent_itemsets[rule[0]] * total_num / (frequent_itemsets[rule[1]] * frequent_itemsets[rule[2]])
    return rule_lift


def printRules(frequent_itemsets, rules_list, total_num, min_sup, min_conf):
    '''
    msg: 以固定格式打印所有规则
    param {
        frequent_itemsets:dict 字典形式的频繁项集的集合
        rules_list:list 以规则为元素的列表，其中规则以三元组 (频繁项集 Z，子集 S，子集 Z-S) 形式组织，对应规则 S⇒Z−S
        total_num:int 事务数据集中所有事务的数量
        min_sup:int 最小支持度计数阈值
        min_conf:double 最小置信度阈值
    } 
    return: None
    '''
    min_sup_threshold = min_sup / total_num
    for rule in rules_list:
        rule_lift = lift(rule, total_num)
        print("{: <40}--> {: <22}: support >= {:.2%}, confidence >= {:.2%}, lift = {:.3}" \
            .format(str(rule[1]), str(rule[2]), min_sup_threshold, min_conf, rule_lift))


if __name__ == "__main__":
    
    min_sup = 400
    min_conf = 0.3

    dataset = np.loadtxt("groceries.csv", delimiter=",", usecols=range(1, 33), skiprows=1, dtype=str)

    frequent_one_itemsets = findFrequentOneItemsets(dataset, min_sup)

    frequent_itemsets = apriori(dataset, frequent_one_itemsets, min_sup)

    rules_list = associationRules(frequent_itemsets, min_conf)

    printRules(frequent_itemsets, rules_list, len(dataset), min_sup, min_conf)