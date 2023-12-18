
import os
import pickle
import math
import pandas as pd
import numpy as np

name = "obs.csv"
data1 = pd.read_csv(f"../source/{name}", index_col=0)
features = data1.columns
attr = features


# 邻域粗糙集属性约简算法
def attribute_reduction(data, attr_list, res, Y):
    # 初始化结果集
    result_list = []
    # 遍历属性列表
    for attr in attr_list:
        # 计算属性的熵
        entropy = calculate_entropy(data, attr, res)
        # 计算属性的邻域熵
        neighbor_entropy = calculate_neighbor_entropy(data, attr, res, Y)
        # 计算属性的粗糙度
        roughness = entropy - neighbor_entropy
        # 将属性和粗糙度添加到结果集
        result_list.append((attr, roughness))
    # 根据粗糙度排序
    result_list.sort(key=lambda x: x[1], reverse=True)
    # 返回结果
    return result_list


# 计算属性的熵
def calculate_entropy(data, attr, res):
    # 来的是单个的熵
    dats = res[attr]
    emp = 0
    for i in dats:
        p = len(i) / len(data)
        emp += -p * math.log2(p)
    return emp


def intersect(l1, l2):
    x = 0
    for i in l1:
        if i in l2:
            x += 1
    return x / len(l1)


# 计算属性的邻域熵
def calculate_neighbor_entropy(data, attr, res, Y):
    mep = 0
    for i in res[attr]:
        p = len(i) / len(data)
        py = 0
        for j in Y:
            ob = intersect(i, j)
            if ob == 0:
                ob = 0.000000001
            py += ob * math.log2(ob)
        mep += p * py
    return -mep


def low_upper(space: list[set], label: set):
    """
    :param space: 这里是划分的空间
    :param label: 概念Y
    :return: 返回上下近似集
    """
    low = set()
    upper = set()
    for item in space:
        if item & label == item:
            low.update(item)
        if item & label:
            upper.update(item)
    return low, upper


def neighborhood_2(data, feature_subset: list, delta: float):
    """
    :param data: 整体的数据集 DataFrame
    :param feature_subset: 进行划分的特征
    :param delta: 邻域阈值
    :return: 返回整体的划分的结果 list[set()]
    邻域划分空间, 注意是否包含划分类别
    """
    try:
        feature_subset = feature_subset.tolist()
    except Exception as turn_list_error:
        # print(turn_list_error)
        pass
    split_space = []
    for index, data_item in data.iterrows():
        all_sample = (data[feature_subset] - data_item[feature_subset]) ** 2
        res = all_sample.sum(axis=1)
        sqrt_series = np.sqrt(res)
        temps = set(sqrt_series[sqrt_series < delta].index)
        if temps not in split_space:
            split_space.append(temps)
    return split_space


def Approximate_classification_accuracy(Approximate_classification_feature: list, target: str) -> float:
    """
    :param target: 决策属性名
    :param Approximate_classification_feature: 近似分类精度的特征集
    :return:
    进行的操作是邻域划分完毕，进行每一个label的决策分类精度的划分
    """
    split_space = neighborhood_2(data1, Approximate_classification_feature, sigima1)
    # 求得下近似
    label_space = neighborhood_2(data1, [target], sigima1)
    low_abs = 0
    upper_abs = 0
    for label_item in label_space:
        low, upper = low_upper(split_space, label_item)
        low_abs += len(low)
        upper_abs += len(upper)
    return low_abs / upper_abs


if __name__ == '__main__':

    si = [0.025] # 这里是设置划分的阈值
    for sigima1 in si:
        datas = {}
        Y = None
        if not os.path.exists(f'{name}dats{sigima1}.txt'):
            f = open(f'{name}dats{sigima1}.txt', 'w')
            for i in attr:
                if i != "OS_time":
                    datas[i] = neighborhood_2(data1, [i], sigima1)
                else:
                    Y = neighborhood_2(data1, [i], sigima1)
            f.write(str(datas))
            f.write('\n')
            f.write(str(Y))
            f.close()
        else:
            f = open(f'{name}dats{sigima1}.txt', 'r')
            datas = eval(f.readline())
            Y = eval(f.readline())
            f.close()
        # datas:字典 Y：列表
        res = attribute_reduction(data1, attr[:-1], datas, Y)
        with open(f"./pkldata/{name}datas{sigima1}.pkl", 'wb') as f:
            pickle.dump(res, f)
        subset_features = []
        for fea, val in res:
            subset_features.append(fea)
            app_res = Approximate_classification_accuracy(subset_features, "OS_time")
            if app_res == 1:
                break
        print("约简结果：" ,subset_features)
