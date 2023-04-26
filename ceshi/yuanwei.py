import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools

def remove_outliers(df, columns):
    # 移除异常值，使用3倍标准差法
    for col in columns:
        df = df[(df[col] - df[col].mean()) / df[col].std() < 3]
    return df

def handle_missing_values(df, columns):
    # 处理缺失值，使用中位数填充
    for col in columns:
        df.fillna(df[col].median(), inplace=True)
    return df

def numericalize_data(df, column_name):
    # 将类别数据进行数值化
    unique_items = df[column_name].unique()
    mapping = {item: index for index, item in enumerate(unique_items)}
    df[column_name] = df[column_name].map(mapping)
    return df

def normalize_data(df, columns):
    # 归一化处理
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def apriori(df, min_sup=0.5, min_conf=0.7):
    """
    实现Apriori算法进行关联规则分析
    :param df: 数据集，每列代表一个项
    :param min_sup: 最小支持度，默认为0.5
    :param min_conf: 最小置信度，默认为0.7
    :return:
    """
    itemsets = {}  # 存储频繁项集及其支持度
    rules = []  # 存储关联规则及其置信度和支持度

    # 第一次扫描：计算项集的支持度
    for col in df.columns:
        count = df[col].sum()
        support = count / len(df)
        if support >= min_sup:
            itemsets[(col,)] = {'support': support, 'count': count}

    # 构建候选项集
    k = 2
    while True:
        candidates = set(itertools.chain(*(itertools.combinations(itemset, k - 1) for itemset in itemsets)))
        if not candidates:
            break

        # 第二次扫描：计算候选项集的支持度
        item_counts = {candidate: 0 for candidate in candidates}
        for _, row in df.iterrows():
            for candidate in candidates:
                if all(item in row.index and row[item] for item in candidate):
                    item_counts[candidate] += 1

        # 削减候选项集
        frequent_itemsets = {}
        for itemset, count in item_counts.items():
            support = count / len(df)
            if support >= min_sup:
                frequent_itemsets[itemset] = {'support': support, 'count': count}

        # 存储频繁项集及其支持度
        itemsets.update(frequent_itemsets)

        # 构建规则
        for itemset in frequent_itemsets.keys():
            if len(itemset) == 1:
                continue
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    consequent = tuple(set(itemset) - set(antecedent))
                    conf = frequent_itemsets[itemset]['support'] / itemsets[antecedent]['support']
                    if conf >= min_conf:
                        rules.append((antecedent, consequent, conf, frequent_itemsets[itemset]['support']))

        k += 1

    return itemsets, rules


def iris_data_summary(file_path):
    # 读取csv文件
    df = pd.read_csv(file_path, header=None)

    # 数据规模
    data_size = df.shape[0]
    print('数据规模:', data_size)

    # 数据形状
    data_shape = df.shape[1]
    print('数据形状:', data_shape)

    # 数据类型
    data_types = df.dtypes
    print('数据类型:\n', data_types)

    # 缺失值
    missing_values = df.isnull().sum().sum()
    print('缺失值数量:', missing_values)

    # # 异常值（离群点）
    # q1 = df.quantile(0.25)
    # q3 = df.quantile(0.75)
    # iqr = q3 - q1
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr
    # outliers = ((df < lower_bound) | (df > upper_bound)).sum().sum()
    # print('异常值数量:', outliers)
# 加载数据集
import numpy as np
df = pd.read_csv('./iris.csv')
all_int_list = []
all_string_list = []
columns = []
all_column = []

for i in range(0, df.shape[0]):
    all_column.append(i+1)
for i, col in enumerate(df.columns):
    if df[col].dtype == 'float' or df[col].dtype == 'int':
        list1 = df.iloc[:, i].tolist()
        # 每条折线的数据
        all_int_list.append(list1)
        # 有哪些折线
        columns.append(col)
    else:
        pie_data_index = list(df[col].value_counts().index)
        pie_data = list(df[col].value_counts())
        for i in range(len(pie_data)):
            dic = {}
            dic['name'] = pie_data_index[i]
            dic['value'] = pie_data[i]
            all_string_list.append(dic)

#  散点图
san_list = []
for j, k in zip(df[columns[0]], df[columns[1]]):
    san_list.append([j, k])
print("colu", columns)
print("san", san_list)
exit(0)
# 计算每一列的离群值
num_outliers = []
for col in df.columns:
    if df[col].dtype == 'float' or df[col].dtype == 'int':

    # 使用 Z-分数方法计算离群值
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        num_outliers.append(sum(z_scores > 3))
    else:
        num_outliers.append('该数据类型不支持该处理！')


# 打印每一列的离群值列表
print(num_outliers)
exit(0)
# 提取每个int特征为列表
all_int_list = []
all_string_list = []
for i, col in enumerate(df.columns):
    if df[col].dtype == 'float' or df[col].dtype == 'int':
        list1 = df.iloc[:, i].tolist()
        all_int_list.append(list1)
    else:
        pie_data_index = list(df[col].value_counts().index)
        pie_data = list(df[col].value_counts())
        for i in range(len(pie_data)):
            dic = {}
            dic['name'] = pie_data_index[i]
            dic['value'] = pie_data[i]
            all_string_list.append(dic)
print("all_int_list", all_int_list)
print("all_string_list", all_string_list)
exit(0)



# 计算每一列的离群值
num_outliers = []
for col in df.columns:
    if df[col].dtype == 'float' or df[col].dtype == 'int':

    # 使用 Z-分数方法计算离群值
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        num_outliers.append(sum(z_scores > 3))

# 打印每一列的离群值列表
print(num_outliers)
exit(0)
# 获取每一列的缺失值数量
missing_values = df.isnull().sum().tolist()
print("miss", missing_values)
# 将缺失值数量转换为百分比
missing_percentages = round((missing_values / len(df)) * 100, 2)
# 将结果存储在一个列表中
result = []
for column in df.columns:
    result.append((column, missing_values[column], missing_percentages[column]))
# 输出结果
print("列名\t\t缺失值数量\t缺失值百分比")
for r in result:
    print(f"{r[0]}\t\t{r[1]}\t\t{r[2]}%")

exit(0)
# 基本统计信息
shape = df.shape
# 输出数据集的形状和大小
data_shape = str(shape[0]) + "行 x " + str(shape[1]) + "列"
print("data_shape", data_shape)



# 计算每个特征的平均值
values = df.values.tolist()
print("value", values)
median_list = []
for feature in df.columns:
    if df[feature].dtype == 'float' or df[feature].dtype == 'int':
        median = round(df[feature].max(), 2)
    else:
        median = '该数据类型不支持该处理！'
    median_list.append(median)

print(median_list)
# iris_data_summary('./iris.csv')





# 指定需要处理的列
cols_to_process = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# 指定需要处理的列和需要数值化的列
target_column = 'species'

# 移除异常值
df = remove_outliers(df, cols_to_process)

# 处理缺失值
df = handle_missing_values(df, cols_to_process)

# 数值化处理
df = numericalize_data(df, target_column)

# 关联分析
df_no_label = df.drop(labels='species', axis=1)  # 删除掉无关列
itemsets, rules = apriori(df_no_label, min_sup=0.4, min_conf=0.6)

print('频繁项集：')
for itemset, support in sorted(itemsets.items(), key=lambda x: -x[1]['support']):
    print('{}: {:.3f}'.format(','.join(itemset), support['support']))
print('\n关联规则：')
for antecedent, consequent, conf, support in sorted(rules, key=lambda x: (-x[2], -x[3])):
    print('{} => {}: {:.3f} (support={:.3f})'.format(','.join(antecedent), ','.join(consequent), conf, support))









