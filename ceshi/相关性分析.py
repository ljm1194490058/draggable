import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_analysis(data):
    '''
    对给定数据集的所有数值特征进行相关性分析

    参数：
    data：DataFrame 类型，原始数据集

    返回值：
    None
    '''

    # 计算相关系数矩阵
    corr_matrix = data.corr()

    # 绘制相关系数矩阵热力图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

    # 绘制散点图和回归线，并在图中显示相关系数
    for i, col1 in enumerate(data.select_dtypes(include=['float', 'int']).columns[:-1]):
        for col2 in data.select_dtypes(include=['float', 'int']).columns[i + 1:]:
            corr_coef = data[col1].corr(data[col2])
            sns.regplot(x=col1, y=col2, data=data)
            plt.title(f'{col1.capitalize()} vs. {col2.capitalize()} (Correlation Coefficient: {corr_coef:.2f})')
            plt.xlabel(col1.capitalize())
            plt.ylabel(col2.capitalize())
            plt.show()
iris = sns.load_dataset('iris')
correlation_analysis(iris)