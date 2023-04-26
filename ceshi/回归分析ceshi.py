import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def iris_linear_regression(file_path, x_col, y_col):
    # 加载数据集
    iris_df = pd.read_csv(file_path)

    # 将鸢尾花的品种转换成三个二元变量
    species_df = pd.get_dummies(iris_df[["species"]])
    iris_df = pd.concat([iris_df, species_df], axis=1)

    # 创建线性回归模型
    lr = LinearRegression()

    # 拟合模型
    X = iris_df.drop([y_col, "species"], axis=1)
    y = iris_df[y_col]
    lr.fit(X, y)

    # 绘制回归曲线和散点图
    sns.regplot(x=X[x_col], y=y, data=iris_df)

    # 显示图形和相关系数
    plt.show()
    print("相关系数：", lr.coef_)

iris_linear_regression("iris.csv", "petal_width", "sepal_length")