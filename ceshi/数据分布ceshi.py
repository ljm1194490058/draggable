import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据集
iris_df = pd.read_csv('./iris.csv')

# 创建图形
sns.displot(data=iris_df, x="sepal_length", hue="species", kind="kde")
sns.displot(data=iris_df, x="sepal_width", hue="species", kind="kde")
sns.displot(data=iris_df, x="petal_length", hue="species", kind="kde")
sns.displot(data=iris_df, x="petal_width", hue="species", kind="kde")

# 显示图形
plt.show()