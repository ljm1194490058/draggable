from apyori import apriori
import pandas as pd

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def assoc_rules(df, min_supp, min_conf):
    # Convert data to list of lists
    transactions = []
    for i in range(len(df)):
        transactions.append([str(df.values[i, j]) for j in range(len(df.columns))])

    # Run Apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_supp, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    # Return results
    return rules
df = pd.read_csv('./iris.csv')
rules = assoc_rules(df, 0.1, 0.5) # (2,9)
rules_df = pd.DataFrame(rules)
rule_column = rules_df.columns.tolist()

print("ru",rules_df)
exit(0)

def iris_assoc_rules(file_path, min_supp, min_conf):
    # Load dataset
    data = pd.read_csv(file_path)

    # Convert data to list of lists
    transactions = []
    for i in range(len(data)):
        transactions.append([str(data.values[i, j]) for j in range(len(data.columns))])
        print(transactions)
    # Run Apriori algorithm
    results = list(apriori(transactions, min_support=min_supp, min_confidence=min_conf))

    # Print results
    for item in results:
        # Print antecedents -> consequents
        print(", ".join([list(item[0])[i] for i in range(len(item[0]) - 1)]) + " -> " + list(item[0])[-1])
        # Print support, confidence, lift
        print("Support: " + str(round(item[1], 4)))
        print("Confidence: " + str(round(item[2][0][2], 4)))
        print("Lift: " + str(round(item[2][0][3], 4)))
        print("-------------------------------------------")
        # 在这段代码中，输出的支持度和置信度代表着满足设定的阈值要求的对应规则的支持度和置信度，可以帮助我们了解不同特征之间的关联程度以及它们对不同鸢尾花品种的影响。
        # 因此，通过调整阈值，我们可以得到不同的关联分析结果，以便进一步分析鸢尾花数据集中不同特征之间的关系。

iris_assoc_rules('./iris.csv', 0.1, 0.5)


"""
在关联规则分析中，支持度（support）、置信度（confidence）和提升度（lift）是常用的评估指标。这些指标可以帮助我们了解数据集中不同项集或规则之间的关系。

支持度（support）：指在所有交易中同时出现A和B的概率。支持度可以用来衡量一个项集的普遍程度，即一组物品经常同时出现的程度。支持度越高，意味着该项集出现的频率越高，代表其重要性与普遍性也越高。

置信度（confidence）：指在交易中包含A的情况下，也包含B的概率，指在包含A的事务中同时包含B的比例，即P(B|A)。置信度可以用来衡量两个项之间的相关性，是条件概率的一种形式。置信度高表示A出现时B也很可能出现，从而反映了两个项之间的强关联程度。

提升度（lift）：指在包含A的交易中，同时包含B的概率比上在所有交易中同时包含B的概率的比值。提升度可以用来判断两个项之间的独立性。如果提升度等于1，则说明A和B是独立的，如果大于1，则说明两者之间存在正向关联，如果小于1，则说明两者之间存在负向关联。
"""