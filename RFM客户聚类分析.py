import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from datetime import datetime
from math import ceil

#  RFM模型
# R 最近一次消费  求出最近一次投资时间内距离提数日天数
# F 消费频率 月均投资次数
# M 消费金额 月均投资金额

# 1 导入数据
data = pd.read_excel('客户样本分析.xlsx',index_col='用户编码') # 将用户编码指定为用户编码
# print(data.head())  # 查看投10个文件，为serrise
# print(len(data)) # 总长度为200条

# 2 数据探索及预处理
# print(data.describe(include='all'))
# 提数日 2016/7/20
exdata_data = datetime(2016,7,20)
# print(exdata_data)

# R 最近一次投资时间内距提数日天数
diff_R = exdata_data - data['最近一次投资时间']
# print(diff_R)  # 类型为datetime 类型
# print(diff_R[0].days) # 24 int类型
R = []
for i in diff_R:
    R.append(i.days)
print(R[:10]) # [24, 24, 31, 34, 23, 28, 39, 22, 38, 25]


# 用户再投时长（月）
# 1 用户再投时长（天）
# 2 月= 天/30 向上取整
diff = exdata_data - data['首次投资时间']
# print(diff.head()) # 45B3CCE7-957B-4D54-9626-6D62731D119B   25 days
# print(diff[0].days) # 25
# print(diff[0].days/30) # 0.8333333333333334\
# print(ceil(diff[0].days/30)) # 1

# for循环获取相差天数，用户再投时长
diff_days = []
for i in diff:
    diff_days.append(i.days)
# print(diff_days[:10]) # [25, 24, 37, 34, 23, 28, 41, 22, 38, 25]

# 将天转换为月
diff_month = []
for i in diff_days:
    diff_month.append(ceil(i/30))
print(diff_month[:10]) # [1, 1, 2, 2, 1, 1, 2, 1, 2, 1]

# F 月均投资次数
F = (data['总计投标总次数']/diff_month).values
print(F[:10])  # [3.  3.  2.  1.5 3.  3.  1.5 3.  1.  3. ]

# 月均投资金额
M = (data['总计投资总金额']/diff_month).values
print(M[:10]) # [ 20000.  50000.  54000.  15000. 100000.  15000.  10000.  16000.   7500.20000.]


# 选取 R F M 三个指标作为聚类分析的指标
cdata = DataFrame([R,list(F),list(M)]).T
# print(cdata.head())
# 指定dataframe的index,coluens
cdata.index = data.index
# print(cdata.head())
cdata.columns = ['R-最近一次投资时间提数日天数','F-月均投资金额','M-月均投资金额']
# cdata.head()


# K-Means 聚类分析
cdata.mean()                        # R-最近一次投资时间提数日天数       22.66500                                   # F-月均投资金额               4.24250                                    # M-月均投资金额           35821.24175
print(cdata.mean())

# 数据标准化
zcdata = (cdata-cdata.mean())/cdata.std()
print(zcdata.head())

# sklearn 机器学习
from sklearn.cluster import KMeans

kmodel = KMeans(n_clusters=4,n_jobs=4,max_iter=100,random_state=0)
print(kmodel.fit(zcdata))
print(kmodel.labels_) # 类别

# 统计每个类别的频率
from pandas import Series
print(Series(kmodel.labels_).value_counts())

# 将类别标签返回元数据

cdata_res = pd.concat([cdata,Series(kmodel.labels_,index=cdata.index)],axis=1)
print(cdata_res)

# 命名最后一列为类别

cdata_res.columns = list(cdata.columns)+['类别']
print(cdata_res.head())

# 分组统计  R F M 指标的均值】
ccc = cdata_res.groupby(cdata_res['类别']).mean()
print(ccc)


