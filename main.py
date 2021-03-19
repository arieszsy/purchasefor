
from glob import glob
import pandas as pd
import os
# from PIL import Image
# from matplotlib import pyplot as plt
# from tqdm import tqdm
from imblearn.over_sampling import SMOTENC
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import *
import numpy as np
import time,datetime
from sklearn.preprocessing import LabelEncoder
# 不限制最大显示列数
pd.set_option('display.max_columns', None)
# 不限制最大显示行数
# pd.set_option('display.max_rows', None)

# 读取数据
Filepath="E:\\gittz\\purchasefor\\files\\train_set.csv"
df = pd.read_csv(Filepath, header=0)


# 分类变量转码
encoder = LabelEncoder()
labels=['job','marital','education','default','housing','loan','contact','poutcome']
for label in labels:
    df[label] = encoder.fit_transform(df[label].values)
df['dd']='2014'+df['month']+df['day'].map(str)
df['dt']=df['dd'].apply(lambda x:datetime.datetime.strptime(x,"%Y%b%d"))

df['dif_day']=(max(df['dt'])-df['dt']).dt.days
df.insert(df.shape[1]-1,'y',df.pop('y'))
df=df.drop(['dd','day','month','ID','dt'],axis=1)
# print(df)
# 查看样本分布是否不平衡
x = df.iloc[:, :-1] #切片，得到输入x
y = df.iloc[:, -1] #切片，得到标签y
# groupby_data_orgianl = df.groupby('y')['y'].count() #对label做分类汇总
# print (groupby_data_orgianl) #打印输出原始数据集样本分类分布

#使用SMOTE方法进行过抽样处理
model_smotenc = SMOTENC(random_state=40,categorical_features=[1,2,3,4,6,7,8,13]) #建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smotenc.fit_resample(x,y) #输入数据并作过抽样处理
# 遍历所有列名，排除不需要的
# cols = [i for i in df.columns if i not in ['y']]
# x_smote_resampled = df[cols]
# print(x_smote_resampled)
print(y_smote_resampled)
print (y_smote_resampled.sum()/y_smote_resampled.count()) #打印输出原始数据集样本分类分布






# 变量相关性分析

# fig = plt.figure()
# names=df.columns.tolist()
# #fig.figsize:(40,40) #图片大小为20*20
# # 以下代码用户显示中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus']=False
#
# plt.figure(figsize=(8,6))
# ax = sns.heatmap(df,cmap=plt.cm.Greys, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
# plt.xticks(np.arange(5)+0.5,names) #横坐标标注点
# plt.yticks(np.arange(5)+0.5,names) #纵坐标标注点
# plt.show()

#使用SMOTE方法进行过抽样处理
# model_smote = SMOTENC(random_state=40,categorical_features=) #建立SMOTE模型对象
# x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y) #输入数据并作过抽样处理
# # 遍历所有列名，排除不需要的
# cols = [i for i in df.columns if i not in ['y']]
# x_smote_resampled = df[cols]
# print(x_smote_resampled)
#
#
# x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=[]) #将数据转换为数据框并命名列名
#
# y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['label']) #将数据转换为数据框并命名列名
#
# smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) #按列合并数据框
#
# groupby_data_smote = smote_resampled.groupby('label').count() #对label做分类汇总
#
# print (groupby_data_smote) #打印输出经过SMOTE处理后的数据集样本分类分布

