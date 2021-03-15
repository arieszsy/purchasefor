
from glob import glob
import pandas as pd
import numpy as np
import os
# from PIL import Image
# from matplotlib import pyplot as plt
# from tqdm import tqdm
from imblearn.over_sampling import SMOTENC


Filepath="E:\\gittz\\purchasefor\\files\\train_set.csv"
df = pd.read_csv(Filepath, header=0)

# 查看样本分布是否不平衡
x = df.iloc[:, :-1] #切片，得到输入x
y = df.iloc[:, -1] #切片，得到标签y
groupby_data_orgianl = df.groupby('y')['y'].count() #对label做分类汇总
print (groupby_data_orgianl) #打印输出原始数据集样本分类分布

#使用SMOTE方法进行过抽样处理
model_smote = SMOTENC(random_state=40,categorical_features=) #建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y) #输入数据并作过抽样处理
# 遍历所有列名，排除不需要的
cols = [i for i in df.columns if i not in ['y']]
x_smote_resampled = df[cols]
print(x_smote_resampled)


x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=[]) #将数据转换为数据框并命名列名

y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['label']) #将数据转换为数据框并命名列名

smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) #按列合并数据框

groupby_data_smote = smote_resampled.groupby('label').count() #对label做分类汇总

print (groupby_data_smote) #打印输出经过SMOTE处理后的数据集样本分类分布

