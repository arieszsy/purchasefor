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
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
# 不限制最大显示列数
pd.set_option('display.max_columns', None)
# 不限制最大显示行数
# pd.set_option('display.max_rows', None)

# 读取数据
Filepath="E:\\gittz\\purchasefor\\files\\train_set.csv"
TestFile="E:\\gittz\\purchasefor\\files\\test_set.csv"
df = pd.read_csv(Filepath, header=0)
df_test = pd.read_csv(TestFile, header=0)


# 分类变量转码
encoder = LabelEncoder()
labels=['job','marital','education','default','housing','loan','contact','poutcome']
for label in labels:
    df[label] = encoder.fit_transform(df[label].values)
    df_test[label] = encoder.fit_transform(df_test[label].values)
df['dd']='2014'+df['month']+df['day'].map(str)
df['dt']=df['dd'].apply(lambda x:datetime.datetime.strptime(x,"%Y%b%d"))
df_test['dd']='2014'+df_test['month']+df_test['day'].map(str)
df_test['dt']=df_test['dd'].apply(lambda x:datetime.datetime.strptime(x,"%Y%b%d"))

df['dif_day']=(max(df['dt'])-df['dt']).dt.days
df.insert(df.shape[1]-1,'y',df.pop('y'))
df=df.drop(['dd','day','month','ID','dt'],axis=1)
df_test['dif_day']=(max(df_test['dt'])-df_test['dt']).dt.days
df_test1=df_test.drop(['dd','day','month','ID','dt'],axis=1)
# print(df)
# 查看样本分布是否不平衡
x = df.iloc[:, :-1] #切片，得到输入x
y = df.iloc[:, -1] #切片，得到标签y
# groupby_data_orgianl = df.groupby('y')['y'].count() #对label做分类汇总
# print (groupby_data_orgianl) #打印输出原始数据集样本分类分布
# print(df_test)

#使用SMOTE方法进行过抽样处理
model_smotenc = SMOTENC(random_state=40,categorical_features=[1,2,3,4,6,7,8,13]) #建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smotenc.fit_resample(x,y) #输入数据并作过抽样处理
# 遍历所有列名，排除不需要的
# cols = [i for i in df.columns if i not in ['y']]
# x_smote_resampled = df[cols]
# print(x_smote_resampled)
# print(y_smote_resampled)
# print (y_smote_resampled.sum()/y_smote_resampled.count()) #打印输出原始数据集样本分类分布
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import warnings
# # 划分为5折交叉验证数据集
# df_y=data_all['status']
# df_X=data_all.drop(columns=['status'])
# df_X=scale(df_X,axis=0)  #将数据转化为标准数据,标准化函数

# 验证集处理
# lr = LogisticRegression(random_state=2018,tol=1e-6)  # 逻辑回归模型

tree = DecisionTreeClassifier(random_state=50) #决策树模型

# svm = SVC(probability=True,random_state=2018,tol=1e-6)  # SVM模型

forest=RandomForestClassifier(n_estimators=100,random_state=50) #　随机森林

Gbdt=GradientBoostingClassifier(random_state=50) #GBDT

Xgbc=XGBClassifier(random_state=50)  #Xgbc

gbm=lgb.LGBMClassifier(random_state=50)  #lgb
from sklearn.model_selection import train_test_split,cross_val_score	#划分数据 交叉验证
# train_X,test_X,train_y,test_y = train_test_split(x_smote_resampled,y_smote_resampled,test_size=1/3,random_state=3)
#这里划分数据以1/3的来划分 训练集训练结果 测试集测试结果
# k_range = range(1,31)
# cv_scores = []		#用来放每个模型的结果值
# for n in k_range:
#     knn = KNeighborsClassifier(n)   #knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
#     scores = cross_val_score(knn,train_X,train_y,cv=10,scoring='accuracy')  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值，具体使用参考下面。
#     cv_scores.append(scores.mean())
# plt.plot(k_range,cv_scores)
# plt.xlabel('K')
# plt.ylabel('Accuracy')		#通过图像选择最好的参数
# plt.show()
# best_knn = KNeighborsClassifier(n_neighbors=3)	# 选择最优的K=3传入模型
# best_knn.fit(train_X,train_y)			#训练模型
# print(best_knn.score(test_X,test_y))	#看看评分



dict_score=dict()
def muti_score(model,name):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, x_smote_resampled, y_smote_resampled, scoring='accuracy', cv=5)
    precision = cross_val_score(model, x_smote_resampled, y_smote_resampled, scoring='precision', cv=5)
    recall = cross_val_score(model, x_smote_resampled, y_smote_resampled, scoring='recall', cv=5)
    f1_score = cross_val_score(model, x_smote_resampled, y_smote_resampled, scoring='f1', cv=5)
    auc = cross_val_score(model, x_smote_resampled, y_smote_resampled, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())
    dict_score[name]=auc.mean()


model_name=["tree","forest","Gbdt","Xgbc","gbm"]
for name in model_name:
    model=eval(name)
    print(name)
    dict_score[name] = 0
    muti_score(model,name)


# print(dict_score)
model_name=max(dict_score,key=dict_score.get)

best_model=eval(model_name)
print(model_name)
b_model = best_model.fit(x_smote_resampled, y_smote_resampled)
predictions = b_model.predict(df_test1)
print(predictions)
output=pd.DataFrame({'id':df_test['ID'],'pre':predictions})
output.to_csv('E:\\gittz\\purchasefor\\files\\predictions.csv',index=False)



# # # 交叉验证
# lgb_train=lgb.Dataset(x_smote_resampled,y_smote_resampled)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#
#
#
# lgm=lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.05,n_estimators=20)
# X_test=
# y_test=
# gbm.fit(x_smote_resampled, y_smote_resampled,
#         eval_set=[(X_test, y_test)],
#         eval_metric='l1',
#         early_stopping_rounds=5)






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

