#%%
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from plotly import express as px

import pandas as pd
import numpy as np

from mysqlModule import df2table, table2df
from preproModule import createPrepReport, makeDateTime, column2cut
from visuModule import makeChart
from mlModeDefine import trainTestAlgo, learningCurveDraw

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


#%%
# csv파일 dataframe으로 불러오기
powercompDf = pd.read_csv('data\powercomp.csv')
proc_dataDf = pd.read_csv('data\proc_data.csv')

# # DataFrame을 mysql의 electricitydb에 table로 넣기
# df2table(powercompDf, 'root', '?', 'electricitydb', 'powercomp')
# df2table(proc_dataDf, 'root', '?', 'electricitydb', 'procdata')


# %%
# mySql 연동하여 table을 dataframe으로 가져오기
powercompDf = table2df('root', '?', 'electricitydb', 'powercomp')
proc_dataDf = table2df('root', '?', 'electricitydb', 'procdata')

print(powercompDf.shape)
print(proc_dataDf.shape)


# %%
# # dataprep으로 eda
# createPrepReport(powercompDf, 'powercomp')
# createPrepReport(proc_dataDf, 'procdata')


# %%
##### 결측 처리 : 없음(완료)
##### 레이블링, 더미 처리
##### 파생 변수 
##### 데이터 스케일링 (dayname 더미 변환)
# 컬럼 제거 - datetime, year, humidity_grade_q
# 더미 처리 - dayname, humidity_grade
# 로그 변환 - general diffuse flows, diffuse flows
# 스케일링 - target빼고 모두
##### 데이터 분리
##### 모델 정의 및 모델 학습 및 평가
##### 하이퍼 파라미터 튜닝
##### 최종결과 산출


#%%
# target값은 zone 1만 사용
df = powercompDf.iloc[:, :-2]
print(df.shape)


# %%
# 컬럼명 소문자로 변경
df.columns = df.columns.str.lower()
df


# %%
# 결측치 확인 - 없음
df.isna().sum()


# %%
# datetime 컬럼을 datetime type으로 변경 및 날짜 파생 변수 생성
df = makeDateTime(df, 'datetime')


# %%
df = column2cut(df, 'humidity')
print(df.humidity_grade.value_counts())


# %%
# 시간별, 월별, 요일별 전력 수요량
cols = ['month', 'dayname', 'hour']

for col in cols:
    makeChart(df, col, 'zone 1 power consumption')
    

#%%
## 요일별 시간대 전력수요량
# 새벽과 늦은 밤은 전력 수요량이 요일과 관계없이 비슷하지만, 
# 업무시간에는 주말과 주중의 전력수요량 차이가 큼

plt.figure(figsize=(20,10))
plt.title('zone 1 power consumption by hour (hue is dayname)')
sns.lineplot(df, x='hour', y='zone 1 power consumption', hue='dayname', ci=None)


# %%
# 상관관계 파악
# hour와 temperature가 타겟값과 상관계수가 높음..
df1 = df[['temperature', 'humidity_grade', 'wind speed', 'general diffuse flows', 'diffuse flows', 'zone 1 power consumption', 'month', 'hour']]
mask = np.zeros_like(df1.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df1.corr(), linewidths=0.5, mask=mask, cmap='coolwarm_r')


# %%
cols = ['hour', 'temperature']
for i, col in enumerate(cols):
    plt.subplot(1,2,i+1)
    plt.title(f"{col}'s boxplot")
    sns.boxplot(df[col])


# %%
# 스케일링
# datetime, year -> 제거
# dayname, cut(grade) -> onehot encoding
df = df.drop(['datetime', 'year'], axis=1)


# %%
# 범주형 변수 one hot encoding
df = pd.concat([df, pd.get_dummies(df.dayname)], axis=1).drop(['dayname'], axis=1)
df = pd.concat([df, pd.get_dummies(df.humidity_grade)], axis=1).drop(['humidity_grade'], axis=1)


#%%
# 타겟 변수 외 나머지 컬럼 정규화(Min Max Scaling)
df2 = pd.concat([df.iloc[:,:5], df.iloc[:,6:-12]], axis=1)
mm_sc = MinMaxScaler()
mm_df = pd.DataFrame(mm_sc.fit_transform(df2), columns=df2.columns)
mm_df1 = pd.concat([mm_df,df.iloc[:,9:]], axis=1)
mm_df2 = pd.concat([mm_df1,df['zone 1 power consumption']], axis=1)


#%%
# int형 컬럼명 str로 변경
mm_df2.rename(columns={1:'grade_1',
                       2:'grade_2',
                       3:'grade_3',
                       4:'grade_4',
                       5:'grade_5'}, inplace=True)


# %%
# 데이터 정의
X = mm_df2.iloc[:,:-1]
y = mm_df2.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# %%
# 모델 정의
rf_rg = RandomForestRegressor()
ln_rg = LinearRegression()
xgb_rg = XGBRegressor()
lgb_rg = LGBMRegressor()

algos = [rf_rg, ln_rg, xgb_rg, lgb_rg]

for algo in algos:
    trainTestAlgo(X_train, X_test, y_train, y_test, algo)


# %%
# 각 모델별 learning curve 그려보기
plt.figure(figsize=(20,20))
for i, algo in enumerate(algos):
    plt.subplot(2,2,i+1)
    learningCurveDraw(algo, X_train, y_train, 1.0, 10)

