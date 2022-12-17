#%%
import pandas as pd
import numpy as np

import pymysql
from sqlalchemy import create_engine
import sqlalchemy

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep.eda import create_report

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import express as px

import warnings
warnings.filterwarnings('ignore')

#%%
# powercompDf = pd.read_csv('data\powercomp.csv')
# proc_dataDf = pd.read_csv('data\proc_data.csv')

# # DataFrame을 mysql의 electricitydb에 
# db_connection_str = 'mysql+pymysql://root:949700@localhost/electricitydb'
# db_connection = create_engine(db_connection_str)
# conn = db_connection.connect()

# powercompDf.to_sql(name='powercomp', con=db_connection, if_exists='fail', index=False)
# proc_dataDf.to_sql(name='procdata', con=db_connection, if_exists='fail', index=False)
# %%
# mySql 연동

# conn = pymysql.connect(host='localhost', user='root', passwd='949700', df='elecricitydb', charset='utf8')
# cur = conn.cursor()

conn = pymysql.connect(host='localhost', user='root', passwd='949700', db='electricitydb', charset='utf8')
cur = conn.cursor()

powercompDf = pd.read_sql('SELECT * FROM powercomp', con=conn)
proc_dataDf = pd.read_sql('SELECT * FROM procdata', con=conn)

# %%
print(powercompDf.shape)
print(proc_dataDf.shape)
# %%
powerReport = create_report(powercompDf)
powerReport.save('powercomp_summary.html')

procReport = create_report(proc_dataDf)
procReport.save('proc_summary.html')

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
df = powercompDf.iloc[:, :-2]
print(df.shape)

# %%
# 컬럼명 소문자로 변경
df.columns = df.columns.str.lower()
df
# %%
# 결측치 확인
df.isna().sum()

# %%
# 레이블링 & 더미 처리
df['datetime'] = pd.to_datetime(df['datetime'])
df['datetime']

# 파생 변수 생성
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayname'] = df['datetime'].dt.day_name()
df.head()

# %%
labels = [1,2,3,4,5]
df['humidity_grade_q'] = pd.qcut(df['humidity'], 5, labels=labels)
df['humidity_grade'] = pd.cut(df['humidity'], 5, labels=labels)
df

# %%
print(df.humidity_grade.value_counts())
print(df.humidity_grade_q.value_counts())

# %%
# 시간별, 월별, 요일별 전력 수요량
plt.figure(figsize=(20,10))
plt.title('zone 1 power consumption by month')
sns.lineplot(df, x='month', y='zone 1 power consumption')

plt.figure(figsize=(20,10))
plt.title('zone 1 power consumption by hour')
sns.lineplot(df, x='hour', y='zone 1 power consumption')

plt.figure(figsize=(20,10))
plt.title('zone 1 power consumption by dayname')
sns.lineplot(df, x='dayname', y='zone 1 power consumption')

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
mask = np.zeros_like(df1, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df1.corr(), linewidths=0.5)
# %%
cols = ['hour', 'temperature']
for i, col in enumerate(cols):
    plt.subplot(1,2,i+1)
    plt.title(f"{col}'s boxplot")
    sns.boxplot(df[col])

# %%
# 스케일링
# datetime, year, qcut -> 제거
# dayname, cut(grade) -> onehot encoding
df = df.drop(['datetime', 'year', 'humidity_qcut', 'humidity_cut', 'humidity_grade_q'], axis=1)

# %%
df = pd.concat([df, pd.get_dummies(df.dayname)], axis=1).drop(['dayname'], axis=1)
# %%
df = pd.concat([df, pd.get_dummies(df.humidity_grade)], axis=1).drop(['humidity_grade'], axis=1)
#%%
from sklearn.preprocessing import MinMaxScaler

df2 = pd.concat([df.iloc[:,:5], df.iloc[:,6:]], axis=1)
mm_sc = MinMaxScaler()
mm_df = pd.concat([pd.DataFrame(mm_sc.fit_transform(df2), columns=df2.columns), df['zone 1 power consumption']], axis=1)
# %%

# %%
