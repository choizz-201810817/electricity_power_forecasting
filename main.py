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

# powercompDf.to_sql(name='powercomp', con=db_connection, if_exists='append', index=False)
# proc_dataDf.to_sql(name='procdata', con=db_connection, if_exists='append', index=False)
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
