from dataprep.eda import create_report
import pandas as pd
import numpy as np

# dataprep으로 eda 파일 저장
def createPrepReport(df, reportName):
    report = create_report(df)
    report.save(f'{reportName}.html')

# datetime 생성
def makeDateTime(df, colName=''):
    df[colName] = pd.to_datetime(df[colName])

    # 파생 변수 생성
    df['year'] = df[colName].dt.year
    df['month'] = df[colName].dt.month
    df['day'] = df[colName].dt.day
    df['hour'] = df[colName].dt.hour
    df['dayname'] = df[colName].dt.day_name()

    return df

# 컬럼의 value들 등급 나누기
def column2cut(df, colName='', kind='cut', classNum=5):
    labels = [i+1 for i in range(classNum)]
    if kind=='cut':
        df[colName+'_grade'] = pd.cut(df[colName], classNum, labels=labels)
        return df
    else:
        df[colName+'_gradeQ'] = pd.qcut(df[colName], classNum, labels=labels)
        return df
