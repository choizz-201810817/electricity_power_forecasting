import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def makeChart(df, xColName='', yColName=''):
    plt.figure(figsize=(20,10))
    plt.title(f'{yColName} by {xColName}')
    sns.lineplot(df, x=xColName, y=yColName)
