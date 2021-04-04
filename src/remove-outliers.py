import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
from sklearn import datasets
from scipy import stats
%matplotlib inline

# Load data and display numerical categories
df = pd.read_csv("./data-set-assignment1/adult.data-3.csv");
#df.describe()
#plt.boxplot(df['age'])
#plt.boxplot(df['hours-per-week'])

Q1a = df['age'].quantile(.25)
Q3a = df['age'].quantile(.75)
q1a = Q1a-1.5*(Q3a-Q1a)
q3a = Q3a+1.5*(Q3a-Q1a)

Q1h = df['hours-per-week'].quantile(.25)
Q3h = df['hours-per-week'].quantile(.75)
q1h = Q1h-1.5*(Q3h-Q1h)
q3h = Q3h+1.5*(Q3h-Q1h)

# Remove hours-per-week outliers
df1 = df[df['hours-per-week'].between(q1h, q3h)]
#plt.boxplot(df2['hours-per-week'])

# Remove age outliers
df2 = df1[df1['age'].between(q1a, q3a)]
#plt.boxplot(df1['age'])

# Save new data set
df2.to_csv('./data-set-assignment1/adult.data-3-cleaned.csv', index=False)