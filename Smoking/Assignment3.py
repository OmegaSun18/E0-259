import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

filename = 'Raw Data_GeneSpring.txt'
smoking_data = pd.read_csv(filename, sep='\t')

#Drop all rows with GeneSymbol = NaN
smoking_data = smoking_data.dropna(subset=['GeneSymbol'])

#Drop all rows with EntrezGeneID = NaN
smoking_data = smoking_data.dropna(subset=['EntrezGeneID'])

#creating N and D arrays
N = []
D = []
for j in range(0, 48):
    if j < 12:
        N.append([1, 0, 1, 0])
        D.append([1, 0, 0, 0])
    elif j < 24:
        N.append([1, 0, 0, 1])
        D.append([0, 1, 0, 0])
    elif j < 36:
        N.append([0, 1, 1, 0])
        D.append([0, 0, 1, 0])
    else:
        N.append([0, 1, 0, 1])
        D.append([0, 0, 0, 1])
N = np.array(N)
D = np.array(D)

#Initializing variables
rank_N = 3
rank_D = 4
n = 48

#Calculating p values
p_values = []
for i in range(len(smoking_data)):
    X = smoking_data.iloc[i, 1:49].values
    X = list(X)
    for i in range(len(X)):
        X[i] = math.pow(2, X[i])
    X = np.array(X)
    X = X.reshape((48, 1))
    #For each value of X do 2^X
    num = np.dot(np.dot(N, np.linalg.pinv(np.dot(N.T, N))), N.T)
    den = np.dot(np.dot(D, np.linalg.pinv(np.dot(D.T, D))), D.T)
    I = np.eye(48)
    f_statistic = ((n - rank_D)/(rank_D - rank_N)) * ((np.dot(X.T, np.dot(I - num, X))/np.dot(X.T, np.dot(I - den, X))) - 1)
    # #Get p value
    # print(f"f_statistic is {f_statistic}")
    p_value = 1 - stats.f.cdf(f_statistic.item(), rank_D - rank_N, n - rank_D)
    p_values.append(p_value)

#Plotting p values
sns.histplot(p_values, bins=100)

plt.xlabel('P-Value')
plt.ylabel('Frequency')
plt.title('P-Value Distribution')

plt.savefig('P-Value_Distribution.png')
