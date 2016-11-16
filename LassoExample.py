
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd

# read data
rawData = pd.read_csv('./data/nhanes3.dat',delim_whitespace = True,header=-1,na_values= ['.'])
rawData.columns = ['dummy','SEQN','SDPPSU6','SDPSTRA6','WTPFHX6','HSAGEIR','HSSEX','DMARACER','BMPWTLBS','BMPHTIN',
                   'PEPMNK1R','PEPMNK5R','HAR1','SMOKE','dummy2','TCP','HBP']

df = rawData.drop(['dummy','dummy2'],1)


print df.describe()
# handling nan variables by droping rows with Nan values
df_no_missing = df.dropna()
print df_no_missing.describe()

# SDPPSU6
df_no_missing['SDPPSU6'].apply(lambda x: x-1)

# DMARACER
df_no_missing['DMARACER'].apply(lambda x: x-1)

# HAR1
df_no_missing['HAR1'].apply(lambda x: x-1)

# SMOKE
df_no_missing['SMOKE'].apply(lambda x: x-1)

# creating dataset
feat_cols =  [col for col in df.columns if col not in ['SMOKE']]
X = [x for x in df_no_missing[feat_cols].values]
Y = [x for x in df_no_missing['SMOKE'].values]

# lasso reqularization with different regularization penalties
C =[0.001,0.01,0.1,1.0]
models = []
plotIndex = 1
for val in C:
    tmpModel = LogisticRegression(C=val, penalty='l1')
    tmpModel.fit(X,Y)
    # getting coefs
    coefs = tmpModel.coef_.ravel()
    # calculating sparsity ratio
    sparsityRatio = np.mean(coefs == 0) * 100
    # calculating accuracy
    accuracy = float(sum(np.equal(tmpModel.predict(X),Y))) / float(len(Y))

    # plotting results
    plots = plt.subplot(len(C) + 1,1,plotIndex)
    plots.imshow(np.abs(coefs.reshape(1,len(coefs))), interpolation='nearest', cmap='binary', vmax=1, vmin=0)
    plots.set_title("C = %.5f, Sparsity Ratio = %.4f, Acc = %.4f" % (val,sparsityRatio,accuracy))
    plotIndex = plotIndex + 1

plt.show()
