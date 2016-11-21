import pandas as pd
import numpy as np
# getting titanic dataset
def getTitanicDataSet():
    # loading data

    rawData = pd.read_csv('./data/Titanic.csv', delimiter=',', index_col=0, header=0)
    df = rawData.drop(['Name', 'Ticket', 'Cabin'], 1)
    #print 'Titanic Dataset Description'
    #print df.describe()
    # handling nan variables by droping rows with Nan values
    df_no_missing = df.dropna()
    print 'Titanic Dataset Description after dropping rows with missing values'
    print df_no_missing.describe()

    print df_no_missing.dtypes

    # converting categorical data into numerical data
    df_no_missing['Gender'] = df_no_missing['Sex'].map({'female': 0, 'male': 1}).astype(int)

    Ports = list(enumerate(np.unique(df_no_missing['Embarked'])))  # determine all values of Embarked,
    Ports_dict = {name: i for i, name in Ports}
    df_no_missing.Embarked = df_no_missing.Embarked.map(lambda x: Ports_dict[x]).astype(int)

    # replacing categorical data with onehotencoding
    embarked_one_hot_coding = pd.get_dummies(df_no_missing['Embarked']).rename(columns=lambda x: 'Emb_' + str(x))
    pclass_one_hot_coding = pd.get_dummies(df_no_missing['Pclass']).rename(columns=lambda x: 'Pclass_' + str(x))

    # concat old df and onehotencodings
    df_no_missing_and_encoded = pd.concat([df_no_missing, embarked_one_hot_coding, pclass_one_hot_coding], axis=1)

    # print df_no_missing_and_encoded

    # drop unused cols and convert data to float
    df_no_missing_and_encoded_cleaned = df_no_missing_and_encoded.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_no_missing_and_encoded_cleaned = df_no_missing_and_encoded_cleaned.applymap(np.float)
    # print df_no_missing_and_encoded_cleaned

    # creating dataset
    feat_cols = [col for col in df_no_missing_and_encoded_cleaned.columns if col not in ['Survived', 'PassengerId']]
    X = [elem for elem in df_no_missing_and_encoded_cleaned[feat_cols].values]
    Y = [elem for elem in df_no_missing_and_encoded_cleaned['Survived'].values]
    return [X,Y,df_no_missing_and_encoded_cleaned.columns]