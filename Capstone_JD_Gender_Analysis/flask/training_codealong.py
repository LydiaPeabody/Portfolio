#import statements and global constants/variables
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import dill

datafile_path = './titanic.csv'
pickle_file_path = './RandomForest_titanic.pkl'

#functions
#load data to pandas
def load_data():
    df = pd.read_csv(datafile_path)
    return df

#data cleaning and EDA
def EDA(df):
    include = ['Survived','Pclass','Sex','Age','SibSp','Fare']
    df['Sex'] = df['Sex'].apply(lambda x: 0 if x=='male' else 1)
    df = df[include].dropna()
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return (X, y)

#model fitting
def fit_model(X, y):
    PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X,y)
    return PREDICTOR

#Serialize/pickling (preserving training information)
def serialize(model):
    with open(pickle_file_path, 'wb') as f:
        dill.dump(model, f)
    print('Random Forest classifier trained on titanic dataset and pickled')

#main() functions
def main():
    try:
        df = load_data()
        X, y = EDA(df)
        model = fit_model(X, y)
        serialize(model)
    except Exception as err:
        print(err.args)
        exit

#program entry point
if __name__ == '__main__':
    main()
