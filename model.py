import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
import pickle

class DataMining():
    def __init__(self, x_test=None, y_test=None,x_train=None, y_train=None):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
       
    def PrepocessingData(self,dataset):
        self.datasets = pd.read_csv(dataset)
        # Mencari nilai kosong (?)
        self.datasets.isin(['?']).any()
        # ganti nilai ? menjadi NaN
        self.datasets['Memiliki_Mobil'].replace('?', np.NaN, inplace=True)
        # mengisi
        self.datasets['Memiliki_Mobil'].fillna(self.datasets['Memiliki_Mobil'].mode()[0], inplace=True)
        return self.datasets
    
    def ScaleData(self):
        # standarS = StandardScaler()
        robustS = RobustScaler()
        self.x_test = robustS.fit_transform(self.x_train)
        self.x_test = robustS.transform(self.x_test)

    def DataSelection(self):
        data = self.datasets[['Usia','Kelamin', 'Penghasilan', 'Memiliki_Mobil']] #features
        target = self.datasets['Beli_Mobil'] 
        self.categorical_columns = ['Memiliki_Mobil']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data,target, test_size = 0.30, random_state=90)

        # XGBoost
    def MetodeXGBoost(self):
        self.DataSelection()
        self.x_train[self.categorical_columns] = self.x_train[self.categorical_columns].astype('category')
        self.x_test[self.categorical_columns] = self.x_test[self.categorical_columns].astype('category')
        self.ScaleData()
        xgb = XGBClassifier(n_estimators=100, enable_categorical=True)
        xgb.fit(self.x_train, self.y_train)
        pickle.dump(xgb,open("model/modelXgBostPembeli.pkl","wb"))

        # Random Forest
    def MetodeRandomForest(self):
        self.DataSelection()
        self.x_train[self.categorical_columns] = self.x_train[self.categorical_columns].astype('category')
        self.x_test[self.categorical_columns] = self.x_test[self.categorical_columns].astype('category')
        self.ScaleData()
        rf = RandomForestClassifier()
        rf.fit(self.x_train,self.y_train)
        pickle.dump(rf,open("model/modelRFPembeli.pkl","wb"))

        # NAIVE BAYES
    def MetodeNaiveBayes(self):
        self.DataSelection()
        self.ScaleData()
        gnb = GaussianNB()
        gnb.fit(self.x_train, self.y_train)
        pickle.dump(gnb,open("model/modelNBPembeli.pkl","wb"))