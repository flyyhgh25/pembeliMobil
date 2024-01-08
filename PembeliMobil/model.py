import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
import pickle

class PrepoceesingData():
    def __init__(self,dataset=None, x_test=None, y_test=None,x_train=None, y_train=None):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
       
    def proses(self,dataset):
        self.dataset = pd.read_csv(dataset)
        numerical = [var for var in self.dataset.columns if self.dataset[var].dtype!='O']
        self.dataset[numerical].isnull().sum()
        categorical = [var for var in self.dataset.columns if self.dataset[var].dtype=='O']
        print(categorical)
        for var in categorical:
            print(var)
            print(self.dataset[var].value_counts())
            self.dataset['Memiliki_Mobil'].replace('?', np.NaN, inplace=True)
        self.dataset.Memiliki_Mobil.value_counts()
        print(self.dataset['Beli_Mobil'].value_counts(normalize=True))
        return self.dataset
    
    def DataSelection(self):
        x = self.dataset[['Usia', 'Status', 'Kelamin', 'Penghasilan', 'Memiliki_Mobil']] #features
        y = self.dataset['Beli_Mobil'] 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, test_size = 0.025, random_state=0)
        print(self.x_train, self.x_test, self.y_train, self.y_test)

    def MetodeRandomForestClassifier(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        classifier = RandomForestClassifier()
        classifier.fit(self.x_train,self.y_train)
        pickle.dump(classifier,open("modelRFPembeli.pkl","wb"))
    
    def MetodeTree(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
        clf_gini.fit(self.x_train, self.y_train)
        pickle.dump(clf_gini,open("modelTreePembeli.pkl","wb"))
    
    def MetodeKnn(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        knn_6 = KNeighborsClassifier(n_neighbors=6)
        knn_6.fit(self.x_train, self.y_train)
        pickle.dump(knn_6,open("modelKnnPembeli.pkl","wb"))

    def MetodeNaiveBayes(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        gnb = GaussianNB()
        gnb.fit(self.x_train, self.y_train)
        pickle.dump(gnb,open("modelNBPembeli.pkl","wb"))


    
# load data

# 
# X = datasets[['Usia', 'Status', 'Kelamin', 'Penghasilan', 'Memiliki_Mobil']] #features
# y = datasets['Beli_Mobil'] 

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.025, random_state=0)
# print(X_train, X_test, y_train, y_test)


# sc = StandardScaler()
# x_test = sc.fit_transform(X_train)
# x_test = sc.transform(X_test)

# # pasang model 
# classifier = RandomForestClassifier()

# # fit model
# classifier.fit(X_train,y_train)

# # ubah ke pkl
# pickle.dump(classifier,open("modelPembeli.pkl","wb"))