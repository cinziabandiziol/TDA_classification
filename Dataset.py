import numpy as np
import sklearn.datasets
import torchvision.datasets as datasets
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn import preprocessing
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class Dataset:
    def __init__(self,label):
        self.label = label
        self.CreateDataset()

    def CreateDataset(self):
        if self.label == 'IRIS':
            iris_dataset = sklearn.datasets.load_iris()
            self.data = iris_dataset.data
            self.target = list(iris_dataset.target)
            self.target_list = [0,1,2]
            self.K = 50
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 15
            self.name = str('Iris Dataset')
            self.max_dim = 3
            self.type = 'BALANCED'
        elif self.label == 'MOON':
            self.data, target = sklearn.datasets.make_moons(n_samples=200, noise=10, random_state = 24)
            self.target = list(target)
            self.target_list = [0,1]
            self.K = 50
            self.n_fold = 20
            self.n_size = int(len(self.data)/self.n_fold) # 15
            self.name = str('Moon Dataset')
            self.max_dim = 2
            self.type = 'BALANCED'
        elif self.label == 'CIRCLES':
            self.data, target = sklearn.datasets.make_circles(n_samples=50, noise=3, random_state = 24)
            self.target = list(target)
            self.target_list = [0,1]
            self.K = 30
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 15 
            self.name = str('Circles Dataset')
            self.max_dim = 2
            self.type = 'BALANCED'
        elif self.label == 'CIFAR-10':
            cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
            #cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
            print(np.asarray(cifar_trainset[1][0])[4])
            list_images = [np.asarray(cifar_trainset[k][0])[j] for k in range(50000) for j in range(32)]
            list_target = np.array([np.asarray(cifar_trainset)[k][1]] for k in range(50000))
            self.data = [list_images[i].flatten() for i in range(len(list_images))]
            print(list_images)
            self.target = iris_dataset.target
            self.target_list = [0,1,2]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 15
            self.name = str('Iris Dataset')
            self.max_dim = 3
            self.type = 'BALANCED'
        elif self.label == 'DIABETIC_RETINOPATHY':
            dataset = fetch_ucirepo(id=329) # total: 1080 (540/540)
            # data (as pandas dataframes) 
            X = dataset.data.features # 19 features (Binay, integer and continous)
            y_old = dataset.data.targets
            df_total = pd.concat([X, y_old], axis=1)
            df_total_0 = df_total[df_total['Class']==0]
            df_total_1 = df_total[df_total['Class']==1].head(540)
            df_total_new = pd.concat([df_total_0, df_total_1], axis=0)
            df_total_new.sort_index(inplace = True)
            data_old = df_total_new.drop(['Class'], axis=1)
            data = [np.array(row) for row in data_old.values]
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data)
            self.target = list(df_total_new['Class'])
            self.target_list = [0,1]
            self.K = 100
            self.n_fold = 20
            self.n_size = int(len(self.data)/self.n_fold) # 15
            self.name = str('Diabetic Retinopathy')
            self.max_dim = 2
            self.type = 'BALANCED'
        elif self.label == 'OPTICAL_RECOGNITION':
            dataset = fetch_ucirepo(id=80)# about 550 for each class (total 5620)  
            # data (as pandas dataframes) 
            X = dataset.data.features # 64 features (all integers between 0 and 16)
            Y = dataset.data.targets
            self.data = [np.array(row) for row in X.values]
            self.target = list(Y['class'])
            self.target_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.K = 200
            self.n_fold = 20
            self.n_size = int(len(self.data)/self.n_fold) # 15
            self.name = str('Optical Recognition')
            self.max_dim = 2
            self.type = 'BALANCED'
        elif self.label == 'RICE':
            dataset = fetch_ucirepo(id=545)# total: 3260 (1630/1630)  
            X = dataset.data.features # 7 features (all real numbers)
            y_old = dataset.data.targets
            df_total = pd.concat([X, y_old], axis=1)
            df_total.replace(['Cammeo','Osmancik'], [0,1], inplace = True)
            df_total_0 = df_total[df_total['Class']==0]
            df_total_1 = df_total[df_total['Class']==1].head(1630)
            df_total_new = pd.concat([df_total_0, df_total_1], axis=0)
            df_total_new.sort_index(inplace = True)
            data_old = df_total_new.drop(['Class'], axis=1)
            data = [np.array(row) for row in data_old.values]
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data)
            self.target = list(df_total_new['Class'])
            self.target_list = [0,1]
            self.K = 100
            self.n_fold = 20
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Rice')
            self.max_dim = 2
            self.type = 'BALANCED'    
        elif self.label == 'CANCER':
            cancer_dataset = sklearn.datasets.load_breast_cancer() # total: 570 (213/357)
            data_old = cancer_dataset.data # 30 features (all real)
            len_data = data_old.shape # It is a tuple, access to elemtns using indexes from 0
            self.data = np.append(cancer_dataset.data,np.array([data_old[len_data[0]-1]]), axis = 0)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(np.append(data_old,np.array([data_old[len_data[0]-1]]), axis = 0))
            target_old = cancer_dataset.target
            self.target = list(np.append(target_old, np.array([target_old[len_data[1]-1]]), axis=0))
            self.target_list = [0,1]
            self.K = 100
            self.n_fold = 30
            self.n_size = int(len(self.data)/self.n_fold) # 15
            self.name = str('Breast Cancer Dataset')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'WINE':
            wine_dataset = sklearn.datasets.load_wine() # total: 178 (59/71/48)
            data = wine_dataset.data # 13 features (all real)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data)
            self.target = list(wine_dataset.target)
            self.target_list = [0,1,2]
            self.K = 50
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold)
            self.name = str('Wine Dataset')
            self.max_dim = 5
            self.type = 'IMBALANCED'
        elif self.label == 'SURGERY':
            dataset = fetch_ucirepo(id=277) # total: 470 (70(T)/400(F))
            X_old = dataset.data.features # 7 features (integer, real)
            X = X_old.replace(['F','T','DGN1','DGN2','DGN3','DGN4','DGN5','DGN6','DGN8','PRZ0','PRZ1','PRZ2','OC11','OC12','OC13','OC14'], [0,1,0,1,2,3,4,5,6,0,1,2,0,1,2,3])
            y_old = dataset.data.targets
            y = y_old.replace(['F','T'], [0,1])
            data = [np.array(row) for row in X.values]
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data)
            self.target = list(y['Risk1Yr'])
            self.target_list = [0,1]
            self.K = 100
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold)
            self.name = str('Thoracic Surgery Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'LIVER':
            dataset = fetch_ucirepo(id=225) # total: 580 (413/167) 
            X_old = dataset.data.features # 10 features (integer, real)
            y = dataset.data.targets
            X = X_old.replace(['Male','Female'], [0,1]).fillna(0)
            df_total = pd.concat([X, y], axis=1)
            X_1 = df_total[df_total['Selector']==1].head(413)
            X_2 = df_total[df_total['Selector']==2]
            df_total_new = pd.concat([X_1, X_2], axis=0)
            df_total_new.sort_index(inplace = True)
            data_old = df_total_new.drop(['Selector'], axis=1)
            data = [np.array(row) for row in data_old.values]
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data)
            self.target = list(df_total_new['Selector'])
            self.target_list = [1,2]
            self.K = 100
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Indian Liver Patient Dataset')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z1_1':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['te2_mm','te2_s-ts2_s','te2_mm-ts2_mm','fase1_min','fase2_min','fase3_media','ACC_max_ms2','TP2_hammer','TC1_3c','Z1_1']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[(dataset1['Z1_1'] != 4) & (dataset1['Z1_1'] != 5)]
            scaler = preprocessing.MinMaxScaler()
            data = dataset.drop(['Z1_1'],axis=1).to_numpy()
            self.data = scaler.fit_transform(data)
            self.target = dataset['Z1_1'].to_numpy()
            self.target_list = [1,2,3]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z1_2':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TP2_hammer','TP2_5b','TC1_3c','TC6_3c','te2_mm','fase2_min','Fl_f','TC3_3c','ACC_min_ms2','fase1_min','Z1_2']
            dataset1 = dataset_originale[columns_list]
            data_old = dataset1.drop(['Z1_2'],axis=1).to_numpy()
            target_old = dataset1['Z1_2'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 140, 2: 102, 3: 30, 4: 24, 5: 24}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [1,2,3,4,5]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z1_3':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TC5_3c','ACC_min_ms2','CP_max_b','Fl_f','TP2_3b','TC4_3c','TP2_hammer','ACC_max_s','TC1_1s','TC1_4s','Z1_3']
            dataset = dataset_originale[columns_list]
            data = dataset.drop(['Z1_3'],axis=1).to_numpy()
            target = dataset['Z1_3'].to_numpy()
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data)
            self.target = target
            self.target_list = [1,2,3]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold)
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z2_2':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TC1_3c','TP2_5b','te2_mm-ts2_mm','TC6_3c','TP2_Ate3','fase3_max','HP_max_b','fase1_min','te2_s-ts2_s','TP2_A','Z2_2']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['Z2_2'] != 1]
            data_old = dataset.drop(['Z2_2'],axis=1).to_numpy()
            target_old = dataset['Z2_2'].to_numpy()
            smote = SMOTE(sampling_strategy = {2: 188, 3: 57, 4: 30, 5: 30}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [2,3,4,5]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z2_3':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TC3_3c','TC6_3c','te3_s-te2_s','Fl_f','fase3_max','te2_mm-ts2_mm','TP2_3b','fase1_max','TC7_3c','ACC_min_ms2','Z2_3']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['Z2_3'] < 3]
            scaler = preprocessing.MinMaxScaler()
            data = dataset.drop(['Z2_3'],axis=1).to_numpy()
            self.data = scaler.fit_transform(data)
            self.target = dataset['Z2_3'].to_numpy()
            self.target_list = [1,2]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z3_1':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['te2_mm','TC1_3c','fase1_max','fase3_max','Fl_f','TC1_1s','TC1_4s','TC2_1s','TC2_4s','TC3_1s','Z3_1']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['Z3_1'] < 4]
            data_old = dataset.drop(['Z3_1'],axis=1).to_numpy()
            target_old = dataset['Z3_1'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 89, 2: 184, 3: 30}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [1,2,3]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z3_3':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TP2_A','TC1_3c','TC7_3c','TP2_3b','TP2_5b','te2_s-ts2_s','Li','HP_max_b','fase3_media','fase3_max','Z3_3']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['Z3_3'] != 5]
            data_old = dataset.drop(['Z3_3'],axis=1).to_numpy()
            target_old = dataset['Z3_3'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 54, 2: 170, 3: 58, 4: 20}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [1,2,3,4]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z4_1':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['Fl_f','TC3_3c','TC5_3c','TC6_3c','fase2_min','TP2_3b','TP2_hammer','TC7_3c','Li','te2_mm','Z4_1']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['Z4_1'] != 5]
            data_old = dataset.drop(['Z4_1'],axis=1).to_numpy()
            target_old = dataset['Z4_1'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 30, 2: 168, 3: 90, 4: 20}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [1,2,3,4]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_Z4_3':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TP2_Ate3','te2_mm-ts2_mm','ACC_max_s','Fl_f','TP2_A','fase1_min','ACC_max_ms2','Li','fase3_max','fase1_max','Z4_3']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['Z4_3'] < 4]
            data_old = dataset.drop(['Z4_3'],axis=1).to_numpy()
            target_old = dataset['Z4_3'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 70, 2: 207, 3: 20}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [1,2,3]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_PIN1':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['ACC_max_s','TC1_3c','fase1_max','TC3_3c','TC6_3c','TP2_hammer','te2_mm','TC5_3c','TP2_Ate3','fase1_min','PIN1']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['PIN1'] < 3]
            scaler = preprocessing.MinMaxScaler()
            data = dataset.drop(['PIN1'],axis=1).to_numpy()
            self.data = scaler.fit_transform(data)
            self.target = dataset['PIN1'].to_numpy()
            self.target_list = [1,2]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_PIN2':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['TP2_Ate3','TP2_A','fase3_media','TC3_3c','TC6_3c','TP2_3b','TP2_5b','te3_s-te2_s','ACC_max_ms2','Fl_f','PIN2']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['PIN2'] < 4]
            data_old = dataset.drop(['PIN2'],axis=1).to_numpy()
            target_old = dataset['PIN2'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 223, 2: 59, 3: 20}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(data_resapled)
            self.target = target_resampled
            self.target_list = [1,2,3]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        elif self.label == 'FONDERIA_PIN3':
            dataset_originale = pd.read_excel("Dati_fonderia.xlsx")
            columns_list = ['fase1_max','HP_max_b','te3_s-te2_s','TP2_Ate3','ACC_max_s','TP2_A','te2_s-ts2_s','Li','fase3_max','CP_max_b','PIN3']
            dataset1 = dataset_originale[columns_list]
            dataset = dataset1[dataset1['PIN3'] != 5]
            data_old = dataset.drop(['PIN3'],axis=1).to_numpy()
            target_old = dataset['PIN3'].to_numpy()
            smote = SMOTE(sampling_strategy = {1: 225, 2: 35, 3: 25, 4: 20}, random_state=42, k_neighbors = 3)
            data_resapled, target_resampled = smote.fit_resample(data_old, target_old)
            rus = RandomUnderSampler(sampling_strategy = {1: 120, 2: 35, 3: 25, 4: 20}, random_state = 42)
            X_res, y_res = rus.fit_resample(data_resapled, target_resampled)
            scaler = preprocessing.MinMaxScaler()
            self.data = scaler.fit_transform(X_res)
            self.target = list(y_res)
            self.target_list = [1,2,3,4]
            self.n_fold = 10
            self.n_size = int(len(self.data)/self.n_fold) # 163
            self.name = str('Foundrys Data')
            self.max_dim = 2
            self.type = 'IMBALANCED'
        else:
            raise TypeError("Name of Dataset not found!!")