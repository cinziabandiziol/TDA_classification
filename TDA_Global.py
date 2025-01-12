import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from statistics import stdev

from Dataset import Dataset
from Dataset_fonderia import Dataset_fonderia
from TDAClassifier import TDAClassifier

##################### DOWNLOAD DATASET #####################
name = 'FONDERIA_Z1_1'
ds = Dataset_fonderia(name)
print(ds.data)
print(ds.target)
print('Download Data Completed')

##################### FILE TO WRITE #####################
#select = 'MAX'
#select = 'RANDOM'
#select = 'HAVG'
#select = 'MEDIAN'
select = 'HAVG'
file_name = './CONFRONTI_FONDERIA/Risultati_Global_' + name + '_' + select + '.csv'
report_file_name = './REPORT/Report_Global_' + name + '_' + select + '.txt'
write_flag = True

##################### BUILD FILTERED SIMPLICIAL COMPLEX #####################
label_complex = 'RIPS'

TDACl = TDAClassifier(ds.data,ds.target,ds.target_list,label_complex,ds.max_dim)
print('Creation of Simplicial Complex Completed')

TDACl.PruneFiltration(0,select)

expected_predicted_list = []
accuracy_list = []
confusion_matrix_list = []
recall_per_class_list = []
recall_avg_list = []

#################### CLASSIFICATION STEP #################

if ds.type == 'BALANCED':
    kf = KFold(n_splits = ds.n_fold, shuffle = True, random_state = 24)
elif ds.type == 'IMBALANCED':
    kf = StratifiedKFold(n_splits = ds.n_fold, shuffle = True, random_state = 24)

for iter, (index_train_set_list, index_test_set_list) in enumerate(kf.split(ds.data,ds.target)):

    print(f'Cross-Validation iteration {iter}')

    local_expected_label = []
    local_expected_predicted_list = []
    
    for val in index_test_set_list:
        
        ##################### LABELING FUNCTION #####################
        # predicted_label is a number
        predicted_label = TDACl.MakePrediction(val, index_test_set_list, ds.target_list)
        local_expected_label.append(ds.target[val])
        local_expected_predicted_list.append((ds.target[val],predicted_label))
        expected_predicted_list.append((ds.target[val],predicted_label))

    lc = [local_expected_predicted_list.count((i,i))/local_expected_label.count(i) for i in ds.target_list]
    recall_per_class_list.append(lc)
    recall_avg_list.append(np.mean(lc))

conf_matrix = np.reshape([expected_predicted_list.count((i,j)) for i in ds.target_list for j in ds.target_list],(len(ds.target_list),len(ds.target_list))).transpose()

average_recall = np.mean(recall_avg_list)
print(recall_avg_list)
std_recall = stdev(recall_avg_list)

if write_flag == True:
    df = pd.DataFrame()
    df['Recall'] = np.asarray(recall_avg_list)
    df['Method'] = 'Global'
    df['Selector'] = select
    df = df[['Method','Recall','Selector']]
    df.to_csv(file_name, sep = ';')
    print(df)
    report_file_name
    f = open(report_file_name, "w")
    f.write('The epsilon_k value is {}\n'.format(TDACl.pe))
    f.write('ico_Empty_Star: {}\n'.format(TDACl.ico_Empty_Star))
    f.write('ico_Link_1: {}\n'.format(TDACl.ico_Link_1))
    f.write('ico_Link_2: {}\n'.format(TDACl.ico_Link_2))
    f.write('ico_ico_KNN: {}\n'.format(TDACl.ico_KNN))
    f.write('ico_Gamma: {}\n'.format(TDACl.ico_Gamma))
    f.write('###### CLASSIFICATION REPORT ######\n')
    f.write('The dataset is {}\n'.format(ds.name))
    f.write('The dataset is {}\n'.format(ds.type))
    f.write('The confusion matrix is \n {}\n'.format(conf_matrix))
    for i in range(len(ds.target_list)):
        var = np.mean(list(list(zip(*recall_per_class_list))[i]))
        f.write('The average of recall of class {} is {}\n'.format(ds.target_list[i],var))

    f.write('The average of total recall is {}\n'.format(average_recall))
    f.write('The standard deviation of total recall is {}\n'.format(std_recall))
    f.close()

print('The epsilon_k value is ', TDACl.pe)
print('ico_Empty_Star: ', TDACl.ico_Empty_Star)
print('ico_Link_1: ', TDACl.ico_Link_1)
print('ico_Link_2: ', TDACl.ico_Link_2)
print('ico_ico_KNN: ', TDACl.ico_KNN)
print('ico_Gamma: ', TDACl.ico_Gamma)
print('###### CLASSIFICATION REPORT ######')
print('The dataset is ', ds.name)
print('The dataset is ', ds.type)
print('The confusion matrix is \n', conf_matrix)
for i in range(len(ds.target_list)):
    print('The average of recall of class ', ds.target_list[i], ' is ', np.mean(list(list(zip(*recall_per_class_list))[i])))

print('The average of total recall is ', average_recall)
print('The standard deviation of total recall is ', std_recall)
print('End script')











