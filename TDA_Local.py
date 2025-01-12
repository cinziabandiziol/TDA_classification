import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from statistics import stdev

from Dataset import Dataset
from Dataset_fonderia import Dataset_fonderia
from TDAClassifier import TDAClassifier

##################### DOWNLOAD DATASET #####################
name = 'FONDERIA_PIN2_d'
ds = Dataset_fonderia(name)
K = 100

##################### BUILD FILTERED SIMPLICIAL COMPLEX #####################
label_complex = 'RIPS'
select = 'HAVG'
# MAX
# RANDOM
# HAVG
# MEDIAN
# AVG

##################### FILE TO WRITE #####################
file_name = './CONFRONTI_FONDERIA/Risultati_Local_' + name + '_' + select + '_K' + str(K) + '.csv'
write_flag = True

#################### CLASSIFICATION STEP #################

expected_predicted_list = []
confusion_matrix_list = []
recall_per_class_list = []
recall_avg_list = []

Empty_Star_count = 0
Link_1_count = 0
Link_2_count = 0
KNN_count = 0
Gamma_count = 0
pe_list = []

index_list = list(range(len(ds.data)))
#K = ds.K

if ds.type == 'BALANCED':
    kf = KFold(n_splits = ds.n_fold, shuffle = True, random_state = 24)
elif ds.type == 'IMBALANCED':
    kf = StratifiedKFold(n_splits = ds.n_fold, shuffle = True, random_state = 24)

for iter, (index_train_set_list, index_test_set_list) in enumerate(kf.split(ds.data,ds.target)):
    print(f'Cross-Validation iteration {iter}')

    data_copy = np.copy(ds.data)
    train_data = data_copy[index_train_set_list]
    # list of tuples (array,index from original dataset)
    train_data_extended = list(zip(train_data,index_train_set_list))

    local_expected_label = []
    local_expected_predicted_list = []

    for val in index_test_set_list:
        
        # print('Il vertice è ', val)
        dist_list = [(np.linalg.norm(train_data[i] - ds.data[val]), train_data_extended[i][1]) for i in range(len(train_data_extended))]

        # Keep only K training points in th neighbourhood of the test point
        sorted_dist_list = sorted(dist_list, reverse=False)
        local_train_points = sorted_dist_list[0:K]
        # zip(*list of tuple) -> it returns a list of 2 element: in the first place
        # all the first entries, and in the second all the second entries as tuples
        local_train_index_list = list(list(zip(*local_train_points))[1]) # Keep only the indexes
        local_train_index_list.append(val)
        #print(local_train_index_list)
        local_train_index_list.sort()
        #print(local_train_index_list)
        index_val = local_train_index_list.index(val)
        #print('index_val ', index_val)
        local_train_set_old = np.copy(ds.data)
        # Keep only local dataset
        local_train_set = local_train_set_old[local_train_index_list]
        local_target_set_old = np.copy(ds.target)
        local_target_set = local_target_set_old[local_train_index_list]

        TDACl = TDAClassifier(local_train_set,local_target_set,ds.target_list,label_complex,ds.max_dim)

        TDACl.PruneFiltration(0,select)

        pe_list.append(TDACl.pe)
        #print('Prune')
        
        ##################### LABELING FUNCTION #####################
        # In each iter the only element of the test set is val
        list_index_test_set = [index_val]
        # predicted_value is a number
        predicted_label = TDACl.MakePrediction(index_val, list_index_test_set, ds.target_list)
        Empty_Star_count = Empty_Star_count + TDACl.ico_Empty_Star
        Link_1_count = Link_1_count + TDACl.ico_Link_1
        Link_2_count = Link_2_count + TDACl.ico_Link_2
        KNN_count = KNN_count + TDACl.ico_KNN
        Gamma_count = Gamma_count + TDACl.ico_Gamma
        del TDACl
        expected_predicted_list.append((local_target_set[index_val],predicted_label))
        local_expected_label.append(local_target_set[index_val])
        local_expected_predicted_list.append((local_target_set[index_val],predicted_label))
    
    lc = [local_expected_predicted_list.count((i,i))/local_expected_label.count(i) for i in ds.target_list]
    print('lc: ', lc)
    recall_per_class_list.append(lc)
    recall_avg_list.append(np.mean(lc))
    
print('local_recall_avg: ', recall_avg_list)

conf_matrix = np.reshape([expected_predicted_list.count((i,j)) for i in ds.target_list for j in ds.target_list],(len(ds.target_list),len(ds.target_list))).transpose()
average_recall = np.mean(recall_avg_list)
std_recall = stdev(recall_avg_list)

if write_flag == True:
    df = pd.DataFrame()
    df['Recall'] = np.asarray(recall_avg_list)
    df['Method'] = 'Local'
    df['Selector'] = select
    df = df[['Method','Recall','Selector']]
    df.to_csv(file_name, sep = ';')
    print(df)

print('The value of K is ', K)
print('Empty_Star_count: ', Empty_Star_count)
print('Link_1_count: ', Link_1_count)
print('Link_2_count: ', Link_2_count)
print('Gamma_count: ', Gamma_count)
print('Average Pe_list: ', np.mean(pe_list))

number_run = ds.n_size * ds.n_fold
print('The method behaves as KNN ', KNN_count, ' times on ', number_run, ' runs.')
print('###### CLASSIFICATION REPORT ######')
print('The dataset is ', ds.name)
print('The dataset is ', ds.type)

target_count_list = [ds.target.count(item) for item in ds.target_list]

print('The distribution of point is: ', target_count_list)
print('The confusion matrix is \n', conf_matrix)
for i in range(len(ds.target_list)):
    print('The average of recall of class ', ds.target_list[i], ' is ', np.mean(list(list(zip(*recall_per_class_list))[i])))

print('The average of total recall is ', average_recall)
print('The standard deviation of total recall is ', std_recall)
print('End script')