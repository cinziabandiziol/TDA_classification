from sklearn.metrics import pairwise_distances
import numpy as np
import gudhi
from scipy.stats import hmean
import random
import matplotlib.pyplot as plt
import statistics
import math

class TDAClassifier:
    def __init__(self,data,target,target_list,label_complex,max_dim):
        self.label_complex = label_complex
        self.max_dim = max_dim
        self.data = data
        self.target = target
        self.target_list = target_list
        self.ico_Empty_Star = 0
        self.ico_KNN = 0
        self.ico_Link_1 = 0
        self.ico_Link_2 = 0
        self.ico_Gamma = 0
        self.CreateSimplicialComplexFiltration()

    def CreateSimplicialComplexFiltration(self):
        if self.label_complex == 'RIPS':
            distance = 'euclidean'
            # distance = 'cosine'
            # ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']
            self.distance_matrix = pairwise_distances(self.data, n_jobs = -1, metric = distance)
            self.rips_complex = gudhi.RipsComplex(distance_matrix = self.distance_matrix)
            self.sim_tree = self.rips_complex.create_simplex_tree(max_dimension = 1)
            # print('Nella filtrazione ci sono ', self.sim_tree.num_simplices(), ' simplessi')
            self.sim_tree.collapse_edges(2)
            #print('Dopo collapse_edges nella filtrazione ci sono ', self.sim_tree.num_simplices(), ' simplessi')
            self.sim_tree.expansion(self.max_dim)
        else:
            raise TypeError('Label of Simplicial Complex Filtration not found')
        
    def PruneFiltration(self,pe=0,select=0):
        if select != 0:
            diag = self.sim_tree.persistence()
            # print(diag)
            # gudhi.plot_persistence_barcode(diag)
            # plt.show()
            # gudhi.plot_persistence_diagram(diag)
            # plt.show()
            pintervals = []

            dimension = self.sim_tree.dimension() # It is the value of q (the kth Homology group are empty set if k > dim(K))

            # In the original code the author doesn't onsider pi with dimension d = 0 (range(1) not range(0))
            for d in range(1, dimension+1):
                pis = self.sim_tree.persistence_intervals_in_dimension(d) # It returns a numpy array of dimension 2
                if len(pis) > 0:
                    pintervals.extend(pis) # extend() method is like append() with more elements to add, not only one

            # This is part of sanitize function in persistent_interval_statistics
            temp_hd = np.nan_to_num(pintervals, posinf=0, nan=0) # replace nan e positive infinity values with 0, it returns an array
            # Max() is a possible choice
            #print(temp_hd)
            max_eps = temp_hd.max()
                    
            temp2 = np.nan_to_num(pintervals, posinf=max_eps*1.25)  # replace infinity values with max_eps + max_eps/4 = max_eps*1.25
            pintervals = temp2

            lifetimes = []

            for pi in pintervals:
                lftime = pi[1] - pi[0]
                lifetimes.append(lftime)

            lifetimes_array = np.array(lifetimes)

            avg = np.mean(lifetimes_array)
            havg = hmean(lifetimes_array)
            median = statistics.median(lifetimes_array)

            ############ HOW TO CHOSE K_i ##############

            choice_flag = select

            if choice_flag == 'MAX':
                # Argmax lifetimes_array
                index = np.argmax(lifetimes_array)
            elif choice_flag == 'RANDOM':
                # Index using random
                index = random.choice(range(0,len(lifetimes_array)))
            elif choice_flag == 'HAVG':
                # Argmin using havg
                index = np.argmin(np.absolute(lifetimes_array - havg))
            elif choice_flag == 'MEDIAN':
                # Argmin using median
                index = np.argmin(np.absolute(lifetimes_array - median))
            elif choice_flag == 'AVG':
                # Argmin using median
                index = np.argmin(np.absolute(lifetimes_array - avg))  
            else:
                print('Choice_flag unknown')
            pi = pintervals[index][0]
            self.pe = pintervals[index][1]

            #print('pe: ', self.pe)
            # how to obtain simplicial complex K_i
            self.sim_tree.prune_above_filtration(self.pe)
        elif pe != 0:
            self.pe = pe
            self.sim_tree.prune_above_filtration(self.pe)
        else:
            raise TypeError('Wrong use of function')
        
    def compute_link(self, sigma, star):#, list_index_test_set):
        # Now sigma is a vertex or a simplex with dim > 1, remove sigma obtaining the link
        star1 = [(list(set(star[i][0]).difference(sigma)),star[i][1]) for i in range(len(star))]
        star2 = [star1[i] for i in range(len(star1)) if star1[i][0] != []]
        if star2 != []:
            link = list(list(zip(*star2))[0])
            epsilon_val = list(list(zip(*star2))[1])
        else:
            link = []
            epsilon_val = []
        return link, epsilon_val

    def compute_link_unlabeled_points(self, star):
        # I come here iff the link is made of only unlabeled points
        simplices_list = star
        for j in range(len(star)):
            tau = star[j][0]
            eps = star[j][1]
            tau_link, tau_eps = self.compute_link(tau, self.sim_tree.get_star(tau))
            if tau_link != []:
                tau_eps1 = list(np.array(tau_eps)+eps)
                print(zip(tau_link,tau_eps1))
                list_aux = list(zip(tau_link,tau_eps1))
                simplices_list = simplices_list + list_aux
        
        print(simplices_list)
        # Remove duplicated simplices
        visited_simplices_with_duplicates = list(list(zip(*simplices_list))[0])
        print(visited_simplices_with_duplicates)
        #visited_simplices_set = list(np.unique(np.array(list(np.concatenate(list(list(zip(*simplices_list))[0])).flat))))
        visited_simplices_set = set(tuple(x) for x in visited_simplices_with_duplicates)
        visited_simplices_list = [list(s) for s in visited_simplices_set]
        
        simplices_list_ordered = sorted(simplices_list) # order by firt component, then second component and so on


        simplices_list1 = []

        for iota in visited_simplices_list:
            print('iota: ', iota)
            list1 = [simplices_list_ordered[i] for i in range(len(simplices_list_ordered)) if simplices_list_ordered[i][0] == iota]
            simplices_list1.append((iota,list1[0][1]))
        # simplices_list1 = [(iota,[simplices_list_ordered[i] for i in range(len(simplices_list_ordered)) if simplices_list_ordered[i][0] == iota][0][1]) for iota in visited_simplices_set]
        
        link = list(list(zip(*simplices_list1))[0])
        epsilon_val = list(list(zip(*simplices_list1))[1])
        
        # Here link can contain points from the test set
        return link, epsilon_val

    def simplicial_neighbourhood(self, vertex):
        Sn = []
        eps_list = []
        array_dist = np.linalg.norm(self.data - self.data[vertex[0]], axis=1)
        # We select only vertices, whose distance from vertex v is < 2*epsilon
        elems = (array_dist <= 2*self.pe)
        tupla = tuple(enumerate(elems, start=0))
        vertices_dist_list = [(tupla[i][0],array_dist[tupla[i][0]]) for i in range(len(tupla)) if ((tupla[i][1] == True) and (array_dist[tupla[i][0]] > 0))]
        for k in range(len(vertices_dist_list)):
            v = vertices_dist_list[k][0]
            star_v = self.sim_tree.get_star([v])
            for j in range(len(star_v)):
                z = list(set(star_v[j][0]).intersection(set(list(zip(*vertices_dist_list))[0])))
                Sn.append(z)
                if len(z) == 1:
                    eps_list.append(array_dist[k])
                else:
                    eps_list.append(star_v[j][1])

        return Sn, eps_list

    def compute_link_for_classification(self, vertex, star, list_index_test_set):

        # vertex is [vertex]
        labelled_link = []
        labelled_epsilon_val = []

        if star == []:
            self.ico_Empty_Star = self.ico_Empty_Star + 1
            link, epsilon_val = self.simplicial_neighbourhood(vertex)
        else:
            # Remove elements of the test set from the star
            star1 = [(list(set(star[i][0]).difference(list_index_test_set)),star[i][1]) for i in range(len(star))]
            # Remove tuple with [] as first component
            star2 = [star1[k] for k in range(len(star1)) if star1[k][0] != []]
            if star2 == []:
                link, epsilon_val = self.compute_link_unlabeled_points(star)
                self.ico_Link_1 = self.ico_Link_1 + 1
            else:
                link, epsilon_val = self.compute_link(vertex, star2)
                self.ico_Link_2 = self.ico_Link_2 + 1
        
        if link == []:
            labelled_link = []
            labelled_epsilon_val = []
        else:
            tuple_list = [(list(set(link[i]).difference(list_index_test_set)),epsilon_val[i]) for i in range(len(link))]
            tuple_list1 = [tuple_list[k] for k in range(len(tuple_list)) if ((tuple_list[k][0] != []) and (tuple_list[k][1] > 0))]

            if tuple_list1 == []:
                labelled_link = []
                labelled_epsilon_val = []
            else:
                labelled_link = list(list(zip(*tuple_list1))[0])
                labelled_epsilon_val = list(list(zip(*tuple_list1))[1])

        if labelled_link == []:
            self.ico_KNN = self.ico_KNN + 1
            ## Apply KNN with K = 1
            array_dist = np.linalg.norm(self.data - self.data[vertex[0]], axis=1)
            tupla1 = tuple(enumerate(array_dist, start=0))
            list1 = [tupla1[j] for j in range(len(tupla1)) if (tupla1[j][0] not in list_index_test_set) and (array_dist[j] > 0)]
            indices = np.argsort(np.array(list1)[:,1])
            tupla2 = list1[indices[0]]
            labelled_link.append([tupla2[0]])
            labelled_epsilon_val.append(tupla2[1])
        
        return labelled_link, labelled_epsilon_val

    def Gamma(self, result):
        # result: a list with length equal to the number of possible target values
        # list: list of the possible target values
        if (result == np.zeros(len(self.target_list))).all():
            predicted_value = random.choice(self.target_list)
            self.ico_Gamma = self.ico_Gamma + 1
            print('!!!!!!!!!!Il valore predetto è Random!!!!!!!!!!')
        else:
            predicted_target_list = [self.target_list[i] for i in range(len(result)) if result[i]==max(result)]
            
            if len(predicted_target_list) > 1:
                # If we have multiple choice for predicted value, the choice is uniformly at random
                predicted_value = random.choice(predicted_target_list)
                self.ico_Gamma = self.ico_Gamma + 1
            else:
                predicted_value = predicted_target_list[0]
        return predicted_value

    def MakePrediction(self, vertex, list_index_test_set, target_list):
        sigma = [vertex]
        star_original = self.sim_tree.get_star(sigma) # A list of tuples, ([142, 149], 0.33166247903553914)

        # Remove tuples with empty list at first position and banal simplices (if there are repeated points)
        star = [star_original[i] for i in range(len(star_original)) if star_original[i][1] != 0.0]

        link, eps_K = self.compute_link_for_classification(sigma, star, list_index_test_set)

        #weight_array = np.ones(len(eps_K))
        #weight_array = np.ones(len(eps_K))/np.square(eps_K)
        #print('weight_array: ', weight_array)
        # exp
        #weight_array = np.ones(len(eps_K)) * np.exp(-np.square(eps_K))
        # inv
        weight_array = np.ones(len(eps_K))/np.sqrt(np.ones(len(eps_K)) + np.square(eps_K))

        result = np.zeros(len(target_list))

        for j in range(len(link)):
            alpha = link[j]
            target_count_list = [self.target[alpha[i]] for i in range(len(alpha))]# if alpha[i] not in list_index_test_set]

            # If some targets are absent, count function returns 0
            to_add = np.array([target_count_list.count(target) for target in target_list])
            #print('to_add: ', to_add)
            result = result + (to_add*weight_array[j])

        ##################### LABELING FUNCTION #####################
        # predicted_value is a number
        return self.Gamma(result)