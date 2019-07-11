import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

class ParticleCompetitionAndCooperation():

    def __init__(self, n_neighbours=1, pgrd=0.5, delta_v=0.35, max_iter=1000):

        self.n_neighbours = n_neighbours
        self.pgrd = pgrd
        self.delta_v = delta_v
        self.max_iter = max_iter
        self.accuracy_score = 0

    def predict(self, data, labels):

        start = time.time()

        storage = {}

        c = len(np.unique(labels))

        storage['class_map'] = self.__genClassMap(labels)
        storage['particles'] = self.__genParticles(data, labels)
        storage['nodes'] = self.__genNodes(data, labels, storage['particles'], c, storage['class_map'])
        storage['dist_table'] = self.__genDistTable(data, storage['particles'])

        graph = self.__genGraph(data)

        self.__labelPropagation(graph, storage, labels, c)

        end = time.time()

        print('finished with time: '+"{0:.0f}".format(end-start)+'s')

        return list(storage['nodes'][:,0])
    
    def __labelPropagation(self, graph, storage, labels, c):

        for it in range(0,self.max_iter):

            for p_i in range(0,len(storage['particles'])):
                if(np.random.random() < self.pgrd):
                    next_node = self.__greedyWalk(storage, p_i, graph[storage['particles'][p_i,0]])
                else:
                    next_node = self.__randomWalk(graph[storage['particles'][p_i,0]])

                self.__update(storage, next_node, p_i, labels, c)

        label_list = np.unique(labels[labels != -1])

        for n_i in range(0,len(storage['nodes'])):
            if(storage['nodes'][n_i,0] == -1):
                storage['nodes'][n_i,0] = label_list[np.argmax(storage['nodes'][n_i,1:])]
    
    def __update(self, storage, n_i, p_i, labels, c):

        current_domain = []
        new_domain = []

        if(labels[n_i] == -1):
            sub = (self.delta_v*storage['particles'][p_i,2])/(len(labels)-1)

            for l in labels.unique():
                if(storage['particles'][p_i,3] != l and l != -1):
                    current_domain.append(storage['nodes'][n_i,storage['class_map'][str(l)]])
                    storage['nodes'][n_i,storage['class_map'][str(l)]] = max([0,storage['nodes'][n_i,storage['class_map'][str(l)]]-sub])
                    new_domain.append(storage['nodes'][n_i,storage['class_map'][str(l)]])


            difference = []
            for i in range(0,len(current_domain)):
                difference.append(current_domain[i]-new_domain[i])

            storage['nodes'][n_i,storage['class_map'][storage['particles'][p_i,3]]] += sum(difference)
        else:
            new_domain.append(0)

        storage['particles'][p_i,2] = storage['nodes'][n_i,storage['class_map'][storage['particles'][p_i,3]]]

        current_node = storage['particles'][p_i,0]
        if(storage['dist_table'][n_i,p_i] > (storage['dist_table'][current_node,p_i]+1)):
            storage['dist_table'][n_i,p_i] = storage['dist_table'][current_node,p_i]+1

        if(storage['nodes'][n_i,storage['class_map'][storage['particles'][p_i,3]]] > max(new_domain)):#NEW OR CURRENT?
            storage['particles'][p_i,0] = n_i

    def __greedyWalk(self, storage, p_i, neighbours):

        prob_sum = 0
        slices = []
        label = storage['particles'][p_i,3]

        for n in neighbours:
            prob_sum += storage['nodes'][n,storage['class_map'][str(label)]]*(1/pow(1+storage['dist_table'][n,p_i],2))

        for n in neighbours:
            slices.append((storage['nodes'][n,storage['class_map'][str(label)]]*(1/pow(1+storage['dist_table'][n,p_i],2)))/prob_sum)

        choice = 0
        roullete_sum = 0
        rand = np.random.uniform(0,prob_sum)

        for i in range(0,len(slices)):
            roullete_sum += slices[i]
            if(roullete_sum > rand):
                choice = i
                break

        return neighbours[choice]
    
    
    def __randomWalk(self, neighbours):

        return neighbours[np.random.choice(len(neighbours))]
    
    def __accuracyScore(self, masked_labels, true_labels, predicted_labels):
    
        df_x = pd.DataFrame()
        df_x['labeled'] = masked_labels
        df_x['true'] = true_labels
        df_x['predicted'] = predicted_labels
    
        true = len(df_x[df_x['labeled'] == -1])
        pred = len(df_x[(df_x['labeled'] == -1) & (df_x['true'] == df_x['predicted'])])
    
        return float(pred/true)
    
    def __accuracyReport(self, predicted_labels, true_labels, labels):
    
        df_x = pd.DataFrame()
        df_x['true'] = true_labels
        df_x['predicted'] = predicted_labels
        df_x['labeled'] = labels
    
        sum_true = 0
        sum_pred = 0
    
        for l in true_labels.unique():
            true = len(df_x[(df_x['labeled'] == -1) & (df_x['true'] == l)])
            pred = len(df_x[(df_x['labeled'] == -1) & (df_x['true'] == l) & (df_x['true'] == df_x['predicted'])])
            print('\n\n'+l+': \n'+str(pred) + '/' +str(true) + ' - ' + "{0:.2f}".format(pred/true))
            sum_true += true
            sum_pred += pred
    
        print('\n\nAccuracy: \n'+str(sum_pred) + '/' +str(sum_true) + ' - ' + "{0:.2f}".format(sum_pred/sum_true))
    
    def __genClassMap(self, true_labels):
    
        i = 1
        class_map = {}
        for c in np.unique(true_labels):
            if(c != -1):
                class_map[c] = i
                i+=1
    
        return class_map
    
    def __genParticles(self, data, labels):
    
        particles = []
    
        for i in range(0,len(labels)):
            if(labels[i] != -1):
                particles.append([int(i),int(i),1,labels[i]])
    
        return np.array(particles, dtype='object')
    
    def __genNodes(self, data, labels, particles, c, class_map):
    
        nodes = np.array([[0] * len(np.unique(labels)) for i in range(len(data))], dtype='object')
        nodes[:,0] = labels.values
    
        for l in np.unique(labels):
            if(l != -1):
                nodes[nodes[:,0] == str(l), class_map[str(l)]] = 1
    
        nodes[nodes[:,0] == -1,1:] = 1/c
    
        return nodes
    
    def __genDistTable(self, data, particles):
    
        dist_table = np.array([[len(data)-1] * len(particles) for i in range(len(data))])
    
        for i in range(0,len(particles)):
            dist_table[particles[i,1],i] = 0
    
        return dist_table
    
    def __genGraph(self, data):
    
        data_v = data.values
        graph = {}
    
        dist = np.array([[float("inf")] * len(data_v) for i in range(len(data_v))])
    
        for i in range(0,len(data_v)):
    
            actual = data_v[i]
    
            for j in range(i+1,len(data_v)):
                    dist[j,i] = dist[i,j] = np.linalg.norm(actual-data_v[j])
    
        for i in range(0,len(data_v)):
            sorted_dist = np.argsort(dist[i])
            graph[i] = list(sorted_dist[0:self.n_neighbours])
    
        for i in range(0,len(data)):
            graph[i] += ([k for k,v in graph.items() if i in v])
            graph[i] = list(set(graph[i]))
    
        return graph