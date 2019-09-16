import time
import numpy as np

class ParticleCompetitionAndCooperation():

    def __init__(self, n_neighbors=1, pgrd=0.5, delta_v=0.35, max_iter=1000, kernel='knn'):

        self.n_neighbors = n_neighbors
        self.pgrd = pgrd
        self.delta_v = delta_v
        self.max_iter = max_iter
        self.accuracy_score = 0
        self.storage = {}
        self.graph = {}
        self.c = 0
        self.labels = []
        self.data = None
        
    def fit(self, data, labels):

        self.data = data
        self.labels = labels
        self.c = len(np.unique(self.labels))

        self.storage['class_map'] = self.__genClassMap()
        self.storage['particles'] = self.__genParticles()
        self.storage['nodes'] = self.__genNodes()
        self.storage['dist_table'] = self.__genDistTable()
        
        self.graph = self.__genGraph()
        
    def predict(self, data):
        
        start = time.time()

        self.__labelPropagation()

        end = time.time()

        print('finished with time: '+"{0:.0f}".format(end-start)+'s')

        return list(self.storage['nodes'][:,0])
    
    def __labelPropagation(self):

        for it in range(0,self.max_iter):

            for p_i in range(0,len(self.storage['particles'])):
                if(np.random.random() < self.pgrd):
                    next_node = self.__greedyWalk(p_i, self.graph[self.storage['particles'][p_i,0]])
                else:
                    next_node = self.__randomWalk(self.graph[self.storage['particles'][p_i,0]])

                self.__update(next_node, p_i)

        label_list = np.unique(self.labels[self.labels != -1])

        for n_i in range(0,len(self.storage['nodes'])):
            if(self.storage['nodes'][n_i,0] == -1):
                self.storage['nodes'][n_i,0] = label_list[np.argmax(self.storage['nodes'][n_i,1:])]
    
    def __update(self, n_i, p_i):

        current_domain = []
        new_domain = []

        if(self.labels[n_i] == -1):
            sub = (self.delta_v*self.storage['particles'][p_i,2])/(len(self.labels)-1)

            for l in np.unique(self.labels):
                if(self.storage['particles'][p_i,3] != l and l != -1):
                    current_domain.append(self.storage['nodes'][n_i,self.storage['class_map'][l]])
                    self.storage['nodes'][n_i,self.storage['class_map'][l]] = max([0,self.storage['nodes'][n_i,self.storage['class_map'][l]]-sub])
                    new_domain.append(self.storage['nodes'][n_i,self.storage['class_map'][l]])


            difference = []
            for i in range(0,len(current_domain)):
                difference.append(current_domain[i]-new_domain[i])

            self.storage['nodes'][n_i,self.storage['class_map'][self.storage['particles'][p_i,3]]] += sum(difference)
        else:
            new_domain.append(0)

        self.storage['particles'][p_i,2] = self.storage['nodes'][n_i,self.storage['class_map'][self.storage['particles'][p_i,3]]]

        current_node = self.storage['particles'][p_i,0]
        if(self.storage['dist_table'][n_i,p_i] > (self.storage['dist_table'][current_node,p_i]+1)):
            self.storage['dist_table'][n_i,p_i] = self.storage['dist_table'][current_node,p_i]+1

        if(self.storage['nodes'][n_i,self.storage['class_map'][self.storage['particles'][p_i,3]]] > max(new_domain)):#NEW OR CURRENT?
            self.storage['particles'][p_i,0] = n_i

    def __greedyWalk(self, p_i, neighbors):

        prob_sum = 0
        slices = []
        label = self.storage['particles'][p_i,3]

        for n in neighbors:
            prob_sum += self.storage['nodes'][n,self.storage['class_map'][label]]*(1/pow(1+self.storage['dist_table'][n,p_i],2))

        for n in neighbors:
            slices.append((self.storage['nodes'][n,self.storage['class_map'][label]]*(1/pow(1+self.storage['dist_table'][n,p_i],2)))/prob_sum)

        choice = 0
        roullete_sum = 0
        rand = np.random.uniform(0,prob_sum)

        for i in range(0,len(slices)):
            roullete_sum += slices[i]
            if(roullete_sum > rand):
                choice = i
                break

        return neighbors[choice]
    
    
    def __randomWalk(self, neighbors):

        return neighbors[np.random.choice(len(neighbors))]
    
    def __genClassMap(self):
    
        i = 1
        class_map = {}
        for c in np.unique(self.labels):
            if(c != -1):
                class_map[c] = i
                i+=1
    
        return class_map
    
    def __genParticles(self):
    
        particles = []
    
        for i in range(0,len(self.labels)):
            if(self.labels[i] != -1):
                particles.append([int(i),int(i),1,self.labels[i]])
    
        return np.array(particles, dtype='object')
    
    def __genNodes(self):
    
        nodes = np.array([[0] * len(np.unique(self.labels)) for i in range(len(self.data))], dtype='object')
        nodes[:,0] = self.labels
    
        for l in np.unique(self.labels):
            if(l != -1):
                nodes[nodes[:,0] == str(l), self.storage['class_map'][l]] = 1
    
        nodes[nodes[:,0] == -1,1:] = 1/self.c
    
        return nodes
    
    def __genDistTable(self):
    
        dist_table = np.array([[len(self.data)-1] * len(self.storage['particles']) for i in range(len(self.data))])
    
        for i in range(0,len(self.storage['particles'])):
            dist_table[self.storage['particles'][i,1],i] = 0
    
        return dist_table
    
    def __genGraph(self):
    
        values = self.data
        self.graph = {}
    
        dist = np.array([[float("inf")] * len(values) for i in range(len(values))])
    
        for i in range(0,len(values)):
    
            actual = values[i]
    
            for j in range(i+1,len(values)):
                    dist[j,i] = dist[i,j] = np.linalg.norm(actual-values[j])
    
        for i in range(0,len(values)):
            sorted_dist = np.argsort(dist[i])
            self.graph[i] = list(sorted_dist[0:self.n_neighbors])
    
        for i in range(0,len(self.data)):
            self.graph[i] += ([k for k,v in self.graph.items() if i in v])
            self.graph[i] = list(set(self.graph[i]))
    
        return self.graph