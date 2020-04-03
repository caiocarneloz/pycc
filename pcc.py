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
        self.unique_labels = []
        self.spent = 0


    def fit(self, data, labels):

        start = time.time()

        self.data = data
        self.labels = labels
        self.unique_labels = np.unique(self.labels)
        self.unique_labels = self.unique_labels[self.unique_labels != -1]
        self.c = len(self.unique_labels)

        self.class_map = self.__genClassMap()
        self.particles = self.__genParticles()
        self.nodes = self.__genNodes()
        self.dist_table = self.__genDistTable()

        self.graph = self.__genGraph()

        end = time.time()

        print('finished with time: '+"{0:.5f}".format(end-start)+'s')


    def predict(self, data):

        start = time.time()

        self.__labelPropagation()

        end = time.time()

        print('finished with time: '+"{0:.5f}".format(self.spent)+'s')

        print('finished with time: '+"{0:.5f}".format(end-start)+'s')

        return list(self.nodes[:,0])


    def __labelPropagation(self):

        for it in range(0,self.max_iter):

            for p_i in range(0,len(self.particles)):

                if(np.random.random() < self.pgrd):
                    next_node = self.__greedyWalk(p_i, self.graph[self.particles[p_i,0]])
                else:
                    next_node = self.__randomWalk(self.graph[self.particles[p_i,0]])

                self.__update(next_node, p_i)

        for n_i in range(0,len(self.nodes)):
            if(self.nodes[n_i,0] == -1):
                self.nodes[n_i,0] = self.unique_labels[np.argmax(self.nodes[n_i,1:])]


    def __update(self, n_i, p_i):

        current_domain = []
        new_domain = []

        if(self.labels[n_i] == -1):
            sub = (self.delta_v*self.particles[p_i,2])/(len(self.labels)-1)

            for l in self.unique_labels:
                if(self.particles[p_i,3] != l):
                    current_domain.append(self.nodes[n_i,self.class_map[l]])
                    self.nodes[n_i,self.class_map[l]] = max([0,self.nodes[n_i,self.class_map[l]]-sub])
                    new_domain.append(self.nodes[n_i,self.class_map[l]])


            difference = []
            for i in range(0,len(current_domain)):
                difference.append(current_domain[i]-new_domain[i])

            self.nodes[n_i,self.class_map[self.particles[p_i,3]]] += sum(difference)
        else:
            new_domain.append(0)

        self.particles[p_i,2] = self.nodes[n_i,self.class_map[self.particles[p_i,3]]]

        current_node = self.particles[p_i,0]
        if(self.dist_table[n_i,p_i] > (self.dist_table[current_node,p_i]+1)):
            self.dist_table[n_i,p_i] = self.dist_table[current_node,p_i]+1

        if(self.nodes[n_i,self.class_map[self.particles[p_i,3]]] > max(new_domain)):#NEW OR CURRENT?
            self.particles[p_i,0] = n_i


    def __greedyWalk(self, p_i, neighbors):
    
        start = time.time()
        
        prob_sum = 0
        slices = []
        label = self.particles[p_i,3]

        for n in neighbors:
            prob_sum += self.nodes[n,self.class_map[label]]*(1/pow(1+self.dist_table[n,p_i],2))
            slices.append(self.nodes[n,self.class_map[label]]*(1/pow(1+self.dist_table[n,p_i],2)))
            
        slices = slices/sum(slices)

        choice = 0
        roullete_sum = 0
        rand = np.random.uniform(0,prob_sum)

        for i in range(0,len(slices)):
            roullete_sum += slices[i]
            if(roullete_sum > rand):
                choice = i
                break
            
        end = time.time()

        self.spent += end - start
            
        return neighbors[choice]


    def __randomWalk(self, neighbors):

        return neighbors[np.random.choice(len(neighbors))]


    def __genClassMap(self):

        i = 1
        class_map = {}
        for c in self.unique_labels:
            class_map[c] = i
            i+=1

        return class_map

    def __genParticles(self):

        labeled = self.labels[self.labels!=-1]
        indexes = np.where(self.labels!=-1)[0]

        particles = np.ones(shape=(len(labeled),4), dtype=int)

        particles[:,0] = particles[:,1] = indexes
        particles[:,3] = labeled

        return particles


    def __genNodes(self):

        nodes = np.full(shape=(len(self.data),len(self.unique_labels)+1), fill_value=float(1/self.c))
        nodes[:,0] = self.labels

        nodes[nodes[:,0] != -1,1:] = 0

        for l in np.unique(self.labels[self.labels!=-1]):
            nodes[nodes[:,0] == l,l+1] = 1

        return nodes


    def __genDistTable(self):

        dist_table = np.full(shape=(len(self.data),len(self.particles)), fill_value=len(self.data)-1,dtype=int)

        for h,i in zip(self.particles[:,1],range(len(self.particles))):
            dist_table[h,i] = 0

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
