import os
import sys
import time
import random
import numpy as np
import pandas as pd

def getAccuracy(nodes, true_labels, labels):
    
    """
    function: prints all classes separately accuracy and mean accuracy

    args:
    ----
    <nodes> (DataFrame): dataframe containing all node information
    <true_labels> (list): list containing dataset true labels
    <labels> (list): list containing the output labels generated
    
    """
    
    df_x = pd.DataFrame()
    df_x['true'] = true_labels
    df_x['predicted'] = nodes['class']
    df_x['labeled'] = labels

    sum_true = 0
    sum_pred = 0

    for l in true_labels.unique():
        true = len(df_x[(df_x['labeled'] == '-1') & (df_x['true'] == l)])
        pred = len(df_x[(df_x['labeled'] == '-1') & (df_x['true'] == l) & (df_x['true'] == df_x['predicted'])])
        print('\n\n'+l+': \n'+str(pred) + '/' +str(true) + ' - ' + "{0:.2f}".format(pred/true))
        sum_true += true
        sum_pred += pred

    print('\n\nAccuracy: \n'+str(sum_pred) + '/' +str(sum_true) + ' - ' + "{0:.2f}".format(sum_pred/sum_true))

def maskData(true_labels, percentage):
    
    """
    function: turns a percentage of labeled data into unlabeled data 

    args:
    ----
    <true_labels> (list): list containing dataset true labels
    <percentage> (float): value with labeled data percentage 
    
    """

    mask = np.ones((1,len(true_labels)),dtype=bool)[0]


    for l in true_labels.unique():
        deck = [i for i, x in enumerate(true_labels == l) if x]
        random.shuffle(deck)

        it = int(percentage * len(true_labels[true_labels == l]))

        for i in range(0,it):
            mask[deck.pop(np.random.choice(len(deck)))] = False

    labels = true_labels.copy()
    labels[mask] = '-1'

    return labels

def genNodes(data, labels, particles, c):
    
    """
    function:

    args:
    ----

    
    """

    nodes = pd.DataFrame(pd.np.empty((len(data), 0)) * pd.np.nan)
    nodes['class'] = labels

    for l in np.unique(labels):
        if(l != '-1'):
            nodes['dom_'+str(l)] = 0.0
            nodes.loc[str(l) == nodes['class'], 'dom_'+str(l)] = 1

    nodes.loc[nodes['class'] == '-1',nodes.columns != 'class'] = 1/c

    for i, p in particles.iterrows():
        nodes['dist_'+str(i)] = 0
        nodes.loc[p['home'],'dist_'+str(i)] = 0
        nodes.loc[nodes.index != p['home'],'dist_'+str(i)] = len(nodes)-1

    return nodes

def genParticles(data, labels):
    
    """
    function:

    args:
    ----

    
    """

    n_particles = len(labels[labels != '-1'])
    particles = pd.DataFrame(pd.np.empty((n_particles, 0)) * pd.np.nan)

    particles['current'] = 0
    particles['home'] = 0
    particles['strength'] = 0.0
    particles['class'] = ''

    p_i = 0
    for i in range(0,len(labels)):
        if(labels[i] != '-1'):
            particles['current'].values[p_i] = particles['home'].values[p_i] = i
            particles['strength'].values[p_i] = 1
            particles['class'].values[p_i] = labels[i]
            p_i += 1

    return particles

def genGraph(data, k_neighbors, sigma, policy):
    
    """
    function:

    args:
    ----

    
    """

    graph = {}
    
    for i in range(0,len(data)):

        dist = [float("inf")] * len(data)

        actual = data.values[i]

        for j in range(0,len(data)):
            if(i != j):
                dist[j] = np.linalg.norm(actual-data.values[j])

        dist = np.array(dist)
        dist = np.argsort(dist)
        graph[i] = list(dist[0:k_neighbors])

    for i in range(0,len(data)):
        graph[i] += ([k for k,v in graph.items() if i in v])
        graph[i] = list(set(graph[i]))

    return graph

def randomWalk(neighbours):
    
    """
    function:

    args:
    ----

    
    """
    
    return neighbours[np.random.choice(len(neighbours))]

def greedyWalk(nodes, particle, p_i, neighbours):
    
    """
    function:

    args:
    ----

    
    """

    prob_sum = 0
    slices = []
    label = particle['class']



    for n in neighbours:
        prob_sum += nodes.loc[n,'dom_'+str(label)]*(1/pow(1+nodes.loc[n,'dist_'+str(p_i)],2))

    for n in neighbours:
        slices.append((nodes.loc[n,'dom_'+str(label)]*(1/pow(1+nodes.loc[n,'dist_'+str(p_i)],2)))/prob_sum)

    choice = 0
    roullete_sum = 0
    rand = np.random.uniform(0,prob_sum)

    for i in range(0,len(slices)):
        roullete_sum += slices[i]
        if(roullete_sum > rand):
            choice = i
            break

    return neighbours[choice]

def update(node, particle, n_i, p_i, labels, c, delta_v):
    
    """
    function:

    args:
    ----

    
    """

    current_domain = []
    new_domain = []
    
    if(labels[n_i] == '-1'):
        sub = (delta_v*particle.loc[p_i,'strength'])/(len(labels)-1)
    
        for l in labels.unique():
            if(particle.loc[p_i,'class'] != l and l != '-1'):
                current_domain.append(node.loc[n_i,'dom_'+str(l)])
                node.loc[n_i,'dom_'+str(l)] = max([0,node.loc[n_i,'dom_'+str(l)]-sub])
                new_domain.append(node.loc[n_i,'dom_'+str(l)])
    
    
        difference = []
        for i in range(0,len(current_domain)):
            difference.append(current_domain[i]-new_domain[i])
    
    
        node.loc[n_i,'dom_'+particle.loc[p_i,'class']] += sum(difference)
    else:
        new_domain.append(0)
        
    
    
    particle.loc[p_i,'strength'] = node.loc[n_i,'dom_'+particle.loc[p_i,'class']]

    current_node = particle.loc[p_i,'current']
    if(node.loc[n_i,'dist_'+str(p_i)] > (node.loc[current_node,'dist_'+str(p_i)]+1)):
        node.loc[n_i,'dist_'+str(p_i)] = node.loc[current_node,'dist_'+str(p_i)]+1

    if(node.loc[n_i,'dom_'+particle.loc[p_i,'class']] > max(new_domain)):#NEW OR CURRENT?
        particle.loc[p_i,'current'] = n_i
    
def labelPropagation(graph, particles, nodes, labels, pgrd, c, delta_v, it):
    
    """
    function:

    args:
    ----

    
    """

    while(it > 0):
        for p_i in range(0,len(particles)):
            if(np.random.random() < pgrd):
                next_node = greedyWalk(nodes, particles.loc[p_i,:], p_i, graph[particles.loc[p_i,'current']])
            else:
                next_node = randomWalk(graph[particles.loc[p_i,'current']])

            update(nodes, particles, next_node, p_i, labels, c, delta_v)
        it-=1

    label_list = np.unique(labels[labels != '-1'])

    for n_i in range(0,len(nodes)):
        if(nodes['class'].values[n_i] == '-1'):
            nodes['class'].values[n_i] = label_list[np.argmax(nodes['dom_'+label_list].values[n_i])]

def main():
    #set params
    filePath = os.path.abspath(sys.argv[0]).replace('pcc.py','')
    policy      = 1
    sigma       = 2
    k_neighbors = 32
    pgrd        = 0.6
    delta_v     = 0.35
    l_data      = 0.1
    iterations  = 1500

    #read the input file
    print('reading dataset')
    df = pd.read_csv(filePath+'iris.csv', sep=',', engine='python')

    #separate data from labels
    print('separating data from labels')
    data = df.loc[:,'0':str(len(df.columns)-2)]
    true_labels = df.loc[:,'label']
    
    #mask the labeled samples
    labels = maskData(true_labels,l_data)

    #get the number of classes
    c = len(np.unique(true_labels))

    #creates the node structure and particles
    print('generating particles and nodes')
    particles = genParticles(data, labels)
    nodes = genNodes(data, labels, particles, c)

    #creates the graph adjacency list
    print('generating adjacency list')
    graph = genGraph(data, k_neighbors, sigma, policy)

    #run the label propagation
    print('running label propagation')
    start = time.time()
    labelPropagation(graph, particles, nodes, labels, pgrd, c, delta_v, iterations)
    end = time.time()
    print('finished with time: '+"{0:.0f}".format(end-start)+'s')

    #show the accuracy obtained
    getAccuracy(nodes, true_labels, labels)

if __name__ == "__main__":
    main()