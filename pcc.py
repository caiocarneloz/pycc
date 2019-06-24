import time
import random
import numpy as np
import pandas as pd

def accuracyScore(masked_labels, true_labels, predicted_labels):

    """
    function: prints all classes separately accuracy and mean accuracy

    args:
    ----
    <nodes> (DataFrame): dataframe containing all node information
    <true_labels> (list): list containing dataset true labels
    <labels> (list): list containing the output labels generated

    """

    df_x = pd.DataFrame()
    df_x['labeled'] = masked_labels
    df_x['true'] = true_labels
    df_x['predicted'] = predicted_labels

    true = len(df_x[df_x['labeled'] == '-1'])
    pred = len(df_x[(df_x['labeled'] == '-1') & (df_x['true'] == df_x['predicted'])])

    return float(pred/true)

def accuracyReport(nodes, true_labels, labels):

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

def genClassMap(true_labels):
    
    i = 1
    class_map = {}
    for c in np.unique(true_labels):
        class_map[c] = i
        i+=1
        
    return class_map

def genParticles(data, labels):
    
    particles = []

    for i in range(0,len(labels)):
        if(labels[i] != '-1'):
            particles.append([int(i),int(i),1,labels[i]])

    return np.array(particles, dtype='object')

def genNodes(data, labels, particles, c, class_map):
    
    nodes = np.array([[0] * len(np.unique(labels)) for i in range(len(data))], dtype='object')
    nodes[:,0] = labels.values

    for l in np.unique(labels):
        if(l != '-1'):            
            nodes[nodes[:,0] == str(l), class_map[str(l)]] = 1
    
    nodes[nodes[:,0] == '-1',1:] = 1/c

    dist_table = np.array([[len(data)-1] * len(particles) for i in range(len(data))])

    for i in range(0,len(particles)):
        dist_table[particles[i,1],i] = 0

    return nodes, dist_table

def genGraph(data, k_neighbors, sigma, policy):
    
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
    
    return neighbours[np.random.choice(len(neighbours))]

def greedyWalk(storage, p_i, neighbours):
    
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

def update(storage, n_i, p_i, labels, c, delta_v):

    current_domain = []
    new_domain = []

    if(labels[n_i] == '-1'):
        sub = (delta_v*storage['particles'][p_i,2])/(len(labels)-1)

        for l in labels.unique():
            if(storage['particles'][p_i,3] != l and l != '-1'):
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

def labelPropagation(graph, storage, labels, pgrd, c, delta_v, iterations):

    for it in range(0,iterations):

        for p_i in range(0,len(storage['particles'])):
            if(np.random.random() < pgrd):
                next_node = greedyWalk(storage, p_i, graph[storage['particles'][p_i,0]])
            else:
                next_node = randomWalk(graph[storage['particles'][p_i,0]])

            update(storage, next_node, p_i, labels, c, delta_v)

    label_list = np.unique(labels[labels != '-1'])

    for n_i in range(0,len(storage['nodes'])):
        if(storage['nodes'][n_i,0] == '-1'):
            storage['nodes'][n_i,0] = label_list[np.argmax(storage['nodes'][n_i,1:])]


def PCC(data, true_labels, policy, sigma, k, pgrd, delta_v, l_data, epochs):

    start = time.time()
    
    storage = {}
    
    #mask the labeled samples
    labels = maskData(true_labels,l_data)

    #get the number of classes
    c = len(np.unique(true_labels))

    #creates the node structure and particles
    #print('generating particles and nodes')
    storage['class_map'] = genClassMap(true_labels)
    storage['particles'] = genParticles(data, labels)
    storage['nodes'], storage['dist_table'] = genNodes(data, labels, storage['particles'], c, storage['class_map'])

    #creates the graph adjacency list
    #print('generating adjacency list')
    graph = genGraph(data, k, sigma, policy)

    #run the label propagation
    #print('running label propagation...')
    
    labelPropagation(graph, storage, labels, pgrd, c, delta_v, epochs)
    
    end = time.time()
    
    print('finished with time: '+"{0:.0f}".format(end-start)+'s')

    return labels, true_labels, list(storage['nodes'][:,0])