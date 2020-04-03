import numpy as np
import random
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from pycc import ParticleCompetitionAndCooperation

#FUNCTION FOR ENCODING STRING LABELS AND GENERATING "UNLABELED DATA"
def maskData(true_labels, percentage):

    mask = np.ones((1,len(true_labels)),dtype=bool)[0]
    labels = true_labels.copy()
    
    for l, enc in zip(np.unique(true_labels),range(0,len(np.unique(true_labels)))):
        
        deck = np.argwhere(true_labels == l).flatten()        
        random.shuffle(deck)
        
        mask[deck[:int(percentage * len(true_labels[true_labels == l]))]] = False

        labels[labels == l] = enc

    labels[mask] = -1
    
    return np.array(labels).astype(int)

#IMPORT DATASETS
iris = datasets.load_iris()
data = iris.data
labels = iris.target

#GENERATE UNLABELED DATA
masked_labels = maskData(labels, 0.1)

#RUN THE MODEL
model = ParticleCompetitionAndCooperation(n_neighbors=32, pgrd=0.6, delta_v=0.35, max_iter=1000)
model.fit(data, masked_labels)
pred = np.array(model.predict(data))

#SEPARATE PREDICTED SAMPLES
labels = np.array(labels[masked_labels == -1]).astype(int)
pred = pred[masked_labels == -1]

#PRINT CONFUSION MATRIX
print(confusion_matrix(labels, pred))
