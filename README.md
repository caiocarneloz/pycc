# Particle Competition and Cooperation
Python code for the semi-supervised learning method "particle competition and cooperation".

## Getting Started
#### Dependencies
You need Python 3.7 or later to use **pycc**. You can find it at [python.org](https://www.python.org/).

You aso need the numpy package, which is available from [PyPI](https://pypi.org). If you have pip, just run:
```
pip install numpy
```
#### Installation
Clone this repo to your local machine using:
```
git clone https://github.com/caiocarneloz/pycc.git
```

## Usage
The usage of this class is pretty similar to [semi-supervised algorithms at scikit-learn](https://scikit-learn.org/stable/modules/label_propagation.html).

#### Execution
Considering the following [**iris.csv**](https://archive.ics.uci.edu/ml/datasets/iris) file located on **/home/datasets**:
```
sepal_l  sepal_w  petal_l  petal_w  label
5.1      3.5      1.4      0.2      Iris-setosa
4.9      3.0      1.4      0.2      Iris-setosa
7.0      3.2      4.7      1.4      Iris-versicolor
6.4      3.2      4.5      1.5      Iris-versicolor
6.3      3.3      6.0      2.5      Iris-virginica
5.8      2.7      5.1      1.9      Iris-virginica
```
One way to execute the particle competition and cooperation algorithm is to run:
```
df = pd.read_csv('/home/datasets/iris.csv')
data = df.loc[:,df.columns != 'label']
labels = df.loc[:,'label']

model = ParticleCompetitionAndCooperation(n_neighbors=32, pgrd=0.6, delta_v=0.35, max_iter=100)
model.fit(data, labels)
pred = model.predict(data)
```

#### Output
As output, a report containing the accuracy for all present classes is shown:
```
Iris-setosa: 
45/45 - 1.00


Iris-versicolor: 
41/45 - 0.91


Iris-virginica: 
39/45 - 0.87


Accuracy: 
125/135 - 0.93
```

#### Parameters
As arguments, **pycc** receives the values explained below:

---
- **n_neighbors:** value that represents the number of neighbours in the graph build.
- **pgrd:** value from 0 to 1 that defines the probability of particles to take the greedy movement.
- **delta_v:** value from 0 to 1 to control changing rate of the domination levels.
- **max_iter:** number of epochs until the label propagation stops.
---

## Citation
If you use this algorithm, please cite the original publication:

`Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, "Particle Competition and Cooperation in Networks for Semi-Supervised Learning," Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012`

https://doi.org/10.1109/TKDE.2011.119

Accepted Manuscript: https://www.fabriciobreve.com/artigos/ieee-tkde-2009.pdf
