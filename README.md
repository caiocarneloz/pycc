# Particle Competition and Cooperation
Python code for the semi-supervised learning method "particle competition and cooperation". This particular code was used in my master's thesis "[Aid in Alzheimer's disease diagnosis from magnetic resonance imaging using particle competition and cooperation](https://repositorio.unesp.br/handle/11449/191774)".

## Getting Started
#### Installation
You need Python 3.7 or later to use **pycc**. You can find it at [python.org](https://www.python.org/).

The package is avaliable at [PyPI](https://pypi.org). If you have pip, just run:
```
pip install pypcc
```

or clone this repo to your local machine using:
```
git clone https://github.com/caiocarneloz/pycc.git
```

## Usage
The usage of this class is pretty similar to [semi-supervised algorithms at scikit-learn](https://scikit-learn.org/stable/modules/label_propagation.html). A "demo" code was added to this repository.

## Parameters
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
