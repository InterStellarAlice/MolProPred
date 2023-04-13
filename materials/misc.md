
## Basics on graph neural network (GNN)
[3B1B: but what is neural network?](https://www.youtube.com/watch?v=aircAruvnKk)

[deeplearningbook-chinese](https://exacity.github.io/deeplearningbook-chinese/)

[Graph Neural Networks for Molecules-A Chapter for Book “Machine Learning in Molecular Sciences"](https://arxiv.org/pdf/2209.05582.pdf)

## trouble-shootings in kaggle notebook

* to response [y]/n when installing packages using `pip` or `conda` in a notebook:

```
!echo y | <installation command>
```

* [How to install RDKit with Conda](http://www.rdkit.org/docs/Install.html) (invalid in kaggle notebook)

## Methods and models
* [continuuous filter convolutional neural network(CFConv) in pytorch](https://docs.dgl.ai/en/1.0.x/generated/dgl.nn.pytorch.conv.CFConv.html)
* [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980v8.pdf)
* [Gradient Descent With Momentum](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/)
* [SchNetPack - Deep Neural Networks for Atomistic Systems](https://github.com/atomistic-machine-learning/schnetpack) package with commits **last week** !!
* [github repo:Molecular Scalar Coupling Constant Prediction using SchNet](https://github.com/jmg764/Molecular-Scalar-Coupling-Constant-Prediction-using-SchNet)
* [model in chainer-chemistry: schnet.py](https://github.com/chainer/chainer-chemistry/blob/master/chainer_chemistry/models/schnet.py)

## Literature reviews on SchNet
original papers:
* [SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/pdf/1706.08566.pdf)
* [SchNet - a deep learning architecture for molecule and materials](https://arxiv.org/pdf/1712.06113.pdf)

## Discussions and notebooks for scalar coupling prediction problem with SchNet
* [3rd solution - BERT in chemistry - End to End is all you need](https://www.kaggle.com/competitions/champs-scalar-coupling/discussion/106572)
* [10th place solution](https://www.kaggle.com/competitions/champs-scalar-coupling/discussion/106271)
* [(notebook)Custom GCN - 10th Place Solution](https://www.kaggle.com/code/joshxsarah/custom-gcn-10th-place-solution/notebook)
* [(notebook)Molecule Animations](https://www.kaggle.com/code/cdeotte/molecule-animations)
