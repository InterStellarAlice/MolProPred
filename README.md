- [MolProPred](#molpropred)
- [Message Passing Graph Neural Network](#message-passing-graph-neural-network)
  * [Initializing a Graph](#initializing-a-graph)
  * [Message Passing Function](#message-passing-function)
  * [Readout Operation](#readout-operation)
- [Molecular Graph Neural Networks](#molecular-graph-neural-networks)
  * [1D String Representation](#1d-string-representation)
  * [2D Graph Representation](#2d-graph-representation)
  * [Converting Molecules to 3D Euclidean Space](#converting-molecules-to-3d-euclidean-space)
- [Introduction to SchNet](#introduction-to-schnet)
  * [Method](#method)
  * [Molecular representation](#molecular-representation)
  * [Atom-wise layers](#atom-wise-layers)
  * [Interaction blocks](#interaction-blocks)
  * [Filter-generating networks](#filter-generating-networks)
    + [Rotationally invariant](#rotationally-invariant)
    + [Periodic boundary conditions](#periodic-boundary-conditions)
  * [Property prediction](#property-prediction)
  * [Training](#training)



# MolProPred

This project entails an exploration of [Kaggle's competition](https://www.kaggle.com/competitions/champs-scalar-coupling/overview), centered on the prediction of molecular properties. Furthermore, we are gaining expertise in employing the [SchNet](https://github.com/atomistic-machine-learning/schnetpack) graph neural network package to derive these properties. The primary program is modeled after the [22nd solution](https://www.kaggle.com/competitions/champs-scalar-coupling/discussion/106424) on the leaderboard, with the aim of enhancing our understanding. Our ranking stands at approximately 1380 out of 2690 in comparison to all participating entities. 

* The nomenclature of the project is assigned by chatGPT. (prompt: Please give me an ultra-cool project name, the project is for coping with an assignment given by a course '[machine learning for physicist](https://github.com/wangleiphy/ml4p)', there are 3 team members, and the project is about predicting molecular properties.)

* Just for fun and academic credit.

* The AI extensions utilized in this project encompassed a range of tools, including [Genie-AI](https://github.com/ai-genie/chatgpt-vscode), [Github Copilot](https://docs.github.com/en/copilot) for code testing, [New Bing](https://www.bing.com/new?cc=sg&setlang=zh-hans) for literature search, [Monica](https://monica.im/) for writing assistance, [chatgpt_academic](https://github.com/binary-husky/chatgpt_academic) for recreational purposes.

# Message Passing Graph Neural Network

Graph Neural Network (GNN) is a type of neural network that can operate on graph data. It has become increasingly popular in recent years due to its ability to model complex relationships between entities in a graph. Message passing is a fundamental operation in GNN, which enables the network to propagate information between nodes in the graph. 

![image](https://user-images.githubusercontent.com/50568855/233074281-8ee144ca-804c-4e65-8ba5-3e49347c67f0.png)

## Initializing a Graph

A graph is a data structure that consists of a set of nodes (or vertices) and a set of edges connecting these nodes. In Message Passing GNNs, the first step is to initialize the graph. The graph can be represented as an adjacency matrix $A$, where $A[i,j] = 1$ if there exists an edge from node $i$ to node $j$, and $A[i,j] = 0$ otherwise. Alternatively, we can represent the graph as an edge list, which is a list of tuples $(i,j)$ representing the edges.

In addition to the adjacency matrix or edge list, we also need to initialize the node features. Each node has a feature vector that describes its properties. For example, in a social network analysis task, the node features can be the age, gender, and occupation of each user. In a drug discovery task, the node features can be the molecular properties of each atom in a molecule.

## Message Passing Function

After initializing the graph and node features, we can start the message-passing process. The message-passing function is the core component of Message Passing GNNs. It aggregates information from neighboring nodes and updates the node features iteratively.

The aggregation operation can be formulated as follows, which aggregates messages from neighbor vertex:

$$
\vec{m}_i^{(k)}=\sum_{u_j \in \mathcal{N}\left(u_i\right)} \phi_m^{(k)}\left(\vec{h}_i^{(k-1)}, \vec{h}_j^{(k-1)}, \vec{a}_{i j}\right).
$$

then the node feature is updated, 

$$
\vec{h}_i^{(k)}=\phi_u^{(k)}=\left(h_i^{(k-1)}, m_i^{(k)}\right),
$$

the functions denoted by $\phi_m^{(k)}(\cdot)$ and $\phi_m^{(k)}(\cdot)$ are referred to as the message function and update function, respectively.

## Readout Operation

After several rounds of message passing, the final step is to perform a readout operation to obtain a graph-level representation. The readout operation aggregates the node features into a single vector, which represents the entire graph,

$$
\vec{h}_G=R \left(  \vec{h}_i^{(k)} \mid u_i \in \mathcal{V}  \right).
$$

There are various methods for readout operation, including sum pooling, max pooling, and attention-based pooling. The choice of readout operation depends on the task at hand and the characteristics of the graph.

# Molecular Graph Neural Networks

![image](https://user-images.githubusercontent.com/50568855/233082788-061c82c3-173c-4f11-8ef4-33a4695badca.png)

## 1D String Representation

The simplest way to represent a molecule is as a 1D string of characters. Each character represents an atom, and the order of the characters represents the order in which the atoms are connected. For example, the string "C1=CC=CC=C1" represents benzene, where each "C" represents a carbon atom and each "=" represents a double bond between carbon atoms.

Algorithms utilizing such methodologies include:

* Simplified molecular-input line-entry system (SMILES)

![SMILES generation algorithm for ciprofloxacin: break cycles, then write as branches off a main backbone](https://user-images.githubusercontent.com/50568855/233084742-6d361d92-c15b-406b-a703-080d8d9524db.png)

* SMILES arbitrary target specification (SMARTS)
* Self-referencing embedded strings (SELFIES)

## 2D Graph Representation

A more powerful representation of a molecule is as a 2D graph. In this representation, each atom is a node in the graph, and each bond between atoms is an edge. The type of bond (single, double, etc.) can be represented as a label on the edge. One common way to represent molecules as 2D graphs is through the use of adjacency matrices. An adjacency matrix is a square matrix that represents the connections between nodes in a graph. In the case of molecules, the adjacency matrix represents the bonds between atoms.

2D graphical representations solely encompass the topological characteristics, yet their ability to convey properties such as distance, angle, and other attributes in 3D Euclidean space remains restricted.

## Converting Molecules to 3D Euclidean Space

In addition to representing molecules as 2D graphs, it is also possible to represent them in 3D Euclidean space. This can be useful for modeling the spatial arrangement of atoms in a molecule.

The SchNet algorithm utilized in our project incorporates meticulously crafted layers that adeptly capture local correlations and facilitate atom-wise updating through continuous filter convolution. By effectively encoding 3D distance information to molecular GNN, SchNet has served as a source of inspiration for numerous subsequent works.

# Introduction to SchNet
SchNet: model atomistic systems by making use of continuous-filter convolutional layers(model interaction between atoms).

Allow to model complex atomic interactions.

- Predict potential energy surfaces.
- Speed up the exploration of chemical space.

Consider fundamental symmetries of atomistic systems.

- Rotational and translational invariance as well as invariance to atom indexing.

## Method
SchNet is a variant of the earlier proposed Deep Tensor Neural Networks(DTNN).
- DTNN: interactions are modeled by tensor layers, i.e., atom representations and interatomic distances are combined using a parameter tensor. 
- SchNet: makes use of continuous-filter convolutions with filter-generating networks to model the interaction term.

At each layer, the molecule is represented atom-wise analogous to pixels in an image.

Interactions between atoms are modeled by the three interaction blocks.

The final prediction is obtained after atom-wise updates of the feature representation and pooling of the resulting atom-wise energy. 

## Molecular representation
Nuclear charges $Z=(Z_1,…,Z_n)$

Positions $ R=(\mathbf r_1,…,\mathbf r_n)$

The atoms are described by a tuple of features $X^l=(\mathbf x_1^l…,\mathbf x_l^l)$
- $\mathbf x_i^l \inℝ^F$, $F$: number of feature maps
- $n$: number of atoms
- $l$: current layer
$\mathbf x_i^0$ is initialized using an embedding dependent on the atom type $Z_i$:

$$
\mathbf{x}_{i}^{0}=\mathbf{a}_{Z_i}.
$$

The atom type embeddings $\mathbf{a}_{Z}$ are initialized randomly and optimized during training.

## Atom-wise layers
Are dense layers.

Applied separately to the representations $\mathbf x_i^l$ of each atom $i$:

$$
\mathbf{x}_{i}^{l+1}=W^l\mathbf{x}_{i}^{l}+\mathbf{b}^l,
$$
- Weights $W^l$ and biases $\mathbf b^l$ are shared across atoms.
- The architecture remains scalable with respect to the number of atoms.

These layers are responsible for the recombination of feature maps.

## Interaction blocks
Updating the atomic representations based on pair-wise interactions with the surrounding atoms.

Continuous-filter convolutional layers, a generalization of the discrete convolutional layers commonly used.
- Atoms are located at arbitrary positions.

Model the filters continuously with a filter-generation neural network $W^l$ that maps the atom positions to the corresponding values of the filter bank.

$$
\begin{aligned}
\mathbf{x}_{i}^{l+1}&=\left( X^l*W^l \right) _i\\
&=\mathbf{x}_{j}^{l}\circ W^l\left( \mathbf{r}_j-\mathbf{r}_i \right) .
\end{aligned}
$$

Activation functions: shifted softplus:

$$
\operatorname{ssp}(x)=\ln(0.5 \mathrm{e}^x+0.5).
$$
- ssp(0) = 0.
- Improves the convergence of the network while having infinite order of continuity.

Obtain:
- smooth potential energy surfaces.
- force fields.
- second derivatives that are required for training with forces as well as the calculation of vibrational modes.

## Filter-generating networks
Determines how interactions between atoms are modeled.

Constrain the model and include chemical knowledge.

Input: a fully-connected neural network that takes the vector pointing from atom $i$ to its neighbor $j$.

Rotationally invariant: requirements for modeling molecular energies. Obtained by using interatomic distances:

$$
d_{ij}=\lVert \mathbf{r}_i-\mathbf{r}_j \rVert 
$$

### Rotationally invariant

Filters would be highly correlated: a neural network after initialization is close to linear.

Expand the distances in a basis of Gaussians:

$$
e_k\left( \mathbf{r}_i-\mathbf{r}_j \right) =\exp \left[ -\gamma \left( \lVert \mathbf{r}_i-\mathbf{r}_j \rVert -\mu _k \right) ^2 \right] 
$$

- $\mu_k$: chosen on a uniform grid between zero and the distance cutoff.

The number of Gaussians and the hyper parameter $\gamma$ determine the resolution of the filter.

### Periodic boundary conditions

Each atom-wise feature vector $\mathbf{x}_{i}$ has to be equivalent across all periodic repetitions.

Given a filter $\tilde{W}^l(\boldsymbol{r}_{jb}-\boldsymbol{r}_{ia})$ over all atoms with $\lVert \boldsymbol{r}_{jb}-\boldsymbol{r}_{ia} \rVert < r_{\text{cut}}$:

$$
\begin{aligned}
	\mathbf{x}_{i}^{l+1}&=\mathbf{x}_{im}^{l+1}=\frac{1}{n_{\text{neighbors\,\,}}}\sum_{j,n}{\mathbf{x}_{jn}^{l}}\circ \tilde{W}^l\left( \mathbf{r}_{jn}-\mathbf{r}_{im} \right)\\
	&=\frac{1}{n_{\text{neighbors\,\,}}}\sum_j{\mathbf{x}_{j}^{l}}\circ \underset{W}{\underbrace{\left( \sum_n{\tilde{W}}^l\left( \mathbf{r}_{jn}-\mathbf{r}_{im} \right) \right) }},\\
\end{aligned}
$$
- $a, b$: unit cell.

More stable when normalizing the filter response $\textbf x_i^{l+1}$ by the number of atoms within the cutoff range.

## Property prediction
Compute atom-wise contributions $\hat P_i$ from the fully-connected prediction network.

Calculate the final prediction $\hat P$ by summing (intensive) or averaging (extensive) over the atomic contributions, respectively.

As SchNet yields rotationally invariant energy predictions, the force predictions are rotationally equivariant by construction.

Predicting atomic forces:

$$
\begin{aligned}
	\mathbf{\hat{F}}_i\left( Z_1,...,Z_n,\mathbf{r}_1,...,\mathbf{r}_n \right) &=-\frac{\partial \hat{E}}{\partial \mathbf{r}_i}\left( Z_1,...,Z_n,\mathbf{r}_1,...,\mathbf{r}_n \right).\\
\end{aligned}
$$
## Training

Train SchNet for each property target $P$ by minimizing the squared loss:

$$
\ell(\hat{P}, P)=\|P-\hat{P}\|^2.
$$

Train energies and forces with combined loss:

$$
\left.\ell\left(\left(\hat{E}, \hat{\mathbf{F}}_1, \ldots, \hat{\mathbf{F}}_n\right)\right),\left(E, \mathbf{F}_1, \ldots, \mathbf{F}_n\right)\right)=\rho\|E-\hat{E}\|^2+\frac{1}{n_{\text {atoms }}} \sum_{i=0}^{n_{\text {atoms }}}\left\|\mathbf{F}_i-\left(-\frac{\partial \hat{E}}{\partial \mathbf{R}_i}\right)\right\|^2.
$$
- $\rho$ : trade-off between energy and force loss.

In each experiment, we split the data into a training set of given size $N$ and use a validation set for early stopping.

Remaining data is used for computing the test errors.
