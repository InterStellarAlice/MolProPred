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
