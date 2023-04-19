# MolProPred
* Dealing with [kaggle's competition-Predicting Molecular Properties](https://www.kaggle.com/competitions/champs-scalar-coupling/overview).

* Project name given by chatGPT. (prompt: Please give me an ultra-cool project name, the project is for coping with an assignment given by a course '[machine learning for physicist](https://github.com/wangleiphy/ml4p)', there are 3 team members, and the project is about predicting molecular properties.)

* The phonetic symbols for MolProPred are /mɑlproʊprɛd/.

* Good luck. :-)

* AI extensions used in this project: [Genie-AI](https://github.com/ai-genie/chatgpt-vscode), [Github Copilot](https://docs.github.com/en/copilot) for code testing, [New Bing](https://www.bing.com/new?cc=sg&setlang=zh-hans) for literature search, [chatgpt_academic](https://github.com/binary-husky/chatgpt_academic) for fun.

# Introduction to Message Passing Graph Neural Network

Graph Neural Network (GNN) is a type of neural network that can operate on graph data. It has become increasingly popular in recent years due to its ability to model complex relationships between entities in a graph. Message passing is a fundamental operation in GNN, which enables the network to propagate information between nodes in the graph. 



## Embedding a Graph

The first step in building a message passing GNN is to embed the graph into a low-dimensional space. This is done by assigning each node in the graph a vector representation, which captures its features and attributes. There are various methods for graph embedding, including random initialization, spectral methods, and neural network-based methods. Neural network-based methods, such as Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), have shown superior performance in many tasks.

## Message Passing Function

Once the graph is embedded, the message passing function is applied to propagate information between nodes. The message passing function takes as input the feature vectors of a node and its neighboring nodes and produces an updated feature vector for the node. The function can be defined as a neural network layer or a set of mathematical equations. The most common message passing functions are based on the graph convolution operation, which applies a filter to the node features and its neighbors' features.

## Readout Operation

After several rounds of message passing, the final step is to perform a readout operation to obtain a graph-level representation. The readout operation aggregates the node features into a single vector, which represents the entire graph. There are various methods for readout operation, including sum pooling, max pooling, and attention-based pooling. The choice of readout operation depends on the task at hand and the characteristics of the graph.

## Conclusion

In summary, message passing GNN is a powerful tool for modeling graph data. It enables the network to capture complex relationships between entities in the graph and make predictions based on the graph structure. In this post, we have introduced the concept of message passing GNN, including how to embed a graph, what message passing functions can do, and how to operate readout. We hope that this post provides a useful introduction to this exciting field and encourages further exploration.
