![logo](https://github.com/RootHarold/LycorisR/blob/master/logo/logo.svg)

**LycorisR** is a lightweight recommendation algorithm framework based on [**LycorisNet**](https://github.com/RootHarold/Lycoris).

# Features
* Sequence-based recommendation.
* Discover correlations between items.
* Elegant and minimalist.

# Installation
The project is based on LycorisNet, and the installation of LycorisNet can be found [**here**](https://github.com/RootHarold/Lycoris#Installation).

```
pip install LycorisR
```

# Documents
The APIs provided by **Recommender** (`from LycorisR import Recommender`):

Function | Description |  Inputs | Returns
-|-|-|-
**Recommender**(config) | Constructor. | **config**: The configuration information, including 11 configuration fields. | An object of the class Recommender.
**embed**(data) | Generate the mapping between one-hot encoding and embedding vectors. | **data**: Each dimension of data is spliced by several one-hot codes. For example, the encoding of "A B-> C:" can be expressed as: [0, 0, 1, 0, 1, 0, 1, 0, 0]. | 
**most_similarity**(positive, negative=None, top_n=10) | Calculate and get the most similar top_n items. | **positive**: List of items that contribute positively.<br/>**negative**: List of items that contribute negatively.<br/>**top_n**: Top N. | The most similar top_n items and their similarities.
**infer**(items) | Inferring based on the trained model. | **items**: Each dimension of data is spliced by several one-hot codes. | One-hot codes of desired items.
**get_vector**(items) | Get the embedding vectors of items. | **items**: one-hot codes. | Embedding vectors of items.
**save**(path1, path2) | Save the model and related configurations. | **path1**: The path to store the model.<br/> **path2**: The path to store the configurations. |
`@staticmethod`<br/>**load**(path1, path2) | Import pre-trained models and related configurations. | **path1**: The path to import the model.<br/> **path2**: The path to import the configurations. |
**set_config**(config) | Set the configuration information of Recommender. | **config**: The configuration information, including 11 configuration fields. |
**set_lr**(learning_rate) | Set the learning rate of the neural network. | **learning_rate**: The learning rate of the neural network. | 
**set_workers**(workers) | Set the number of worker threads to train the model. | **workers**: The number of worker threads. | 
`@staticmethod`<br/>**version**() |  |  | Returns the version information of Recommender.

Configurable fields:

Field | Description |Default
-|-|-
capacity | Capacity of LycorisNet. |
dimension | Dimension of each item. |
sequence | The number of items. | 
nodes | The number of hidden nodes added for each neural network. |
connections| The number of connections added for each neural network. |
depths| Total layers of each neural network. |
batch_size| Batch size. |
epoch| Epoch. |
middle_layer | Select the number of layers as the embedding vector.<br/>It starts at index 0. | 
evolution| Number of LycorisNet evolutions. | 0
verbose| Whether to output intermediate information. | False

# Usage
LycorisR is a sequence-based recommendation framework. The input data is a sequential list of items, and the framework automatically explores correlations between these items. Here is a simple example with 8 groups of sequences: `A B -> C`, `A B -> D`, `B A -> C`, `B A -> D`, `C D -> A`, `C D -> B`, `D C -> A`, `D C -> B`.

Import the dependent modules:

```python
from LycorisR import Recommender
```

Prepare the data:

```python
data = [[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
```

Prepare the configuration information and instantiate the **Recommender** object:

```python
conf = {"capacity": 64, "dimension": 4, "sequence": 3, "nodes": 1000, "connections": 30000, "depths": 8,
        "batch_size": 8, "epoch": 16, "evolution": 8, "middle_layer": 4, "verbose": True}
lre = Recommender(conf)
```

Embedding:

```python
lre.embed(data)
```

Inferring `B C -> ?`:

```python
lre.infer([0, 0, 1, 0, 0, 1, 0, 0])
```

Get the most similar items to "A":

```python
lre.most_similarity([0, 0, 0, 1])
```

Get the embedding vectors of "D":

```python
lre.get_vector([1, 0, 0, 0])
```

*More examples will be released in the future.*

# License
LycorisR is released under the [LGPL-3.0](https://github.com/RootHarold/LycorisR/blob/master/LICENSE) license. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.