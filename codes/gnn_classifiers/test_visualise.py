import os.path as osp

import numpy as np
import pandas as pd
from PIL import ImageColor
import seaborn as sns
import networkx as nx
from matplotlib import pyplot as plt

from torchvision.datasets import MNIST
from torch_geometric.datasets import MNISTSuperpixels
import time

def visualize(image, data):
    plt.figure(figsize=(16, 8))

    # plot the mnist image
    plt.subplot(1, 2, 1)
    plt.title("MNIST")
    np_image = np.array(image)
    plt.imshow(np_image)

    # plot the super-pixel graph
    plt.subplot(1, 2, 2)
    x, edge_index = data.x, data.edge_index

    # construct networkx graph
    df = pd.DataFrame({'source': edge_index[0], 'target': edge_index[1]})
    print(df.head())
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    
    # Investigating the dataset
    print("Dataset type: ", type(data))
    print("Dataset features: ", data.num_features)
    # print("Dataset target: ", data.num_classes)
    # print("Dataset length: ", data.len)
    print("Dataset sample: ", data)
    print("Sample  nodes: ", data.num_nodes)
    print("Sample  edges: ", data.num_edges)
   
    # pos = {i: np.array([data.pos[i][0], data.pos[i][1]]) for i in range(data.num_nodes)}
    # print(pos)
    # flip over the axis of pos, this is because the default axis direction of networkx is different
    pos = {i: np.array([data.pos[i][0], 27 - data.pos[i][1]]) for i in range(data.num_nodes)}
    # print(pos)
    # get the current node index of G
    print(G)
    idx = list(G.nodes())
    print(idx, len(idx))

    # set the node sizes using node features
    print(x[idx])
    # size = x[idx] * 500 + 200
    size = x[idx] * 500 + 500

    # set the node colors using node features
    color = []
    for i in idx:
        grey = x[i]
        if grey == 0:
            color.append('blue')
        else:
            color.append('yellow')

    nx.draw(G, with_labels=False, node_size=size, node_color=color, pos=pos)
    plt.title("MNIST Superpixel")
    
    # Save the figure to the specified output path
    plt.tight_layout()  # Adjust layout to make it fit well
    plt.savefig("./test.png")  # Save the image
    plt.close()  # Close the figure to prevent it from displaying


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
    image_dataset = MNIST(root=path, download=True)
    graph_dataset = MNISTSuperpixels(root=path)

    example = 25
    for example in range(100):
        image, label = image_dataset[example]
        data = graph_dataset[example]
        print("Image Label:", label)
        visualize(image, data)
        time.sleep(2) 