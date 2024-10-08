import torch
from skimage.segmentation import slic
from torchvision.datasets import MNIST
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

NUM_FEATURES = 3
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

def generate_graph_from_image(image: torch.tensor, nodes: int=75, slic_zero: bool=True, grayscale: bool = True):
  
    if image.shape[0] == 1:
        grayscale = True
    segments = slic(image, n_segments=nodes, slic_zero=slic_zero, channel_axis=False)
    segments = np.array(segments)   
    print(segments)
    print(segments.shape)

    number_nodes = np.max(segments)
    print(number_nodes)
   
    nodes = {
        node : {
           "rgb_list": list([]),
           "pos_list": list([]),
        } for node in range(number_nodes+1)
    }
   
    print(np.all(image==0))
    height = image.shape[1]
    width = image.shape[2]
    # height, width = 28, 28
    for y in range(height):
        for x in range(width): 
            node = segments[y,x]
            rgb = image[y,x,:]
            pos = np.array([float(x)/width, float(y)/height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)

    for node in nodes:
        # if not np.all(nodes[node]["rgb_list"][0]==0.):
        #     print(nodes[node]["rgb_list"]) 
        # print(nodes[node]["pos_list"])
        if len(nodes[node]["rgb_list"]) == 0:
            continue
        print("here")
        print(np.stack(nodes[node]["pos_list"]))
   
    G = nx.Graph()
    
    for node in nodes:
        if len(nodes[node]["rgb_list"]) == 0 or len(nodes[node]["pos_list"]) == 0:
            # Skip nodes with no pixels
            continue
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        #rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        #rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        #pos_std = np.std(nodes[node]["pos_list"], axis=0)
        #pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        # Debug
        
        print(rgb_mean, pos_mean)
        
        features = np.concatenate(
          [
            np.reshape(rgb_mean, -1),
            #np.reshape(rgb_std, -1),
            #np.reshape(rgb_gram, -1),
            np.reshape(pos_mean, -1),
            #np.reshape(pos_std, -1),
            #np.reshape(pos_gram, -1)
          ]
        )
        
        # print(features.shape)
        G.add_node(node, features = list(features))
    #end
    
    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)
    print(segments_ids)

    print(np.mean(np.nonzero(segments==28), axis=1))
    print([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    print(centers)

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            G.add_edge(bneighbors[0,i],bneighbors[1,i])
    
    # Self loops
    for node in nodes:
        G.add_edge(node,node)
    
    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n,NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2*m,2]).astype(NP_TORCH_LONG_DTYPE)
    
    print(G.nodes(data=True))
    
    for e,(s,t) in enumerate(G.edges):
        edges[e,0] = s
        edges[e,1] = t
        
        edges[m+e,0] = t
        edges[m+e,1] = s
    #end for
    # for i in G.nodes:
        # # features = np.array(G.nodes(data=True)[0][i][i]["features"], dtype=NP_TORCH_FLOAT_DTYPE)
        # h[i, :] = features
        # h[i,:] = G.nodes[i]["features"]
        
    for i in G.nodes:
        if "features" in G.nodes[i] and len(G.nodes[i]["features"]) > 0:
            features = np.array(G.nodes[i]["features"], dtype=NP_TORCH_FLOAT_DTYPE)
            if features.shape[0] == h.shape[1]:
                h[i, :] = features
            else:
                print(f"Feature length mismatch for node {i}")
        else:
            print(f"Node {i} does not have valid 'features'")
    #end for
    del G
    return h, edges

def plot_graph_from_image(image,desired_nodes=75,save_in="./test_superpixel.png"):
    segments = slic(image, slic_zero = True)

    # show the output of SLIC
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(image, segments), cmap="gray")
    ax.imshow(image)#, cmap="gray")
    plt.axis("off")

    asegments = np.array(segments)

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    plt.scatter(centers[:,1],centers[:,0], c='r')

    for i in range(bneighbors.shape[1]-1):
        y0,x0 = centers[bneighbors[0,i]]
        y1,x1 = centers[bneighbors[1,i]]

        l = Line2D([x0,x1],[y0,y1], c="r", alpha=0.5)
        ax.add_line(l)

    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in,bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    
    dataset = MNIST(root = "./data", download=True, train=False)
    image = np.array(dataset.__getitem__(10)[0])
    if len(image.shape) <= 2: 
        image = np.expand_dims(image, axis=0)
        image = np.tile(image, (3, 1, 1))
        image = np.transpose(image, (1, 2, 0))
    print(image.shape)
    h, edges = generate_graph_from_image(image)
    print(h.shape, edges.shape)
    print(h, edges) 
   
    plot_graph_from_image(image) 