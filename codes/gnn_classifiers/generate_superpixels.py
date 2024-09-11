import torch
from skimage.segmentation import slic
from torchvision.datasets import MNIST, FashionMNIST
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time
from PIL import Image, ImageDraw

# Global constants
NUM_FEATURES = 3
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

def generate_graph_from_image(image: torch.tensor, nodes: int=75, slic_zero: bool=True, grayscale: bool = True):
    # Check if the image is grayscale
    if grayscale or image.shape[0] == 1:
        channel_axis = False
    else:
        channel_axis = -1
    
    segments = slic(image, n_segments=nodes, slic_zero=slic_zero, channel_axis=channel_axis)
    segments = np.array(segments)   
    print(segments)
    print(segments.shape)

    number_nodes = np.max(segments)
    print(number_nodes)
   
    nodes = {
        node : {
           "intensity_list": list([]) if grayscale else None,  # Store intensity values for grayscale
           "rgb_list": list([]) if not grayscale else None,  # Store RGB values for color
           "pos_list": list([]),
        } for node in range(number_nodes+1)
    }
   
    height = image.shape[1]
    width = image.shape[2]
    
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            pos = np.array([float(x) / width, float(y) / height])
            nodes[node]["pos_list"].append(pos)
            
            if grayscale:
                intensity = image[0, y, x]  # Access the grayscale intensity correctly
                nodes[node]["intensity_list"].append(intensity)
            else:
                rgb = image[y, x, :]  # For RGB images
                nodes[node]["rgb_list"].append(rgb)
    
    G = nx.Graph()

    for node in nodes:
        if len(nodes[node]["pos_list"]) == 0:
            # Skip nodes with no pixels
            continue

        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])

        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)

        # Calculate features
        if grayscale:
            intensity_mean = np.mean(nodes[node]["intensity_list"])
            features = np.concatenate([np.array([intensity_mean]), np.reshape(pos_mean, -1)])
        else:
            rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
            features = np.concatenate([np.reshape(rgb_mean, -1), np.reshape(pos_mean, -1)])

        # Add node with features to the graph
        G.add_node(node, features=list(features))

    # Generate edges
    segments_ids = np.unique(segments)
    print(segments_ids)

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    # Self-loops
    for node in nodes:
        G.add_edge(node, node)
        
    # Remove non-connected nodes
    isolated_nodes = list(nx.isolates(G))
    print(f"Removing isolated nodes: {isolated_nodes}")
    G.remove_nodes_from(isolated_nodes)
    
    # Remove node 0 explicitly
    # if 0 in G:
    #     print("Removing node 0 from the graph")
    #     G.remove_node(0)


    # Prepare final output
    n = len(G.nodes)
    m = len(G.edges)
    NUM_FEATURES = len(G.nodes[1]["features"])
    h = np.zeros([n, NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2 * m, 2]).astype(NP_TORCH_LONG_DTYPE)

    for e, (s, t) in enumerate(G.edges):
        edges[e, 0] = s
        edges[e, 1] = t
        edges[m + e, 0] = t
        edges[m + e, 1] = s
        
    print(G.nodes(data=True)) 
    
    for i in G.nodes:
        if "features" in G.nodes[i] and len(G.nodes[i]["features"]) > 0:
            features = np.array(G.nodes[i]["features"], dtype=NP_TORCH_FLOAT_DTYPE)
            if features.shape[0] == h.shape[1]:
                h[i, :] = features
            else:
                print(f"Feature length mismatch for node {i}")
        else:
            print(f"Node {i} does not have valid 'features'")

    # del G
    return G, h, edges

def visualize(image, graph, h):
    fig = plt.figure(figsize=(16, 8))

    # Plot the MNIST image
    plt.subplot(1, 2, 1)
    # plt.title("MNIST Image")
    
    # Remove the channel dimension if it's a grayscale image (shape 1, 28, 28)
    if image.shape[0] == 1:
        np_image = np.squeeze(image, axis=0)
    else:
        np_image = image
    
    plt.imshow(np_image, cmap="gray")

    # Plot the graph
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(graph)  # Generate a layout for the graph
    # Invert the y-coordinates to match the image orientation
    pos = {k: (v[0], -v[1]) for k, v in pos.items()}
    node_colors = ['black' if np.mean(h[i]) > 0.8 else 'white' for i in graph.nodes]
    node_sizes = [ max(np.mean(h[i]), 0.8) * 25 for i in graph.nodes]

    nx.draw(graph, pos, node_color=node_colors, node_size=node_sizes, with_labels=False)
    # plt.title("Superpixel Graph")

    # Save and display the visualization
    plt.tight_layout()
    plt.savefig("./graph_visualization.png")
    plt.show()
    plt.close()
    
    return fig
    
    
    
if __name__ == "__main__":
    frames = []  # List to store the frames for the GIF
    dataset = MNIST(root="./data", download=True, train=False)
    
    for i in range(10):    
        image = np.array(dataset.__getitem__(i)[0])

        # Grayscale images already have a single channel
        if len(image.shape) == 2:  # It's grayscale
            image = np.expand_dims(image, axis=0)
            grayscale = True
        else:
            grayscale = False

        print(image.shape)
        G, h, edges = generate_graph_from_image(image, grayscale=grayscale)
        print(h.shape, edges.shape)
        print(h, edges)
        
        G.remove_node(0)
        
        # Visualize the graph on the image and save it as a frame
        fig = visualize(image, G, h)
        fig.canvas.draw()  # Draw the image on the canvas

        # Convert the Matplotlib figure to an image array (RGB format)
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img)
        
        # Add the PIL image to the list of frames
        frames.append(pil_img)
        
        plt.close(fig)  # Close the figure to save memory

        time.sleep(2)
    
    # Save the frames as a GIF
    frames[0].save(
        'MNIST_superpixels.gif', 
        format='GIF',
        append_images=frames[1:], 
        save_all=True, 
        duration=200,  # Duration between frames in milliseconds
        loop=0  # 0 means loop forever
    ) 