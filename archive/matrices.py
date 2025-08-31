import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Define the graph G
G = nx.Graph({
    0: [1],
    4: [3],
    1: [2, 5, 8],
    2: [3, 6, 9],
    3: [7, 10]
})

# Visualize the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Position nodes using the spring layout
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=500, font_size=10)
plt.title("Graph G Visualization")
plt.show()

# Define the Chemical Equitable Partition (CEP)
CEP = [[0, 4], [1, 3], [2], [6, 9], [5, 7, 8, 10]]

def quotient_matrix(graph, partition):
    n = len(partition)  # Number of parts in the partition
    Q = np.zeros((n, n), dtype=float)  # Initialize the quotient matrix

    for i, part_i in enumerate(partition):
        for j, part_j in enumerate(partition):
            # Count edges between part_i and part_j
            count = sum(1 for u in part_i for v in part_j if graph.has_edge(u, v))
            Q[i, j] = count / len(part_i)  # Normalize by size of part_i (optional)

    return Q

# Compute and display the quotient matrix
Qu = quotient_matrix(G, CEP)
print("Quotient Matrix:")
print(Qu)
