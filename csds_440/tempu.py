import numpy as np
#import tensorflow as tf
import networkx as nx

# Generate a random graph
N = 10  # Number of nodes
D = 5   # Dimension of node attributes
graph = nx.erdos_renyi_graph(N, 0.3)  # Random graph

# Assign random attributes to nodes
for node in graph.nodes():
    graph.nodes[node]['attributes'] = np.random.rand(D)

# Adjacency matrix
adj_matrix = nx.adjacency_matrix(graph).todense()

# Extract node attributes into a matrix
node_attributes = np.array([graph.nodes[i]['attributes'] for i in range(N)])

# Define a simple GCN layer
class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphConvLayer, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[input_shape[-1], self.output_dim])

    def call(self, inputs, adjacency_matrix):
        support = tf.matmul(inputs, self.kernel)
        output = tf.matmul(adjacency_matrix, support)
        return tf.nn.relu(output)

# Initialize the GCN layer
gcn_layer = GraphConvLayer(output_dim=8)

# Forward pass through the GCN layer
initial_embeddings = node_attributes
output_embeddings = gcn_layer(initial_embeddings, adj_matrix)

# Display the final node embeddings
print("Final Node Embeddings:")
print(output_embeddings.numpy())
