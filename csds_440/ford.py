import collections

class FordFulkerson:
    def __init__(self, graph):
        self.graph = graph

    def max_flow(self, source, sink):
        # Initialize the residual graph with the same capacities as the original graph
        residual_graph = self.graph.copy()
        
        max_flow = 0  # Initialize the maximum flow
        
        while True:
            # Find an augmenting path in the residual graph using BFS
            path, bottleneck = self.find_augmenting_path(residual_graph, source, sink)
            
            # If there are no more augmenting paths, break the loop
            if not path:
                break
            
            # Update the residual graph with the new flow along the augmenting path
            self.update_residual_graph(residual_graph, path, bottleneck)
            
            # Add the bottleneck capacity to the maximum flow
            max_flow += bottleneck
        
        return max_flow

    def find_augmenting_path(self, residual_graph, source, sink):
        visited = set()
        parent = {}
        queue = collections.deque()
        
        queue.append(source)
        visited.add(source)
        
        while queue:
            current_node = queue.popleft()
            
            if current_node == sink:
                break
            
            for neighbor, capacity in residual_graph[current_node].items():
                if neighbor not in visited and capacity > 0:
                    parent[neighbor] = current_node
                    queue.append(neighbor)
                    visited.add(neighbor)
        
        path = []
        bottleneck = float("inf")
        
        if sink in parent:
            # Reconstruct the augmenting path
            node = sink
            while node != source:
                path.append(node)
                parent_node = parent[node]
                bottleneck = min(bottleneck, residual_graph[parent_node][node])
                node = parent_node
            
            path.append(source)
            path.reverse()
        
        return path, bottleneck

    def update_residual_graph(self, residual_graph, path, bottleneck):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            residual_graph[u][v] -= bottleneck
            residual_graph[v][u] += bottleneck


class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, capacity):
        if u not in self.graph:
            self.graph[u] = {}
        self.graph[u][v] = capacity

    def get_neighbors(self, u):
        return self.graph.get(u, {})

    def __getitem__(self, u):
        return self.get_neighbors(u)

    def copy(self):
        # Create a deep copy of the graph
        copied_graph = Graph()
        for u, neighbors in self.graph.items():
            for v, capacity in neighbors.items():
                copied_graph.add_edge(u, v, capacity)
        return copied_graph


# Example usage:
if __name__ == "__main__":
    # Create a flow network graph represented as a Graph object
    graph = Graph()
    graph.add_edge('s', 'A', 1)
    graph.add_edge('s', 'B', 1)
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 1)
    graph.add_edge('B', 'C', 1)
    graph.add_edge('C', 't', 1)

    ford_fulkerson = FordFulkerson(graph)
    max_flow = ford_fulkerson.max_flow('s', 't')
    print("Maximum Flow:", max_flow)

