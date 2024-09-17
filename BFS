from collections import defaultdict

def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        node = queue.pop(0)
        print(node, end=" ")
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

def main():
    graph = defaultdict(list)
    
    # Input number of nodes
    num_nodes = int(input("Enter the number of nodes: "))
    
    # Input edges
    print("Enter edges in the format 'node1 node2'. Type 'done' when finished.")
    while True:
        edge_input = input()
        if edge_input.lower() == 'done':
            break
        node1, node2 = map(int, edge_input.split())
        graph[node1].append(node2)
        graph[node2].append(node1)  # Assuming undirected graph; remove this line for directed graph
    
    # Input start node
    start_node = int(input("Enter the start node: "))
    
    print("BFS traversal:")
    bfs(graph, start_node)

if __name__ == "__main__":
    main()
