import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost from this node to goal
        self.f = 0  # Total cost (g + h)
        
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

def astar(grid, start, end):
    open_list = []
    closed_list = set()

    start_node = Node(start)
    end_node = Node(end)
    
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)
        
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path
        
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4 possible movements: right, down, left, up
        for move in neighbors:
            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])
            
            if (0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0]) and
                grid[node_position[0]][node_position[1]] == 0):
                
                neighbor_node = Node(node_position, current_node)
                
                if neighbor_node.position in closed_list:
                    continue
                
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = (abs(neighbor_node.position[0] - end_node.position[0]) +
                                   abs(neighbor_node.position[1] - end_node.position[1]))
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                if any(open_node.position == neighbor_node.position and open_node.f <= neighbor_node.f for open_node in open_list):
                    continue
                
                heapq.heappush(open_list, neighbor_node)
    
    return None  # No path found

def print_grid(grid, path):
    
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if (row, col) in path:
                print(' * ', end='')
            else:
                print(f' {grid[row][col]} ', end='')
        print()

def get_grid_input():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    grid = []
    
    print("Enter the grid values row by row (0 for passable, 1 for blocked):")
    for i in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            raise ValueError(f"Row should have {cols} values.")
        grid.append(row)
    
    return grid

def get_position_input(name):
    row = int(input(f"Enter the row for {name} (0-based index): "))
    col = int(input(f"Enter the column for {name} (0-based index): "))
    return (row, col)

def main():
    grid = get_grid_input()
    print("Grid:")
    print_grid(grid, [])
    
    start = get_position_input("start")
    end = get_position_input("end")
    
    if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
        print("Start or end position is blocked.")
        return
    
    path = astar(grid, start, end)
    
    if path:
        print("Path found:")
        print_grid(grid, path)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
