MAX, MIN = 1000, -1000

def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]
    
    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

def get_user_input():
    num_leaf_nodes = int(input("Enter the number of leaf nodes (must be a power of 2): "))
    if (num_leaf_nodes & (num_leaf_nodes - 1)) != 0:
        raise ValueError("The number of leaf nodes must be a power of 2.")
    
    values = []
    print("Enter the values for the leaf nodes:")
    for i in range(num_leaf_nodes):
        value = int(input(f"Value for leaf node {i + 1}: "))
        values.append(value)

    return values

if __name__ == "__main__":
    try:
        values = get_user_input()
        optimal_value = minimax(0, 0, True, values, MIN, MAX)
        print("The optimal value is:", optimal_value)
    except ValueError as e:
        print(e)
