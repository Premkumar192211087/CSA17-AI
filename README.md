
 1. Water Jug Problem (BFS Algorithm)

Algorithm:
1. Initialize Data Structures:
   - Create a set `visited` to keep track of visited states.
   - Initialize a queue with the starting state `(0, 0)` (both jugs are empty).

2. Process Queue:
   - While the queue is not empty, dequeue the front element.
   - Check if either jug contains the target amount of water. If yes, return `True`.

3. Generate Possible States:
   - For each possible action (fill, empty, pour between jugs), generate the new state.
   - If the new state has not been visited before, enqueue it and mark it as visited.

4. Check for Solution:
   - If the queue becomes empty without finding the target, return `False`.



 
 A* Algorithm

Algorithm:
1. Initialize Data Structures:**
   - Create a priority queue (min-heap) for the open list with the starting node.
   - Maintain dictionaries for the cost `g_score` and estimated total cost `f_score`.
   - Create a dictionary to keep track of the path from the start to each node.

2. Process Open List:
   - While the open list is not empty, dequeue the node with the lowest `f_score`.
   - If the goal node is reached, reconstruct and return the path.

3. Generate and Evaluate Neighbors:
   - For each neighbor, calculate tentative `g_score` and update if it's better than the previously known score.
   - Update the `f_score` for each neighbor and enqueue them in the open list.

4. Check for Solution:
   - If the open list becomes empty without reaching the goal, return `None`.



 
  Depth-First Search (DFS)
Algorithm:
1. Initialize Data Structures:
   - Use a set `visited` to keep track of visited nodes.
   - Start DFS from the initial node.

2. Recursive DFS Function:
   - Mark the current node as visited.
   - For each neighbor of the current node, recursively perform DFS if it hasn’t been visited yet.

3. Complete Traversal
   - Continue until all reachable nodes have been visited.





 Breadth-First Search (BFS)

Algorithm:
1. Initialize Data Structures:
   - Use a set `visited` to keep track of visited nodes.
   - Use a queue to manage the nodes to be processed.

2. Process Queue:
   - While the queue is not empty, dequeue a node and process it.
   - For each unvisited neighbor, mark it as visited and enqueue it.

3. Complete Traversal:
   - Continue until all reachable nodes have been processed.



 Cryptarithm Solver

Algorithm:
1. Generate Permutations:
   - Generate all permutations of digits for the letters in the cryptarithm.

2. Test Permutations:
   - For each permutation, create a mapping from letters to digits.
   - Substitute the letters in the cryptarithm expression with digits and evaluate.

3. Check Solution Validity:
   - Check if the evaluated expression satisfies the cryptarithm condition.

4. Return Solution:
   - Return the first valid mapping found or `None` if no valid mapping exists.




 8-Queen Problem

Algorithm:
1. Initialize Board:
   - Create an empty `n x n` board.

2. Backtracking Function:
   - Try placing a queen in each row of the current column.
   - Check if placing the queen is safe using `is_safe`.

3. Recursive Placement:
   - If placing the queen is safe, recursively attempt to place queens in the next column.
   - If placing queens in all columns is successful, return the board configuration.

4. Backtrack:
   - If a placement leads to no solution, backtrack by removing the queen and trying the next position.
  



 Map Coloring with Constraint Satisfaction Problems (CSP)

**Algorithms:**

1. **Backtracking Algorithm:**
   - **Description:** This is a depth-first search approach where you try to assign a color to a region and recursively try to color the rest of the map. If you hit a conflict (i.e., two adjacent regions have the same color), you backtrack and try a different color.
   - **Steps:**
     1. Assign a color to a region.
     2. Move to the next region and try to assign a valid color.
     3. If you can't assign a color that satisfies all constraints, backtrack and try a different color for previous regions.
     4. Repeat until the map is properly colored or all options are exhausted.

2. **Forward Checking:**
   - **Description:** This is an enhancement to backtracking. It involves checking ahead to ensure that assigning a color to a region does not lead to a situation where no valid color can be assigned to future regions.
   - **Steps:**
     1. Assign a color to a region.
     2. Update the domains of the neighboring regions (i.e., remove the assigned color from their possible choices).
     3. Proceed to the next region and repeat the process.

3. **Constraint Propagation:**
   - **Description:** This is another enhancement where you propagate the constraints through the CSP. Techniques like Arc Consistency (e.g., AC-3 algorithm) can be used to reduce the search space.
   - **Steps:**
     1. Apply arc consistency algorithms to reduce the possible colors for each region.
     2. Use backtracking or forward checking in conjunction with reduced domains.


 
 
Traveling Salesman Problem (TSP)
**Algorithms:**

1. **Exact Algorithms:**
   - **Brute Force:**
     - **Description:** Evaluate all possible permutations of the cities and choose the one with the minimal distance.
     - **Complexity:** O(n!), where n is the number of cities.

   - **Dynamic Programming (Held-Karp Algorithm):**
     - **Description:** Uses a dynamic programming approach to solve TSP with better efficiency than brute force.
     - **Complexity:** O(n^2 * 2^n), where n is the number of cities.

   - **Branch and Bound:**
     - **Description:** A tree-based search algorithm that systematically explores all possible routes but prunes branches that cannot yield a better solution than the current best.
     - **Complexity:** Varies depending on the implementation and problem constraints.

2. Approximation Algorithms:
   - **Nearest Neighbor:**
     - **Description:** Start from an arbitrary city and repeatedly visit the nearest unvisited city until all cities are visited.
     - **Complexity:** O(n^2), where n is the number of cities.
     - **Quality:** Provides a solution that is not guaranteed to be optimal but is simple and fast.

   - **Christofides' Algorithm:**
     - **Description:** Provides a solution that is guaranteed to be within 3/2 of the optimal solution for metric TSP (where the triangle inequality holds).
     - **Complexity:** O(n^3), where n is the number of cities.

 
 
 
 Tic-Tac-Toe Game

**Tic-Tac-Toe Problem:** This is a classic game where two players (X and O) take turns marking spaces in a 3x3 grid. The goal is to get three of their marks in a row, column, or diagonal.

**Algorithms:**

1. **Minimax Algorithm:**
   - **Description:** A recursive algorithm for choosing the optimal move in a zero-sum game. It considers all possible moves and their outcomes, assuming both players play optimally.
   - **Steps:**
     1. Generate all possible moves for the current player.
     2. For each move, recursively evaluate the outcome using minimax, assuming the opponent also plays optimally.
     3. Choose the move that maximizes the current player's chances of winning or minimizes losses.

2. **Alpha-Beta Pruning:**
   - **Description:** An optimization technique for the minimax algorithm that reduces the number of nodes evaluated in the search tree by "pruning" branches that don't need to be explored.
   - **Steps:**
     1. Use alpha and beta values to keep track of the best scores for the maximizing and minimizing players.
     2. Prune branches that cannot affect the final decision.

3. **Heuristic Evaluation (for larger grids or variants):**
   - **Description:** For more complex versions of Tic-Tac-Toe or similar games, heuristic evaluation functions can be used to estimate the desirability of a board state.
   - **Steps:**
     1. Define a heuristic function that evaluates the board state.
     2. Use this heuristic in conjunction with search algorithms to guide the decision-making process.





Feedforward Neural Network Algorithm

Algorithm: Feedforward Neural Network

1. Initialization:
   - Set the input size, hidden size, output size, and learning rate.
   - Initialize weights between input and hidden layers with random values.
   - Initialize biases for the hidden and output layers as zeros.
2. Activation Function:
   - Define the sigmoid activation function:
     - \( \text{sigmoid}(z) = \frac{1}{1 + e^{-z}} \)
   - Define the derivative of the sigmoid function.
3. Forward Propagation:
   - For each input sample:
     - Compute hidden layer input: \( \text{hidden\_input} = X \cdot W_{\text{input-hidden}} + b_{\text{hidden}} \)
     - Compute hidden layer output using the sigmoid function: \( \text{hidden\_output} = \text{sigmoid}(\text{hidden\_input}) \)
     - Compute final layer input: \( \text{final\_input} = \text{hidden\_output} \cdot W_{\text{hidden-output}} + b_{\text{output}} \)
     - Compute final output using the sigmoid function: \( \text{final\_output} = \text{sigmoid}(\text{final\_input}) \)
4. Loss Calculation:
   - Compute the mean squared error between predicted outputs and true labels:
     - \( \text{loss} = \frac{1}{N} \sum (y_{\text{true}} - y_{\text{pred}})^2 \)
5. Backward Propagation:
   - Calculate the error at the output layer: \( \text{output\_error} = y_{\text{pred}} - y_{\text{true}} \)
   - Compute the gradient (delta) for the output layer: \( \text{output\_delta} = \text{output\_error} \cdot \text{sigmoid\_derivative}(\text{final\_output}) \)
   - Compute the error for the hidden layer: \( \text{hidden\_error} = \text{output\_delta} \cdot W_{\text{hidden-output}}^T \)
   - Compute the gradient for the hidden layer: \( \text{hidden\_delta} = \text{hidden\_error} \cdot \text{sigmoid\_derivative}(\text{hidden\_output}) \)
6. Weights and Biases Update:
   - Update the weights and biases using the learning rate:
     - \( W_{\text{hidden-output}} -= \text{hidden\_output}^T \cdot \text{output\_delta} \cdot \text{learning rate} \)
     - \( b_{\text{output}} -= \sum \text{output\_delta} \cdot \text{learning rate} \)
     - \( W_{\text{input-hidden}} -= X^T \cdot \text{hidden\_delta} \cdot \text{learning rate} \)
     - \( b_{\text{hidden}} -= \sum \text{hidden\_delta} \cdot \text{learning rate} \)
7. Training:
   - Repeat the forward propagation, loss calculation, and backward propagation for a specified number of epochs.
8. Prediction:
   - Use the forward propagation function to get the output for new input samples.




Minimax Algorithm with Alpha-Beta Pruning

Algorithm: Minimax with Alpha-Beta Pruning
1. Initialization:
   - Define constants `MAX` and `MIN` for comparison.
2. Minimax Function:
   - Define a recursive function `minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta)`:
     - Base Case:
       - If `depth` equals the maximum depth (e.g., 3), return the value of the node from the `values` array.
     - Maximizing Player:
       - Initialize `best` to `MIN`.
       - For each child node (2 children in binary tree):
         - Recursively call `minimax` for the child node.
         - Update `best` with the maximum value.
         - Update `alpha` with the maximum of `alpha` and `best`.
         - If `beta` is less than or equal to `alpha`, break (prune the search).
     - Minimizing Player:
       - Initialize `best` to `MAX`.
       - For each child node:
         - Recursively call `minimax` for the child node.
         - Update `best` with the minimum value.
         - Update `beta` with the minimum of `beta` and `best`.
         - If `beta` is less than or equal to `alpha`, break (prune the search).
3. Main Function:
   - Get user input for the number of leaf nodes and their values.
   - Call the `minimax` function starting from the root node (index 0) and the initial values for `alpha` and `beta`.
4. Output:
   - Print the optimal value obtained from the minimax function.



 Decision Tree Algorithm (ID3)
1. Initialization:
   - Start with the entire dataset as the root node of the tree.
   - Define the maximum depth of the tree (if applicable).
2. Stopping Criteria:
   - If all examples in the dataset belong to the same class, create a leaf node and label it with that class.
   - If there are no more features to split on, create a leaf node and label it with the majority class of the dataset.
   - If the maximum depth is reached, create a leaf node and label it with the majority class.
3. Feature Selection:
   - For each feature in the dataset:
     - Calculate the Information Gain or Gini Index for all possible splits of the dataset based on that feature.
     - Select the feature that provides the highest Information Gain (or lowest Gini Index).
4. Splitting:
   - Split the dataset into subsets based on the selected feature and its threshold value (for continuous features).
   - For each subset, repeat the process recursively:
     - Create a child node for each subset.
     - Call the decision tree algorithm on the subset.
5. Recursion:
   - Continue the process for each child node until reaching the stopping criteria.
6. Leaf Node Creation:
   - Once the stopping criteria are met, label the node with the class determined by the majority class in the subset.
7. Tree Traversal for Prediction:
   - To predict the class for a new instance, start at the root node:
     - Traverse the tree based on the feature values of the instance.
     - Follow the branches corresponding to the feature values until reaching a leaf node.
     - Return the class label at the leaf node as the predicted class for the instance.
    
   Sure! Below are programming algorithms for the 8-puzzle, Cannibal Problem, and Vacuum Cleaner problem. Each algorithm is described in a step-by-step manner, focusing on clarity.




8-Puzzle (Using A* Search)
Algorithm
1. Define the State: Represent the puzzle as a 3x3 array.
2. Heuristic Function: Implement a heuristic function (e.g., Manhattan distance).
3. Priority Queue: Use a priority queue to store and retrieve states based on their cost (g + h).
4. Explore States:
   - Initialize the priority queue with the initial state.
   - While the queue is not empty:
     - Dequeue the state with the lowest cost.
     - If it’s the goal state, return the solution.
     - Generate possible moves (up, down, left, right).
     - For each move, calculate the new state and its cost.
     - If the new state hasn’t been visited, add it to the queue.


 
 
 Cannibal Problem (Using Backtracking)
Algorithm
1. State Representation: Define a state as a tuple (left_cannibals, left_missionaries, boat_position).
2. Base Case: Check if all missionaries and cannibals are on the right side.
3. Generate Moves: Generate all possible valid moves (1 or 2 people) from the left to the right.
4. Check Validity: Ensure that cannibals do not outnumber missionaries on either side after the move.
5. Recursive Backtracking:
   - For each valid move, recursively call the function with the new state.
   - If a solution is found, return it.



 
 
 Vacuum Cleaner Problem (Reflex Agent)

 Algorithm
1. Define the Environment: Represent the grid as a 2D array, with each cell indicating if it's dirty or clean.
2. Agent Action:
   - If the current cell is dirty, clean it.
   - Otherwise, move to an adjacent cell (choose based on some strategy).
3. Loop: Continue until all cells are clean.



