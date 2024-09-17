
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


 2. A* Algorithm

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


 3. Depth-First Search (DFS)

Algorithm:
1. Initialize Data Structures:
   - Use a set `visited` to keep track of visited nodes.
   - Start DFS from the initial node.

2. Recursive DFS Function:
   - Mark the current node as visited.
   - For each neighbor of the current node, recursively perform DFS if it hasn’t been visited yet.

3. Complete Traversal
   - Continue until all reachable nodes have been visited.



 4. Breadth-First Search (BFS)

Algorithm:
1. Initialize Data Structures:
   - Use a set `visited` to keep track of visited nodes.
   - Use a queue to manage the nodes to be processed.

2. Process Queue:
   - While the queue is not empty, dequeue a node and process it.
   - For each unvisited neighbor, mark it as visited and enqueue it.

3. Complete Traversal:
   - Continue until all reachable nodes have been processed.

 5. Cryptarithm Solver

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


 6. 8-Queen Problem

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


