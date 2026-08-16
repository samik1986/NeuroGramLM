# Biological Branching Grammar Tokenization Algorithm

This document outlines the core algorithm used to convert a 3D structural graph (neuron tree) into a linear, scale-independent sequence of vocabulary tokens for Language Model training. 

This approach uses a **Depth-First Search (DFS)** traversal combined with **Vector Quantization (VQ)** to achieve a "SMILES-like" representation of 3D geometry.

## Phase 1: Vector Extraction & Quantization
Since biological volumes may not be perfectly registered to CCFv3 and trees may be floating sub-neurites without a soma, we discard absolute $(X, Y, Z)$ coordinates.

1. **Graph Parsing:** Parse the tree into nodes containing geometry $(X, Y, Z, R)$ and build a directed adjacency list (`parent -> children`).
2. **Relative Vector Computation:** For every edge (from parent $u$ to child $v$), compute the scale/translation independent vector: 
   $$ \Delta \vec{v} = (\Delta X, \Delta Y, \Delta Z, \Delta R) $$
   where $\Delta X = X_v - X_u$, etc.
3. **K-Means Clustering (VQ):** Pool all $\Delta \vec{v}$ vectors across the dataset and run K-Means clustering (e.g., $K=512$). Each cluster center receives an integer ID, forming the geometric vocabulary `<GEO_0>` to `<GEO_511>`.

## Phase 2: Iterative DFS Sequence Generation
To convert the branching tree into a flat string, we use structural control tokens: `<START>`, `<BIF>` (bifurcation/branch point), `<POP>` (backtrack), and `<END>`. 

To prevent stack overflow on extremely deep trees (10,000+ nodes), the sequence generation is implemented as an **Iterative DFS** using a custom action stack.

### Initialization
- Find the root node(s) of the tree/subtree.
- Start the sequence with `["<START>"]`.
- Initialize an execution stack with the action: `PUSH("VISIT", root_node)`.

### Traversal Loop
While the execution stack is not empty:
1. `POP` an action from the top of the stack.
2. **If action is `("TOKEN", string)`:**
   - Append `string` to the final sequence.
3. **If action is `("VISIT", node u)`:**
   - **Cycle Protection:** If $u$ is already in the `visited` set, `CONTINUE` to avoid infinite loops from dirty cyclic graphs.
   - Add $u$ to the `visited` set.
   - Retrieve all children of $u$ that have *not* been visited.
   - If no unvisited children remain, `CONTINUE` (it is a leaf node).
   - Initialize an empty list of `actions_to_push`.
   - **Branching Check:** If $u$ has >1 child, append `("TOKEN", "<BIF>")` to `actions_to_push`.
   - **Iterate over children ($v_0, v_1, ..., v_n$):**
     - Compute the relative vector $\Delta \vec{v}$ between $u$ and $v_i$.
     - Predict the K-Means cluster ID $k$ for $\Delta \vec{v}$.
     - Append `("TOKEN", "<GEO_k>")` to `actions_to_push`.
     - Append `("VISIT", v_i)` to `actions_to_push`.
     - **Backtrack Check:** If this is NOT the last child (i.e., $i < n-1$), append `("TOKEN", "<POP>")` to `actions_to_push`.
   - **Reverse and Push:** Iterate through `actions_to_push` in *reverse* order and `PUSH` them onto the execution stack. *(Reversing ensures that the actions are popped and executed in the correct chronological order).*

### Termination
- Append `["<END>"]` to the sequence.

---

### Sequence Example
Given a root node $A$ that moves to $B$, where $B$ splits into two leaves $C$ and $D$:
1. Start at $A$: `<START>`
2. $A \rightarrow B$: `<GEO_12>`
3. $B$ splits: `<BIF>`
4. $B \rightarrow C$: `<GEO_45>` (Leaf reached, backtrack)
5. Backtrack to $B$: `<POP>`
6. $B \rightarrow D$: `<GEO_88>` (Leaf reached)
7. Tree fully traversed: `<END>`

**Final Sequence:** `[<START>, <GEO_12>, <BIF>, <GEO_45>, <POP>, <GEO_88>, <END>]`
