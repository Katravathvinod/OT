import numpy as np

def find_closed_loop(alloc_mask, start_cell):
    """
    Finds an alternating horizontal/vertical loop through basic cells using DFS.
    """
    def dfs(curr, target, path, look_for_horizontal):
        if len(path) > 3 and curr == target:
            return path
        r, c = curr
        if look_for_horizontal:
            for next_c in range(alloc_mask.shape[1]):
                if next_c != c and (alloc_mask[r, next_c] or (r, next_c) == target):
                    if (r, next_c) not in path:
                        res = dfs((r, next_c), target, path + [(r, next_c)], not look_for_horizontal)
                        if res: 
                            return res
        else:
            for next_r in range(alloc_mask.shape[0]):
                if next_r != r and (alloc_mask[next_r, c] or (next_r, c) == target):
                    if (next_r, c) not in path:
                        res = dfs((next_r, c), target, path + [(next_r, c)], not look_for_horizontal)
                        if res: 
                            return res
        return None

    return dfs(start_cell, start_cell, [start_cell], True) or \
           dfs(start_cell, start_cell, [start_cell], False)


def solve_modi(cost_matrix, initial_allocation):
    """
    Optimizes a given BFS using the MODI (u-v) Method.
    """
    cost = cost_matrix.copy().astype(float)
    alloc = initial_allocation.copy().astype(float)
    m, n = cost.shape
    iteration = 1

    while True:
        # Step 1: Calculate dual variables u_i and v_j
        u = [None] * m
        v = [None] * n
        u[0] = 0.0  # Baseline convention

        changed = True
        while changed:
            changed = False
            for i in range(m):
                for j in range(n):
                    if alloc[i, j] > 0:
                        if u[i] is not None and v[j] is None:
                            v[j] = cost[i, j] - u[i]
                            changed = True
                        elif v[j] is not None and u[i] is None:
                            u[i] = cost[i, j] - v[j]
                            changed = True

        # Check for non-connecting components (handles basic degeneracy)
        for i in range(m):
            if u[i] is None:
                u[i] = 0.0

        # Step 2: Calculate opportunity costs d_ij = c_ij - (u_i + v_j) for non-basic cells
        min_delta = 0.0
        pivot_cell = None
        for i in range(m):
            for j in range(n):
                if alloc[i, j] == 0:
                    v_val = v[j] if v[j] is not None else 0.0
                    delta = cost[i, j] - (u[i] + v_val)
                    if delta < min_delta:
                        min_delta = delta
                        pivot_cell = (i, j)

        # Step 3: Optimality check
        if min_delta >= -1e-7:
            print(f"Optimal solution reached at iteration {iteration}.")
            break

        print(f"Iteration {iteration}: Most negative delta = {min_delta:.2f} at cell {pivot_cell}")

        # Step 4: Trace stepping-stone loop
        alloc_mask = alloc > 0
        loop = find_closed_loop(alloc_mask, pivot_cell)
        if not loop:
            raise RuntimeError(f"Could not construct a closed loop for entering cell {pivot_cell}.")

        loop_cells = loop[:-1]
        neg_cells = [loop_cells[k] for k in range(1, len(loop_cells), 2)]
        theta = min(alloc[r, c] for r, c in neg_cells)

        # Step 5: Update allocations along the loop
        for k, (r, c) in enumerate(loop_cells):
            if k % 2 == 0:
                alloc[r, c] += theta
            else:
                alloc[r, c] -= theta

        iteration += 1

    return alloc


# Example Case Study
if __name__ == "__main__":
    cost = np.array([
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40,  8, 70, 20]
    ])

    # Provided Basic Feasible Solution (can be from VAM, NW-Corner, etc.)
    given_bfs = np.array([
        [5., 0., 0., 2.],
        [0., 2., 7., 0.],
        [0., 6., 0., 12.]
    ])

    print("=== MODI Method ===")
    optimal_allocation = solve_modi(cost, given_bfs)
    optimal_cost = np.sum(optimal_allocation * cost)

    print("\nOptimal Allocation Matrix:")
    print(optimal_allocation)
    print(f"Minimum Total Transportation Cost: {optimal_cost:.2f}")