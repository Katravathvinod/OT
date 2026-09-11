import numpy as np

def vogels_approximation_method(cost_matrix, supply, demand):
    """
    Computes Initial Basic Feasible Solution (IBFS) using VAM.
    """
    c = cost_matrix.copy().astype(float)
    s = supply.copy().astype(float)
    d = demand.copy().astype(float)
    m, n = c.shape
    allocation = np.zeros((m, n), dtype=float)

    # Balance check
    if np.sum(s) != np.sum(d):
        raise ValueError(f"Unbalanced problem: Supply sum ({np.sum(s)}) != Demand sum ({np.sum(d)})")

    row_active = [True] * m
    col_active = [True] * n

    while any(row_active) and any(col_active):
        # 1. Compute row penalties (difference between 2 lowest costs)
        row_penalties = []
        for i in range(m):
            if not row_active[i]:
                row_penalties.append(-1)
                continue
            active_costs = [c[i, j] for j in range(n) if col_active[j]]
            if len(active_costs) > 1:
                sorted_c = sorted(active_costs)
                row_penalties.append(sorted_c[1] - sorted_c[0])
            elif len(active_costs) == 1:
                row_penalties.append(active_costs[0])
            else:
                row_penalties.append(-1)

        # 2. Compute column penalties
        col_penalties = []
        for j in range(n):
            if not col_active[j]:
                col_penalties.append(-1)
                continue
            active_costs = [c[i, j] for i in range(m) if row_active[i]]
            if len(active_costs) > 1:
                sorted_c = sorted(active_costs)
                col_penalties.append(sorted_c[1] - sorted_c[0])
            elif len(active_costs) == 1:
                col_penalties.append(active_costs[0])
            else:
                col_penalties.append(-1)

        max_row_pen = max(row_penalties)
        max_col_pen = max(col_penalties)

        # 3. Select line with maximum penalty and find its minimum cost cell
        if max_row_pen >= max_col_pen:
            i = row_penalties.index(max_row_pen)
            active_cols = [j for j in range(n) if col_active[j]]
            j = min(active_cols, key=lambda col: c[i, col])
        else:
            j = col_penalties.index(max_col_pen)
            active_rows = [i for i in range(m) if row_active[i]]
            i = min(active_rows, key=lambda row: c[row, j])

        # 4. Allocate maximum possible quantity
        qty = min(s[i], d[j])
        allocation[i, j] = qty
        s[i] -= qty
        d[j] -= qty

        # 5. Eliminate satisfied rows/columns
        if s[i] == 0:
            row_active[i] = False
        if d[j] == 0:
            col_active[j] = False

    return allocation


# Example Case Study
if __name__ == "__main__":
    cost = np.array([
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40,  8, 70, 20]
    ])
    supply = np.array([7, 9, 18])
    demand = np.array([5, 8, 7, 14])

    ibfs_allocation = vogels_approximation_method(cost, supply, demand)
    total_vam_cost = np.sum(ibfs_allocation * cost)

    print("=== Vogel's Approximation Method (VAM) ===")
    print("Initial Basic Feasible Allocation:")
    print(ibfs_allocation)
    print(f"Total Transportation Cost: {total_vam_cost:.2f}")