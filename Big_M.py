import numpy as np

def big_m_simplex():
    # Large penalty constant M
    M = 1e4
    
    # Problem definition: Maximize Z' = -4*x1 - x2 - M*A1 - M*A2
    # Variables order: [x1, x2, S1, s2, A1, A2]
    c = np.array([-4.0, -1.0, 0.0, 0.0, -M, -M])
    
    # Constraint matrix:
    # 3*x1 + 1*x2 + 0*S1 + 0*s2 + 1*A1 + 0*A2 = 3
    # 4*x1 + 3*x2 - 1*S1 + 0*s2 + 0*A1 + 1*A2 = 6
    # 1*x1 + 2*x2 + 0*S1 + 1*s2 + 0*A1 + 0*A2 = 4
    A = np.array([
        [3.0, 1.0,  0.0, 0.0, 1.0, 0.0],
        [4.0, 3.0, -1.0, 0.0, 0.0, 1.0],
        [1.0, 2.0,  0.0, 1.0, 0.0, 0.0]
    ])
    b = np.array([3.0, 6.0, 4.0])
    
    # Initial basic feasible solution indices: A1 (4), A2 (5), s2 (3)
    basis = [4, 5, 3]
    var_names = ["x1", "x2", "S1", "s2", "A1", "A2"]
    
    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    
    # Calculate initial row 0: z_j - c_j = c_B * B^(-1) * A_j - c_j
    c_B = c[basis]
    tableau[-1, :n] = np.dot(c_B, tableau[:m, :n]) - c
    tableau[-1, -1] = np.dot(c_B, tableau[:m, -1])
    
    iteration = 0
    max_iter = 100
    
    while iteration < max_iter:
        # Optimality condition: all reduced costs (z_j - c_j) >= 0 (for maximization)
        reduced_costs = tableau[-1, :n]
        if np.all(reduced_costs >= -1e-6):
            break
            
        # Entering column: most negative reduced cost
        pivot_col = np.argmin(reduced_costs)
        
        # Minimum ratio test
        ratios = []
        for i in range(m):
            val = tableau[i, pivot_col]
            rhs = tableau[i, -1]
            if val > 1e-6:
                ratios.append(rhs / val)
            else:
                ratios.append(np.inf)
                
        ratios = np.array(ratios)
        if np.all(np.isinf(ratios)):
            raise ValueError("Problem is unbounded.")
            
        pivot_row = np.argmin(ratios)
        
        # Update basis
        basis[pivot_row] = pivot_col
        
        # Pivot operation
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
                
        iteration += 1

    # Extract solution
    solution = np.zeros(n)
    for i, b_idx in enumerate(basis):
        solution[b_idx] = tableau[i, -1]
        
    x1, x2 = solution[0], solution[1]
    min_z = 4 * x1 + 1 * x2
    
    print("=== Big-M Simplex Results ===")
    print(f"Optimal x1: {x1:.4f}")
    print(f"Optimal x2: {x2:.4f}")
    print(f"Optimal Minimum Z: {min_z:.4f}")
    print(f"Slack (s2): {solution[3]:.4f}")
    print(f"Surplus (S1): {solution[2]:.4f}")

big_m_simplex()