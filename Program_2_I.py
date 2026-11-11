from math import *

# Helper function to evaluate linear equations and extract matrix A and vector b
def parse_linear_equations(equations, n):
    var_names = ['x', 'y', 'z', 'w']
    A = []
    b = []
    
    for eq in equations:
        eq = eq.replace('^', '**')
        if '=' in eq:
            lhs, rhs = eq.split('=')
            expr = f"({lhs}) - ({rhs})"
        else:
            expr = eq
            
        # Evaluate at all zeros to find -b_i
        env_zero = {f"x{j+1}": 0.0 for j in range(n)}
        for j in range(min(n, 4)):
            env_zero[var_names[j]] = 0.0
        b_i = -eval(expr, {"__builtins__": None}, env_zero)
        b.append(b_i)
        
        # Evaluate at unit vectors to find coefficients a_ij
        row = []
        for j in range(n):
            env_unit = env_zero.copy()
            env_unit[f"x{j+1}"] = 1.0
            if j < 4:
                env_unit[var_names[j]] = 1.0
            a_ij = eval(expr, {"__builtins__": None}, env_unit) + b_i
            row.append(a_ij)
        A.append(row)
        
    return A, b

def main():
    print("=== Linear System Iterative Solver ===")
    print("Tip: Write equations using x, y, z OR x1, x2, x3 (e.g., 10*x + 2*y - z = 27)")
    
    n = int(input("\nEnter number of equations (n): "))
    equations = []
    for i in range(n):
        eq = input(f"Enter equation {i+1}: ")
        equations.append(eq)
        
    A, b = parse_linear_equations(equations, n)
    
    print("\nSelect a Method:")
    print("1. Jacobi Method")
    print("2. Gauss-Seidel Method")
    print("3. Relaxation Method")
    print("4. Successive Over-Relaxation (SOR) Method")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    omega = 1.0
    if choice == '4':
        omega = float(input("Enter relaxation factor w (e.g., 1.2): "))
        
    tol = float(input("Enter tolerance (e.g., 0.0001): "))
    max_iter = int(input("Enter maximum iterations (e.g., 50): "))
    
    x = [0.0] * n
    init_choice = input("Enter initial guesses manually? (y/n, default is 0): ").strip().lower()
    if init_choice == 'y':
        for i in range(n):
            x[i] = float(input(f"Enter initial guess x{i+1}^(0): "))

    # ---------------------------------------------------------
    # 1. JACOBI, 2. GAUSS-SEIDEL & 4. SOR METHODS
    # ---------------------------------------------------------
    if choice in ['1', '2', '4']:
        header = f"{'Iter':<6} | " + " | ".join([f"{'x'+str(i+1):<12}" for i in range(n)]) + f" | {'Max Error':<12}"
        print("\n" + header)
        print("-" * len(header))
        
        for k in range(1, max_iter + 1):
            x_old = x.copy()
            x_new = x.copy()
            
            for i in range(n):
                if choice == '1':  # Jacobi uses only old values
                    sigma = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
                    x_new[i] = (b[i] - sigma) / A[i][i]
                elif choice == '2':  # Gauss-Seidel uses updated values immediately
                    sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
                    x_new[i] = (b[i] - sigma) / A[i][i]
                elif choice == '4':  # SOR applies relaxation factor omega
                    sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
                    gs_val = (b[i] - sigma) / A[i][i]
                    x_new[i] = (1 - omega) * x_old[i] + omega * gs_val
                    
            error = max(abs(x_new[i] - x_old[i]) for i in range(n))
            x = x_new
            
            row_str = f"{k:<6} | " + " | ".join([f"{val:<12.6f}" for val in x]) + f" | {error:<12.6f}"
            print(row_str)
            
            if error < tol:
                break
                
        print("-" * len(header))
        print("Final Answer:")
        for i in range(n):
            print(f"x{i+1} ≈ {x[i]:.6f}")

    # ---------------------------------------------------------
    # 3. RELAXATION METHOD (Southwell Residual Relaxation)
    # ---------------------------------------------------------
    elif choice == '3':
        header = (f"{'Iter':<6} | " + 
                  " | ".join([f"{'x'+str(i+1):<10}" for i in range(n)]) + " | " +
                  " | ".join([f"{'R'+str(i+1):<10}" for i in range(n)]) + f" | {'Max |R|':<10}")
        print("\n" + header)
        print("-" * len(header))
        
        for k in range(1, max_iter + 1):
            # Calculate residuals R_i = b_i - sum(a_ij * x_j)
            R = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
            max_r = max(abs(r) for r in R)
            
            row_str = (f"{k:<6} | " + 
                       " | ".join([f"{val:<10.6f}" for val in x]) + " | " +
                       " | ".join([f"{r:<10.6f}" for r in R]) + f" | {max_r:<10.6f}")
            print(row_str)
            
            if max_r < tol:
                break
                
            # Find index of largest residual and relax that variable
            idx = max(range(n), key=lambda i: abs(R[i]))
            dx = R[idx] / A[idx][idx]
            x[idx] += dx
            
        print("-" * len(header))
        print("Final Answer:")
        for i in range(n):
            print(f"x{i+1} ≈ {x[i]:.6f}")

    else:
        print("Invalid choice! Please select a number between 1 and 4.")

if __name__ == "__main__":
    main()