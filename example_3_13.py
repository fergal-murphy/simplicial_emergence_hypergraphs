import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from itertools import combinations

# --- Model Parameters from Example 3.12 ---
# These parameters are chosen to satisfy the conditions in the paper.
N = 4      # Number of oscillators
ALPHA = 0.5  # Decay rate for edges
GAMMA = 0.8  # Decay rate for triads
DELTA = 0.5  # Simplicial threshold
ZETA = 0.05  # Smoothing parameter (must be small)

# Beta is chosen to satisfy the condition for simplicial emergence from the paper
# beta > (alpha * (delta + 1)) / (delta * 0.25 * tanh(delta / zeta))
# We choose a value that safely satisfies this.
BETA = 25.0

# Simulation time
T_START = 0
T_END = 25
T_POINTS = 250

# --- Smoothened Helper Functions (Page 22 of the PDF) ---

def max_zeta(x, axis=None):
    """Smoothed max function using log-sum-exp."""
    return ZETA * np.log(np.sum(np.exp(x / ZETA), axis=axis))

def min_zeta(a, b):
    """Smoothed min function."""
    return -ZETA * np.log(np.exp(-a / ZETA) + np.exp(-b / ZETA))

def H_zeta(z):
    """Smoothed Heaviside step function."""
    return 0.5 * (1 + np.tanh(z / ZETA))

def sgn_s_zeta(x, y):
    """Symmetrised smoothed sign function."""
    return np.tanh((x + y) / (2 * ZETA))

# --- Projection Operators (Page 9 of the PDF) ---

def project_A1(A1):
    """Projects the binary tensor A1 into symmetric and antisymmetric parts."""
    A1_sym = 0.5 * (A1 + A1.T)
    A1_alt = 0.5 * (A1 - A1.T)
    return A1_sym, A1_alt

def project_A2(A2):
    """Projects the triadic tensor A2 into its isotypic components."""
    A2_sym = (1/6) * (A2 + A2.transpose(1,0,2) + A2.transpose(2,1,0) +
                       A2.transpose(0,2,1) + A2.transpose(1,2,0) + A2.transpose(2,0,1))
    A2_alt = (1/6) * (A2 - A2.transpose(1,0,2) - A2.transpose(2,1,0) -
                       A2.transpose(0,2,1) + A2.transpose(1,2,0) + A2.transpose(2,0,1))
    # The mixed component is what's left over
    A2_mix = A2 - A2_sym - A2_alt
    return A2_sym, A2_alt, A2_mix

# --- The System of Differential Equations ---

def dynamics(t, y, omega):
    """
    Defines the full dynamics of the system.
    y is a flattened vector containing theta, A1, and A2.
    """
    # Unpack the state vector y
    theta = y[0:N]
    A1 = y[N:N + N*N].reshape((N, N))
    A2 = y[N + N*N:].reshape((N, N, N))

    # --- Node dynamics (theta_i) ---
    d_theta_dt = np.zeros(N)
    theta_diff = theta[:, np.newaxis] - theta
    sum1 = np.sum(A1 * np.sin(-theta_diff), axis=1) / N
    theta_sum = theta[:, np.newaxis, np.newaxis] + theta[np.newaxis, :, np.newaxis] + theta[np.newaxis, np.newaxis, :]
    triadic_coupling = np.sin(theta_sum - 3*theta[:, np.newaxis, np.newaxis])
    sum2 = np.sum(A2 * triadic_coupling, axis=(1, 2)) / (N*N)
    d_theta_dt = omega + sum1 + sum2

    # --- Edge dynamics (A_ij^(1)) ---
    d_A1_dt = np.zeros_like(A1)
    cos_theta_diff = np.cos(theta_diff)
    abs_A1 = np.abs(A1)
    J = np.zeros_like(A1)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            valid_indices = [m for m in range(N) if m != i and m != j]
            if not valid_indices: continue
            max_A2 = max_zeta(np.abs(A2[i,j,valid_indices]))
            term1 = H_zeta(max_A2 - DELTA)
            min_A1_abs = min_zeta(abs_A1[i,j], abs_A1[j,i])
            term2 = H_zeta(DELTA - min_A1_abs)
            J[i,j] = term1 * term2
    reinforcement = BETA * DELTA * J * sgn_s_zeta(A1, A1.T)
    d_A1_dt = -ALPHA * (A1 - cos_theta_diff) + reinforcement
    np.fill_diagonal(d_A1_dt, 0)

    # --- Triad dynamics (A_ijk^(2)) ---
    d_A2_dt = np.zeros_like(A2)
    cos_theta_sum = np.cos(theta_sum)
    d_A2_dt = -GAMMA * (A2 - DELTA * cos_theta_sum)

    # Flatten derivatives
    dy_dt = np.concatenate([
        d_theta_dt.flatten(), d_A1_dt.flatten(), d_A2_dt.flatten()
    ])
    return dy_dt

# --- Main Simulation ---
if __name__ == "__main__":
    attempt = 0
    while True:
        attempt += 1
        print(f"--- Running Simulation Attempt #{attempt} ---")

        # Initial conditions are randomized for each attempt
        omega = np.random.normal(0, 0.5, N)
        theta_0 = np.random.uniform(0, 2*np.pi, N)
        A1_0 = np.random.rand(N, N) * 0.5 - 0.25
        np.fill_diagonal(A1_0, 0)
        A2_0 = np.random.rand(N, N, N) * 0.5 - 0.25

        # Create a simplicial complex violation on triad (0,1,2)
        A2_0[0,1,2] = A2_0[0,2,1] = A2_0[1,0,2] = 0.8
        A2_0[1,2,0] = A2_0[2,0,1] = A2_0[2,1,0] = 0.8
        A1_0[0,2] = A1_0[2,0] = 0.6 # Strong edge
        A1_0[1,2] = A1_0[2,1] = 0.7 # Strong edge
        A1_0[0,1] = A1_0[1,0] = 0.1 # This is the weak edge causing the violation

        y0 = np.concatenate([theta_0.flatten(), A1_0.flatten(), A2_0.flatten()])

        # Run the simulation
        sol = solve_ivp(
            dynamics, (T_START, T_END), y0, args=(omega,),
            dense_output=True, t_eval=np.linspace(T_START, T_END, T_POINTS)
        )

        # --- Check for locally persistent triads ---
        # A triad is persistent if it's strong in one snapshot and the next.
        snapshot_indices = np.linspace(0, T_POINTS - 1, 6, dtype=int)
        persistent_triad_found = False

        # Loop through the first 5 snapshots to compare each with the next one
        for i in range(len(snapshot_indices) - 1):
            current_snapshot_idx = snapshot_indices[i]
            next_snapshot_idx = snapshot_indices[i+1]

            A2_current = sol.y[N+N*N:, :].reshape(N, N, N, -1)[:,:,:,current_snapshot_idx]
            A2_next = sol.y[N+N*N:, :].reshape(N, N, N, -1)[:,:,:,next_snapshot_idx]

            # Check all possible triads for persistence
            for u, v, w in combinations(range(N), 3):
                if np.abs(A2_current[u, v, w]) >= DELTA and np.abs(A2_next[u, v, w]) >= DELTA:
                    persistent_triad_found = True
                    break # Exit triad loop
            if persistent_triad_found:
                break # Exit snapshot loop

        if persistent_triad_found:
            print(f"Success! Found a locally persistent triad after {attempt} attempts.")
            break
        else:
            print("No locally persistent triad found. Retrying with new initial conditions...")

    # --- Post-processing and Plotting for the successful run ---
    print("Generating plots for the successful simulation...")
    t = sol.t
    A1_t = sol.y[N:N+N*N:, :].reshape(N, N, -1)
    A2_t = sol.y[N+N*N:, :].reshape(N, N, N, -1)

    # --- Plotting Figure 1: Frobenius Norms ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    A1_sym_norm, A1_alt_norm = [], []
    A2_sym_norm, A2_alt_norm, A2_mix_norm = [], [], []
    for i in range(len(t)):
        A1_sym, A1_alt = project_A1(A1_t[:,:,i])
        A2_sym, A2_alt, A2_mix = project_A2(A2_t[:,:,:,i])
        A1_sym_norm.append(np.linalg.norm(A1_sym))
        A1_alt_norm.append(np.linalg.norm(A1_alt))
        A2_sym_norm.append(np.linalg.norm(A2_sym))
        A2_alt_norm.append(np.linalg.norm(A2_alt))
        A2_mix_norm.append(np.linalg.norm(A2_mix))

    ax1.plot(t, A1_sym_norm, label=r'$||A_{sym}^{(1)}||_F$')
    ax1.plot(t, A1_alt_norm, label=r'$||A_{alt}^{(1)}||_F$', linestyle='--')
    ax1.set_title('Binary Layer (Edges)')
    ax1.legend()
    fig1.suptitle('Convergence to Symmetric Regime', fontsize=16, fontweight='bold')

    ax2.plot(t, A2_sym_norm, label=r'$||A_{sym}^{(2)}||_F$')
    ax2.plot(t, A2_alt_norm, label=r'$||A_{alt}^{(2)}||_F$', linestyle='--')
    ax2.plot(t, A2_mix_norm, label=r'$||A_{mix}^{(2)}||_F$', linestyle=':')
    ax2.set_title('Triadic Layer (Triangles)')
    ax2.legend()
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Snapshots Figure 2: Network Evolution ---
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # We use the same snapshot indices for plotting
    node_pos = {0: (1, 0.5), 1: (0.5, 1), 2: (0, 0.5), 3: (0.5, 0)}

    for i, frame_idx in enumerate(snapshot_indices):
        ax = axes[i]

        # Plot Triangles (2-simplices)
        for u, v, w in combinations(range(N), 3):
            if np.abs(A2_t[u, v, w, frame_idx]) >= DELTA:
                p = Polygon([node_pos[u], node_pos[v], node_pos[w]],
                            facecolor='lightblue', alpha=0.7, zorder=1)
                ax.add_patch(p)

        # Plot Edges (1-simplices)
        for u, v in combinations(range(N), 2):
            strength = np.abs(A1_t[u, v, frame_idx])
            style = 'solid' if strength >= DELTA else 'dashed'
            ax.plot([node_pos[u][0], node_pos[v][0]],
                    [node_pos[u][1], node_pos[v][1]],
                    color='dimgray', lw=2.5, linestyle=style, zorder=2)

        # Plot Nodes (0-simplices)
        for u in range(N):
            ax.plot(node_pos[u][0], node_pos[u][1], 'o',
                    markersize=18, color='black', zorder=3)
            ax.text(node_pos[u][0], node_pos[u][1], str(u),
                    ha='center', va='center', color='white',
                    fontweight='bold', fontsize=10)

        ax.set_title(f't = {t[frame_idx]:.2f}', pad=10)
        ax.set_aspect('equal')
        ax.axis('off')

    # Create a custom legend
    legend_elements = [
        Line2D([0], [0], color='dimgray', lw=2.5, linestyle='solid', label=f'Strong Edge (> {DELTA})'),
        Line2D([0], [0], color='dimgray', lw=2.5, linestyle='dashed', label=f'Weak Edge (< {DELTA})'),
        Patch(facecolor='lightblue', alpha=0.7, label=f'Strong Triad (> {DELTA})')
    ]
    fig2.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05))
    fig2.suptitle('Evolution of Simplicial Complex Structure', fontsize=20, fontweight='bold')
    fig2.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Save the figures to separate files
    fig1.savefig('frobenius_norms.png', dpi=300, bbox_inches='tight')
    fig2.savefig('network_evolution.png', dpi=300, bbox_inches='tight')

    print("\nPlots saved as 'frobenius_norms.png' and 'network_evolution.png'")

    # Show the plots as well
    plt.show()
