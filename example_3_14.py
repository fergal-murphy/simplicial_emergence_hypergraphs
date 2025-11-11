import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from itertools import combinations

# --- Model Parameters for Consensus-Based Dynamics ---
N = 4
ALPHA = 0.5  # Decay rate for edges
GAMMA = 0.8  # Decay rate for triads
DELTA = 0.5  # Simplicial threshold
ZETA = 0.05  # Smoothing parameter

# Parameters for the new consensus model
KAPPA_1 = 1.0  # Target strength for edges
KAPPA_2 = 1.2  # Target strength for triads (should be > DELTA)
LAMBDA_1 = 2.0  # Sensitivity for edge strength
LAMBDA_2 = 5.0  # Sensitivity for triad strength (high sensitivity is key)
BETA = 25.0    # Simplicial closure enforcement strength

# Simulation time
T_START = 0
T_END = 25
T_POINTS = 250

# --- Smoothened Helper Functions (Unchanged from original code) [cite: 588-603] ---
def max_zeta(x, axis=None):
    return ZETA * np.log(np.sum(np.exp(x / ZETA), axis=axis))

def min_zeta(a, b):
    return -ZETA * np.log(np.exp(-a / ZETA) + np.exp(-b / ZETA))

def H_zeta(z):
    return 0.5 * (1 + np.tanh(z / ZETA))

def sgn_s_zeta(x, y):
    return np.tanh((x + y) / (2 * ZETA))

# --- Projection Operators (Unchanged from original code) [cite: 193, 194, 208] ---
def project_A1(A1):
    A1_sym = 0.5 * (A1 + A1.T)
    A1_alt = 0.5 * (A1 - A1.T)
    return A1_sym, A1_alt

def project_A2(A2):
    A2_sym = (1/6) * (A2 + A2.transpose(1,0,2) + A2.transpose(2,1,0) +
                      A2.transpose(0,2,1) + A2.transpose(1,2,0) + A2.transpose(2,0,1))
    A2_alt = (1/6) * (A2 - A2.transpose(1,0,2) - A2.transpose(2,1,0) -
                      A2.transpose(0,2,1) + A2.transpose(1,2,0) + A2.transpose(2,0,1))
    A2_mix = A2 - A2_sym - A2_alt
    return A2_sym, A2_alt, A2_mix

# --- The NEW System of Differential Equations (Consensus Model) ---
def dynamics(t, y):
    """
    Defines the consensus-based dynamics.
    y is a flattened vector containing x, A1, and A2.
    """
    # Unpack the state vector y
    x = y[0:N]
    A1 = y[N:N + N*N].reshape((N, N))
    A2 = y[N + N*N:].reshape((N, N, N))

    # --- Node dynamics (x_i): Consensus dynamics ---
    x_diff = x[:, np.newaxis] - x
    sum1 = np.sum(A1 * (-x_diff), axis=1) # Sum over j of A_ij * (x_j - x_i)

    # Vectorized triadic consensus term
    triadic_target = 0.5 * (x[np.newaxis, :, np.newaxis] + x[np.newaxis, np.newaxis, :]) - x[:, np.newaxis, np.newaxis]
    sum2 = np.sum(A2 * triadic_target, axis=(1, 2))
    d_x_dt = sum1 + sum2

    # --- Edge dynamics (A_ij^(1)): Hebbian rule + Simplicial Closure ---
    # Hebbian part: strength depends on node proximity
    hebbian_target = KAPPA_1 * np.exp(-LAMBDA_1 * x_diff**2)
    # Simplicial closure part (unchanged logic)
    abs_A1 = np.abs(A1)
    J = np.zeros_like(A1)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            # Find all other nodes k to check for parent triads A_ijk
            valid_indices = [m for m in range(N) if m != i and m != j]
            if not valid_indices: continue

            max_A2 = max_zeta(np.abs(A2[i, j, valid_indices]))
            term1 = H_zeta(max_A2 - DELTA)
            min_A1_abs = min_zeta(abs_A1[i, j], abs_A1[j, i])
            term2 = H_zeta(DELTA - min_A1_abs)
            J[i, j] = term1 * term2

    reinforcement = BETA * DELTA * J * sgn_s_zeta(A1, A1.T)
    d_A1_dt = -ALPHA * (A1 - hebbian_target) + reinforcement
    np.fill_diagonal(d_A1_dt, 0)

    # --- Triad dynamics (A_ijk^(2)): Persistence via local consensus ---
    # Calculate local variance V_ijk for all triads
    V = (1/3) * (
        x_diff[:, :, np.newaxis]**2 +
        x_diff[:, np.newaxis, :]**2 +
        x_diff[np.newaxis, :, :]**2
    )
    # Target strength T_ijk depends on low variance
    persistence_target = KAPPA_2 * np.exp(-LAMBDA_2 * V)
    d_A2_dt = -GAMMA * (A2 - persistence_target)

    # Flatten derivatives for the solver
    dy_dt = np.concatenate([d_x_dt.flatten(), d_A1_dt.flatten(), d_A2_dt.flatten()])
    return dy_dt

# --- Main Simulation ---
if __name__ == "__main__":
    print("--- Running Consensus-Based Simulation for Persistent Triads ---")

    # --- Initial Conditions ---
    # Nodes for triad (0,1,2) start in consensus, node 3 is an outlier.
    # This will encourage the triad (0,1,2) to persist.
    x_0 = np.array([0.1, 0.15, 0.2, 2.0])

    # Initial weights
    A1_0 = np.random.rand(N, N) * 0.5 - 0.25
    np.fill_diagonal(A1_0, 0)
    A2_0 = np.random.rand(N, N, N) * 0.5 - 0.25

    # Create the same initial simplicial violation on triad (0,1,2)
    # This triad is strong...
    A2_0[0,1,2] = A2_0[0,2,1] = A2_0[1,0,2] = 0.8
    A2_0[1,2,0] = A2_0[2,0,1] = A2_0[2,1,0] = 0.8
    # ...it has two strong edges...
    A1_0[0,2] = A1_0[2,0] = 0.6
    A1_0[1,2] = A1_0[2,1] = 0.7
    # ...but one WEAK edge, creating the violation.
    A1_0[0,1] = A1_0[1,0] = 0.1

    y0 = np.concatenate([x_0.flatten(), A1_0.flatten(), A2_0.flatten()])

    # Run the simulation
    sol = solve_ivp(
        dynamics, (T_START, T_END), y0,
        dense_output=True, t_eval=np.linspace(T_START, T_END, T_POINTS)
    )

    # --- Post-processing and Plotting ---
    print("Generating plots...")
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
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frobenius Norm')
    ax1.legend()
    fig1.suptitle('Convergence to Symmetric Regime with Persistence', fontsize=16, fontweight='bold')

    ax2.plot(t, A2_sym_norm, label=r'$||A_{sym}^{(2)}||_F$')
    ax2.plot(t, A2_alt_norm, label=r'$||A_{alt}^{(2)}||_F$', linestyle='--')
    ax2.plot(t, A2_mix_norm, label=r'$||A_{mix}^{(2)}||_F$', linestyle=':')
    ax2.set_title('Triadic Layer (Triangles)')
    ax2.set_xlabel('Time')
    ax2.legend()
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Plotting Figure 2: Network Evolution Snapshots ---
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    snapshot_indices = np.linspace(0, T_POINTS - 1, 6, dtype=int)
    node_pos = {0: (1, 0.5), 1: (0.5, 1), 2: (0, 0.5), 3: (0.5, 0)}

    for i, frame_idx in enumerate(snapshot_indices):
        ax = axes[i]
        A1_snap = np.mean(A1_t[:,:,frame_idx-2:frame_idx+2], axis=2) if frame_idx > 1 else A1_t[:,:,frame_idx]
        A2_snap = np.mean(A2_t[:,:,:,frame_idx-2:frame_idx+2], axis=3) if frame_idx > 1 else A2_t[:,:,:,frame_idx]

        for u, v, w in combinations(range(N), 3):
            if np.abs(A2_snap[u, v, w]) >= DELTA:
                ax.add_patch(Polygon([node_pos[u], node_pos[v], node_pos[w]], facecolor='lightblue', alpha=0.7, zorder=1))

        for u, v in combinations(range(N), 2):
            strength = np.abs(A1_snap[u, v])
            style = 'solid' if strength >= DELTA else 'dashed'
            ax.plot([node_pos[u][0], node_pos[v][0]], [node_pos[u][1], node_pos[v][1]], color='dimgray', lw=2.5, linestyle=style, zorder=2)

        for u in range(N):
            ax.plot(node_pos[u][0], node_pos[u][1], 'o', markersize=18, color='black', zorder=3)
            ax.text(node_pos[u][0], node_pos[u][1], str(u), ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        ax.set_title(f't = {t[frame_idx]:.2f}', pad=10)
        ax.set_aspect('equal')
        ax.axis('off')

    legend_elements = [
        Line2D([0], [0], color='dimgray', lw=2.5, linestyle='solid', label=f'Strong Edge (>{DELTA})'),
        Line2D([0], [0], color='dimgray', lw=2.5, linestyle='dashed', label=f'Weak Edge (<{DELTA})'),
        Patch(facecolor='lightblue', alpha=0.7, label=f'Strong Triad (>{DELTA})')
    ]
    fig2.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05))
    fig2.suptitle('Evolution of a Persistent Simplicial Complex', fontsize=20, fontweight='bold')
    fig2.tight_layout(rect=[0, 0.1, 1, 0.95])

    fig1.savefig('consensus_frobenius_norms.png', dpi=300, bbox_inches='tight')
    fig2.savefig('consensus_network_evolution.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved as 'consensus_frobenius_norms.png' and 'consensus_network_evolution.png'")
    plt.show()
