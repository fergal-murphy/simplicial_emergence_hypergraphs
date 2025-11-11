import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import permutations

# ---------- helper functions (unchanged) ----------
def decompose_matrix(A):
    S = (A + A.T) / 2      # symmetric
    Q = (A - A.T) / 2      # antisymmetric
    return S, Q

def decompose_tensor(T):
    sym = np.zeros_like(T)
    alt = np.zeros_like(T)
    for perm in permutations(range(3)):
        permuted = np.transpose(T, perm)
        sym += permuted
        sign = 1 if perm in [(0,1,2), (1,2,0), (2,0,1)] else -1
        alt += sign * permuted
    sym /= 6
    alt /= 6
    mix = T - sym - alt
    return sym, alt, mix

def frob_norm(A):
    return np.linalg.norm(A)

# (tiny) Levi-Civita builder for N×N×N
def epsilon_tensor(N):
    eps = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if len({i, j, k}) < 3:           # repeated indices ⇒ 0
                    continue
                perm = [i, j, k]
                inv = sum(perm[a] > perm[b] for a in range(3) for b in range(a+1, 3))
                eps[i, j, k] = -1 if inv % 2 else 1
    return eps
# --------------------------------------------------


def derivatives_alt(Y, t, N, omega, delta1, delta2, eps):
    """ODEs for the antisymmetric example."""
    theta = Y[:N]
    A1 = Y[N:N+N*N].reshape(N, N)
    A2 = Y[N+N*N:].reshape(N, N, N)

    # ---------- phase dynamics ----------
    theta_dot = np.zeros(N)
    for i in range(N):
        # same interaction kernels you used before
        interaction1 = sum(A1[i, j] * np.sin(theta[j] - theta[i])
                           for j in range(N))
        interaction2 = sum(A2[i, j, k] * np.sin(theta[j] + theta[i] - 2*theta[k])
                           for j in range(N) for k in range(N))
        theta_dot[i] = omega[i] + interaction1 / N + interaction2 / N**2

    # ---------- edge & hyper-edge dynamics ----------
    N_driver = np.sin(theta[:, None] - theta[None, :])            # antisymmetric
    A1_dot = -delta1 * (A1 - N_driver)                            # dA = -δ (A - N)

    sum_theta = (theta[:, None, None] + theta[None, :, None] + theta[None, None, :])
    H_driver = eps * np.sin(sum_theta)                            # ε_ijk sin(…)
    A2_dot = -delta2 * (A2 - H_driver)

    return np.concatenate([theta_dot, A1_dot.flatten(), A2_dot.flatten()])


# ---------------- simulation parameters ----------------
N       = 5
delta1  = 0.1
delta2  = 0.1
t       = np.linspace(0, 50, 500)
omega   = np.linspace(-1.0, 1.0, N)        # intrinsic frequencies
eps     = epsilon_tensor(N)                # Levi-Civita tensor

# initial conditions
theta0  = np.random.rand(N)
A1_0    = np.random.rand(N, N)
A2_0    = np.random.rand(N, N, N)
Y0      = np.concatenate([theta0, A1_0.flatten(), A2_0.flatten()])

# integrate
solution = odeint(derivatives_alt, Y0, t, args=(N, omega, delta1, delta2, eps))

# ---------------- post-processing ----------------
e1, e2 = np.zeros_like(t), np.zeros_like(t)
f1, f2, f3 = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

for idx, Yi in enumerate(solution):
    A1 = Yi[N:N+N*N].reshape(N, N)
    A2 = Yi[N+N*N:].reshape(N, N, N)

    S, Q                 = decompose_matrix(A1)
    e1[idx], e2[idx]     = frob_norm(S), frob_norm(Q)

    sym3, alt3, mix3     = decompose_tensor(A2)
    f1[idx], f2[idx], f3[idx] = (frob_norm(sym3),
                                 frob_norm(alt3),
                                 frob_norm(mix3))

# ---------------- plots ----------------
plt.figure()
plt.plot(t, e1, label=r'‖$A_{1,\mathrm{sym}}$‖$_F$')
plt.plot(t, e2, label=r'‖$A_{1,\mathrm{alt}}$‖$_F$')
plt.xlabel('Time'); plt.ylabel('Frobenius norm'); plt.legend()
plt.title('Binary (antisymmetric)')

plt.figure()
plt.plot(t, f1, label=r'‖$A_{2,\mathrm{sym}}$‖$_F$')
plt.plot(t, f2, label=r'‖$A_{2,\mathrm{alt}}$‖$_F$')
plt.plot(t, f3, label=r'‖$A_{2,\mathrm{mix}}$‖$_F$')
plt.xlabel('Time'); plt.ylabel('Frobenius norm'); plt.legend()
plt.title('Triadic (antisymmetric)')

plt.show()
