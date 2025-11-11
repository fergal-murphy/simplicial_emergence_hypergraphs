import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import permutations


# ---------- helper functions (unchanged) ----------
def decompose_matrix(A):
    S = (A + A.T) / 2        # symmetric part
    Q = (A - A.T) / 2        # antisymmetric part
    return S, Q

def decompose_tensor(T):
    sym  = np.zeros_like(T)
    alt  = np.zeros_like(T)
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
# --------------------------------------------------


def derivatives(Y, t, N, omega, delta1, delta2):
    theta = Y[:N]
    A1 = Y[N:N+N*N].reshape(N, N)
    A2 = Y[N+N*N:].reshape(N, N, N)

    # phase dynamics ----------------------------------------------------------
    theta_dot = np.zeros(N)
    for i in range(N):
        interaction1 = sum(A1[i, j] * np.sin(theta[j] - theta[i]) for j in range(N))
        interaction2 = sum(A2[i, j, k] * np.sin(theta[j] + theta[i] - 2*theta[k])
                           for j in range(N) for k in range(N))
        theta_dot[i] = omega[i] + (1/N) * interaction1 + (1/N**2) * interaction2

    # edge and hyper-edge dynamics -------------------------------------------
    A1_dot = -delta1 * (A1 + np.cos(theta[:, None] - theta[None, :]))   # ← cos(θ_i − θ_j)
    A2_dot = -delta2 * (A2 + np.cos(theta[:, None, None]
                                    + theta[None, :, None]
                                    + theta[None, None, :]))

    return np.concatenate([theta_dot, A1_dot.flatten(), A2_dot.flatten()])


# -------------------- simulation parameters --------------------
N = 5
delta1 = 0.1
delta2 = 0.1
t = np.linspace(0, 50, 500)

# 🔄  deterministic intrinsic frequencies (e.g. equally spaced)
omega = np.linspace(-1.0, 1.0, N)        # or choose all zeros for a purely driven system
# ----------------------------------------------------------------

# initial conditions (keep them random if you like)
theta0 = np.random.rand(N)
A1_0   = np.random.rand(N, N)
A2_0   = np.random.rand(N, N, N)
Y0 = np.concatenate([theta0, A1_0.flatten(), A2_0.flatten()])

# integrate
solution = odeint(derivatives, Y0, t, args=(N, omega, delta1, delta2))

# -------------------- post-processing --------------------
e1 = np.zeros_like(t)
e2 = np.zeros_like(t)
f1 = np.zeros_like(t)
f2 = np.zeros_like(t)
f3 = np.zeros_like(t)

for idx, Yi in enumerate(solution):
    A1 = Yi[N:N+N*N].reshape(N, N)
    A2 = Yi[N+N*N:].reshape(N, N, N)

    S, Q = decompose_matrix(A1)
    e1[idx] = frob_norm(S)
    e2[idx] = frob_norm(Q)

    sym3, alt3, mix3 = decompose_tensor(A2)
    f1[idx] = frob_norm(sym3)
    f2[idx] = frob_norm(alt3)
    f3[idx] = frob_norm(mix3)

# -------------------- plots --------------------
plt.figure()
plt.plot(t, e1, label='Symmetric part ‖A₁_sym‖_F')
plt.plot(t, e2, label='Antisymmetric part ‖A₁_alt‖_F')
plt.xlabel('Time'); plt.ylabel('Frobenius norm'); plt.legend()
plt.title('Binary matrix Norms')

plt.figure()
plt.plot(t, f1, label='Symmetric 3-tensor ‖A₂_sym‖_F')
plt.plot(t, f2, label='Antisymmetric 3-tensor ‖A₂_alt‖_F')
plt.plot(t, f3, label='Mixed 3-tensor ‖A₂_mix‖_F')
plt.xlabel('Time'); plt.ylabel('Frobenius norm'); plt.legend()
plt.title('Triadic tensor Norms')

plt.show()
