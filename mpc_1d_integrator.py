"""
MPC Tutorial — 1D Integrator (from zero to first MPC)
=====================================================
Requirements:
    python 3.10+
    pip install numpy cvxpy matplotlib

This script teaches MPC on the simple discrete-time system:
    x_{k+1} = x_k + u_k
We track a target position x_ref with control limit |u| <= umax.

What you'll learn:
- Build the cost function J = sum (x_k - x_ref)^2 * q + sum u_k^2 * r + terminal cost
- Assemble the (A,B) prediction model over a finite horizon N
- Solve a QP with cvxpy and apply only the first control (receding horizon)
- Plot the closed-loop behavior

Run:
    python mpc_1d_integrator.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def mpc_1d_integrator(x0=0.0, x_ref=25.0, N=20, q=1.0, r=0.1, qf=5.0, umax=0.2, T=200):
    """
    Args:
        x0: initial position
        x_ref: target (constant reference)
        N: horizon steps
        q, r, qf: weights (stage state, stage input, terminal state)
        umax: |u| limit
        T: number of simulation steps
    """
    x_hist = [x0]
    u_hist = []
    x = x0

    for t in range(T):
        # Decision variables over horizon
        x_var = cp.Variable((N+1, 1))
        u_var = cp.Variable((N, 1))

        # Build constraints: dynamics and bounds
        constr = []
        constr += [x_var[0] == x]

        A = 1.0
        B = 1.0
        for k in range(N):
            constr += [x_var[k+1] == A * x_var[k] + B * u_var[k]]
            constr += [cp.abs(u_var[k]) <= umax]

        # Cost
        cost = 0
        for k in range(N):
            cost += q * cp.square(x_var[k] - x_ref) + r * cp.square(u_var[k])
        cost += qf * cp.square(x_var[N] - x_ref)

        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.OSQP, verbose=False)

        if u_var.value is None:
            raise RuntimeError("QP infeasible — try increasing umax or reducing weights.")

        u0 = float(u_var.value[0])
        # Apply control
        x = A * x + B * u0

        x_hist.append(x)
        u_hist.append(u0)

    return np.array(x_hist), np.array(u_hist)

if __name__ == "__main__":
    x_hist, u_hist = mpc_1d_integrator()
    t = np.arange(len(x_hist))
    plt.figure()
    plt.plot(t, x_hist, label="x")
    plt.axhline(5.0, linestyle="--", label="x_ref")
    plt.xlabel("step")
    plt.ylabel("position")
    plt.legend()
    plt.title("1D MPC: position")
    plt.grid(True)

    plt.figure()
    plt.plot(u_hist, label="u")
    plt.xlabel("step")
    plt.ylabel("control")
    plt.legend()
    plt.title("1D MPC: control")
    plt.grid(True)
    plt.show()