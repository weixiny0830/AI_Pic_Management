
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def f_discrete(x, u, Ts):
    px, py, th = x
    v, om = u
    return np.array([
        px + Ts * v * np.cos(th),
        py + Ts * v * np.sin(th),
        th + Ts * om
    ])

def jacobians(x_bar, u_bar, Ts):
    px, py, th = x_bar
    v, om = u_bar
    A = np.eye(3)
    A[0,2] = -Ts * v * np.sin(th)
    A[1,2] =  Ts * v * np.cos(th)

    B = np.zeros((3,2))
    B[0,0] = Ts * np.cos(th)
    B[1,0] = Ts * np.sin(th)
    B[2,1] = Ts
    return A, B

# ---------- Path helpers ----------
def _polyline_cumlen(pts):
    d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])

def _project_onto_segment(p, a, b):
    d = b - a
    l2 = float(d @ d)
    if l2 == 0.0:
        return a, 0.0, float(((p - a)**2).sum())
    t = float((p - a) @ d) / l2
    t = max(0.0, min(1.0, t))
    q = a + t * d
    return q, t, float(((p - q)**2).sum())

def _project_onto_polyline(p, pts, cumlen):
    best = (None, None, float('inf'), 0)
    for i in range(len(pts)-1):
        q, t, dist2 = _project_onto_segment(p, pts[i], pts[i+1])
        if dist2 < best[2]:
            best = (q, t, dist2, i)
    q, t, _, i = best
    seg_len = float(np.linalg.norm(pts[i+1]-pts[i]))
    s0 = float(cumlen[i] + t * seg_len)
    return s0, i, t

def _sample_polyline_by_arclen(pts, cumlen, s):
    s = float(np.clip(s, 0.0, cumlen[-1]))
    i = np.searchsorted(cumlen, s, side='right') - 1
    i = int(np.clip(i, 0, len(pts)-2))
    seg_len = float(cumlen[i+1] - cumlen[i])
    if seg_len == 0.0:
        pos = pts[i].copy()
        d = pts[min(i+1, len(pts)-1)] - pts[i]
    else:
        t = (s - cumlen[i]) / seg_len
        pos = (1.0 - t) * pts[i] + t * pts[i+1]
        d = pts[i+1] - pts[i]
    th = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 0 else 0.0
    return pos, th

def _unwrap_anchored(th_list, theta_anchor):
    """Return an 'unwrapped' heading sequence, anchored near theta_anchor.
    At each step choose the 2π-shift of th_k that is closest to previous.
    """
    th = np.asarray(th_list, dtype=float)
    out = np.empty_like(th)
    acc = float(theta_anchor)
    for k in range(len(th)):
        raw = th[k]
        # choose integer n to minimize |(raw + 2πn) - acc|
        n = np.round((acc - raw) / (2*np.pi))
        out[k] = raw + 2*np.pi*n
        acc = out[k]
    return out

# ---------- Reference builder (arc-length + anchored unwrap + slow-down) ----------
def build_ref_traj_arc(x, waypoints, Ts, N, v_cruise=0.6, k_slow=0.7, v_min=0.05):
    """
    Build x_refs,u_refs along path with:
      1) arc-length sampling,
      2) heading sequence unwrapped w.r.t. current theta (avoid 2π drift),
      3) corner slow-down: v_ref = v_cruise / (1 + k_slow*|omega_ref|), clamped to [v_min, v_cruise].
    """
    pts = np.asarray(waypoints, dtype=float)
    cumlen = _polyline_cumlen(pts)

    s0, _, _ = _project_onto_polyline(x[:2], pts, cumlen)

    # first pass: positions and raw headings from path
    x_refs = np.zeros((N+1, 3))
    th_raw = []
    for k in range(N+1):
        s = s0 + v_cruise * Ts * k
        pos, th = _sample_polyline_by_arclen(pts, cumlen, s)
        x_refs[k,:2] = pos
        th_raw.append(th)

    # anchored unwrap to keep heading close to current theta
    th_unwrap = _unwrap_anchored(th_raw, x[2])
    x_refs[:,2] = th_unwrap

    # feedforward omega from unwrapped heading
    omega_ref = np.diff(th_unwrap) / Ts

    # slow-down near turns (bounded)
    v_ref = v_cruise / (1.0 + k_slow * np.abs(omega_ref))
    v_ref = np.clip(v_ref, v_min, v_cruise)

    u_refs = np.zeros((N,2))
    u_refs[:,0] = v_ref
    u_refs[:,1] = np.clip(omega_ref, -2.5, 2.5)
    return x_refs, u_refs

# ---------- MPC ----------
def tvlqr_mpc_step(x, x_refs, u_refs, Ts, Q, R, Qf, u_bounds, du_bounds=None):
    N = u_refs.shape[0]
    nx, nu = 3, 2

    A_seq, B_seq, c_seq = [], [], []
    for k in range(N):
        A, B = jacobians(x_refs[k], u_refs[k], Ts)
        c = f_discrete(x_refs[k], u_refs[k], Ts) - (A @ x_refs[k] + B @ u_refs[k])
        A_seq.append(A); B_seq.append(B); c_seq.append(c)

    X = cp.Variable((N+1, nx))
    U = cp.Variable((N, nu))

    constr = [X[0,:] == x]
    for k in range(N):
        constr += [X[k+1,:] == A_seq[k] @ X[k,:] + B_seq[k] @ U[k,:] + c_seq[k]]
        vmin, vmax = u_bounds[0]
        omin, omax = u_bounds[1]
        constr += [U[k,0] >= vmin, U[k,0] <= vmax]
        constr += [U[k,1] >= omin, U[k,1] <= omax]

    if du_bounds is not None:
        dumin, domin = du_bounds[0]
        dumax, domax = du_bounds[1]
        for k in range(N-1):
            constr += [U[k+1,0] - U[k,0] >= dumin, U[k+1,0] - U[k,0] <= dumax]
            constr += [U[k+1,1] - U[k,1] >= domin, U[k+1,1] - U[k,1] <= domax]

    # angle error wrap (approx): map X[k,2] into same 2π sheet as x_refs[k,2] by adding multiples of 2π
    # We can't wrap variable in CVXPY; instead keep theta weight smaller in Q to avoid fighting sheets.

    cost = 0
    for k in range(N):
        dx = X[k,:] - x_refs[k,:]
        du = U[k,:] - u_refs[k,:]
        cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)
    dxN = X[N,:] - x_refs[N,:]
    cost += cp.quad_form(dxN, Qf)

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if U.value is None:
        raise RuntimeError("QP infeasible; relax bounds or add slack.")

    return U.value[0,:].copy(), X.value.copy()

def simulate_mpc(start, waypoints, Ts=0.1, N=25, steps=500,
                 Q=np.diag([8.0,8.0,0.2]), R=np.diag([0.05,0.05]), Qf=None,
                 v_bounds=(0.0, 0.8), om_bounds=(-3.0, 3.0),
                 du_bounds=((-0.4,-0.6),(0.4,0.6)),
                 v_cruise=0.45, k_slow=1.0):
    if Qf is None:
        Qf = Q * 8.0
    x = np.array(start, dtype=float)
    traj = [x.copy()]
    u_applied = []

    for t in range(steps):
        x_refs, u_refs = build_ref_traj_arc(x, waypoints, Ts, N, v_cruise=v_cruise, k_slow=k_slow)
        u0, _ = tvlqr_mpc_step(
            x, x_refs[:N+1], u_refs[:N],
            Ts, Q, R, Qf,
            u_bounds=(v_bounds, om_bounds),
            du_bounds=du_bounds
        )
        x = f_discrete(x, u0, Ts)
        traj.append(x.copy())
        u_applied.append(u0.copy())

        if np.linalg.norm(x[:2] - waypoints[-1]) < 0.03:
            break

    return np.array(traj), np.array(u_applied)

def wheel_speeds(v, omega, wheel_radius=0.05, wheel_base=0.30):
    wL = (v - omega * wheel_base * 0.5) / wheel_radius
    wR = (v + omega * wheel_base * 0.5) / wheel_radius
    return wL, wR

if __name__ == "__main__":
    waypoints = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ], dtype=float)
    start = [0.0, -0.2, np.deg2rad(90)]
    traj, u = simulate_mpc(start, waypoints, Ts=0.1, N=25, steps=600)

    plt.figure()
    plt.plot(waypoints[:,0], waypoints[:,1], 'o--', label="waypoints")
    plt.plot(traj[:,0], traj[:,1], label="MPC traj")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Differential-drive MPC (anchored unwrap + slow-down)")

    plt.figure()
    plt.plot(u[:,0], label="v [m/s]")
    plt.plot(u[:,1], label="omega [rad/s]")
    plt.grid(True); plt.legend(); plt.title("Applied controls")
    plt.xlabel("step")
    plt.show()
