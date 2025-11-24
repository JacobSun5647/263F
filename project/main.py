import numpy as np
import matplotlib.pyplot as plt

from crossMat import crossMat
from getFb import getFb
from getFs import getFs
from gradEb import gradEb
from gradEs import gradEs
from hessEs import hessEs
from hessEb import hessEb
from objfun import objfun


# ============================================================
# Utility functions: geometry, torsion, external force
# ============================================================

def build_geometry(nv, RodLength, theta0_rad):
    """
    Build initial straight beam geometry of length RodLength,
    inclined at angle theta0 (w.r.t. x-axis).
    """
    deltaL = RodLength / (nv - 1)
    nodes = np.zeros((nv, 2))
    for c in range(nv):
        s = c * deltaL
        nodes[c, 0] = s * np.cos(theta0_rad)
        nodes[c, 1] = s * np.sin(theta0_rad)
    return nodes, deltaL


def build_masses_and_gravity(nv, R, rho, g_vec):
    """
    Build lumped mass vector, mass matrix, and constant gravity load.
    """
    ndof = 2 * nv
    m = np.zeros(ndof)
    W_gravity = np.zeros(ndof)

    for k in range(nv):
        mass_k = 4.0 / 3.0 * np.pi * R[k]**3 * rho
        m[2*k]     = mass_k
        m[2*k + 1] = mass_k
        W_gravity[2*k]     = mass_k * g_vec[0]
        W_gravity[2*k + 1] = mass_k * g_vec[1]

    mMat = np.diag(m)
    return m, mMat, W_gravity


def get_base_angle_and_rate(q, u, i0=0, i1=1):
    """
    Compute the angle and angular rate of the segment (i0 -> i1),
    along with its length and a unit normal vector.
    """
    x0, y0 = q[2*i0],   q[2*i0 + 1]
    x1, y1 = q[2*i1],   q[2*i1 + 1]
    vx0, vy0 = u[2*i0], u[2*i0 + 1]
    vx1, vy1 = u[2*i1], u[2*i1 + 1]

    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L < 1e-12:
        return 0.0, 0.0, 1.0, (0.0, 1.0)

    ex, ey = dx / L, dy / L
    nx, ny = -ey, ex
    phi = np.arctan2(dy, dx)

    dvx, dvy = vx1 - vx0, vy1 - vy0
    phi_dot = (dvx * nx + dvy * ny) / L

    return phi, phi_dot, L, (nx, ny)


def torsion_forces_on_base(q, u, phi0, k_theta, c_theta, i0=0, i1=1):
    """
    Equivalent nodal forces for a torsional spring + damper
    between nodes i0 and i1:

        M = -k_theta * (phi - phi0) - c_theta * phi_dot
    """
    phi, phi_dot, L, (nx, ny) = get_base_angle_and_rate(q, u, i0, i1)
    M = -k_theta * (phi - phi0) - c_theta * phi_dot

    Fmag = M / L
    F = np.zeros_like(q)

    # Node i0: +F * n
    F[2*i0]     += Fmag * nx
    F[2*i0 + 1] += Fmag * ny
    # Node i1: -F * n
    F[2*i1]     -= Fmag * nx
    F[2*i1 + 1] -= Fmag * ny

    return F


def external_tip_force(t, ndof, tip_node, P0, omega):
    """
    Sinusoidal force at the tip in x-direction:
        F_x(t) = P0 * sin(omega * t)
    """
    F = np.zeros(ndof)
    Fx = P0 * np.sin(omega * t)
    F[2 * tip_node] += Fx
    return F, Fx


# ============================================================
# Single-run simulation function
# ============================================================

def run_simulation(control_mode,
                   dt=0.05,
                   totalTime=20.0,
                   nv=51,
                   RodLength=20.0,
                   initial_angle_deg=60.0,
                   pre_time=10.0,
                   excitation_duration=4.0):   # shorter forcing window
    """
    Run a single simulation case.

    control_mode:
        "open"  -> fixed base damping
        "semi"  -> semi-active damping: c_theta uses sign(phi * phi_dot)
    """
    ndof = 2 * nv
    Nsteps = int(totalTime / dt) + 1

    # ------ geometry ------
    theta0 = np.deg2rad(initial_angle_deg)
    nodes, deltaL = build_geometry(nv, RodLength, theta0)

    # ------ radii & material ------
    R = np.zeros(nv)
    for k in range(nv):
        R[k] = deltaL / 10.0
    midNode = nv // 2
    R[midNode] = 0.025

    rho_metal = 2700.0
    rho = rho_metal
    ro = 0.10
    ri = 0.09
    Y  = 70e6        # Pa (softened a bit to see visible motion)

    EI = Y * np.pi * (ro**4 - ri**4) / 4.0
    EA = Y * np.pi * (ro**2 - ri**2)

    maximum_iter = 1000
    tol = 1e-3

    # ------ masses & gravity ------
    g_vec = np.array([0.0, -9.8])
    m, mMat, W_gravity = build_masses_and_gravity(nv, R, rho, g_vec)

    # ------ structural damping matrix (Rayleigh mass damping) ------
    # small global damping so base damper matters
    alpha_M = 0.002
    C = alpha_M * mMat

    # ------ initial state ------
    q0 = np.zeros(ndof)
    for c in range(nv):
        q0[2*c]     = nodes[c, 0]
        q0[2*c + 1] = nodes[c, 1]
    u0 = np.zeros(ndof)

    all_DOFs = np.arange(ndof)
    # clamp first two nodes (0,1) and (1,2) -> 0,1,2,3 DOFs
    fixed_index = np.array([0, 1, 2, 3], dtype=int)
    free_index  = np.setdiff1d(all_DOFs, fixed_index)

    # ------ base torsional spring & damping ------
    # reference angle from node 1 -> node 2
    phi0 = np.arctan2(nodes[2, 1] - nodes[1, 1],
                      nodes[2, 0] - nodes[1, 0])

    k_theta      = 3e3    # [N·m/rad] a bit softer
    c_theta_base = 2e2    # small baseline damping
    c_theta_min  = 0.0    # allow almost free rotation
    c_theta_max  = 1.2e4  # strong damping when "on"

    # ------ tip force parameters ------
    tip_node = nv - 1
    P0 = 500.0
    T_load = 8.0
    omega  = 2.0 * np.pi / T_load

    # ============================================================
    # 1) Pre-settle: gravity only, fixed damping
    # ============================================================
    N_pre = int(pre_time / dt)
    ctime = 0.0

    for _ in range(N_pre):
        F_tip = np.zeros(ndof)
        c_theta = c_theta_base
        F_torsion = torsion_forces_on_base(q0, u0, phi0, k_theta, c_theta,
                                           i0=1, i1=2)

        W_total = W_gravity + F_tip + F_torsion

        q_new, error = objfun(q0, u0, dt, tol, maximum_iter,
                              m, mMat, EI, EA, W_total, C,
                              deltaL, free_index)
        if error < 0:
            print("Pre-settle: did not converge")
            break

        u_new = (q_new - q0) / dt
        q0 = q_new.copy()
        u0 = u_new.copy()
        ctime += dt

    # ===== reset time & damping for main run =====
    ctime = 0.0
    c_theta = c_theta_base

    # ============================================================
    # 2) Main dynamic run
    # ============================================================
    t_arr = np.linspace(0.0, totalTime, Nsteps)
    x_tip_series      = np.zeros(Nsteps)
    theta_base_series = np.zeros(Nsteps)
    c_theta_series    = np.zeros(Nsteps)
    Fx_series         = np.zeros(Nsteps)

    x_tip_series[0]      = q0[2*tip_node]
    theta_base_series[0] = phi0
    c_theta_series[0]    = c_theta_base
    Fx_series[0]         = 0.0

    snapshot_times = set(range(0, int(totalTime) + 1, 10))
    snapshots = {}

    for step in range(1, Nsteps):
        t = t_arr[step]

        # ---- tip load only during excitation_duration ----
        if ctime <= excitation_duration:
            F_tip, Fx = external_tip_force(ctime, ndof, tip_node, P0, omega)
        else:
            F_tip = np.zeros(ndof)
            Fx = 0.0

        # ---- base angle & angular rate from CURRENT state ----
        phi_curr, phi_dot, _, _ = get_base_angle_and_rate(q0, u0, i0=1, i1=2)

        # ------ choose torsional damping according to control_mode ------
        if control_mode == "open":
            c_theta = c_theta_base

        elif control_mode == "semi":
            # s > 0 : moving toward equilibrium → high damping
            # s < 0 : moving away from equilibrium → low damping
            s = phi_curr * phi_dot

            if s > 0:
                c_target = c_theta_max
            elif s < 0:
                c_target = c_theta_min
            else:
                c_target = c_theta_base

            # smooth update to avoid jumps
            beta = 0.3
            c_theta = (1.0 - beta) * c_theta + beta * c_target

        else:
            raise ValueError("control_mode must be 'open' or 'semi'.")

        # ----- compute updated torsional force for THIS step -----
        F_torsion = torsion_forces_on_base(q0, u0, phi0, k_theta, c_theta,
                                           i0=1, i1=2)

        # ---- total external force ----
        W_total = W_gravity + F_tip + F_torsion

        # ---- implicit step ----
        q_new, error = objfun(q0, u0, dt, tol, maximum_iter,
                              m, mMat, EI, EA, W_total, C,
                              deltaL, free_index)
        if error < 0:
            print(f"Simulation {control_mode} did not converge at t = {ctime:.3f} s")
            break

        u_new = (q_new - q0) / dt
        ctime += dt

        # ---- record history ----
        x_arr = q_new[::2]
        y_arr = q_new[1::2]

        x_tip_series[step]      = x_arr[tip_node]
        theta_base_series[step] = phi_curr
        c_theta_series[step]    = c_theta
        Fx_series[step]         = Fx

        if int(ctime) in snapshot_times and int(ctime) not in snapshots:
            snapshots[int(ctime)] = (x_arr.copy(), y_arr.copy())

        q0 = q_new.copy()
        u0 = u_new.copy()

    result = {
        "t_arr": t_arr,
        "x_tip": x_tip_series,
        "theta_base": theta_base_series,
        "c_theta": c_theta_series,
        "Fx_tip": Fx_series,
        "snapshots": snapshots,
        "nodes_init": nodes,
        "midNode": midNode
    }
    return result




if __name__ == "__main__":
    dt = 0.05
    totalTime = 20.0
    nv = 51
    RodLength = 20.0
    initial_angle_deg = 60.0

    print("Running open-loop simulation (fixed damping)...")
    res_open = run_simulation(
        control_mode="open",
        dt=dt,
        totalTime=totalTime,
        nv=nv,
        RodLength=RodLength,
        initial_angle_deg=initial_angle_deg
    )

    print("Running semi-active damping simulation...")
    res_semi = run_simulation(
        control_mode="semi",
        dt=dt,
        totalTime=totalTime,
        nv=nv,
        RodLength=RodLength,
        initial_angle_deg=initial_angle_deg
    )

    t_arr = res_open["t_arr"]

    # ----------------- Plot: tip horizontal displacement -----------------
    plt.figure()
    plt.plot(t_arr, res_open["x_tip"], label="Open-loop (fixed damping)")
    plt.plot(t_arr, res_semi["x_tip"], label="Closed-loop (semi-active damping)")
    plt.xlabel("t [s]")
    plt.ylabel("Tip x-displacement [m]")
    plt.title("Tip horizontal displacement vs time")
    plt.legend()
    plt.tight_layout()

    # ----------------- Plot: base angle -----------------
    plt.figure()
    plt.plot(t_arr, res_open["theta_base"], label="Open-loop")
    plt.plot(t_arr, res_semi["theta_base"], label="Semi-active")
    plt.xlabel("t [s]")
    plt.ylabel("Base angle φ(t) [rad]")
    plt.title("Base segment angle vs time")
    plt.legend()
    plt.tight_layout()

    # ----------------- Plot: torsional damping coefficient -----------------
    plt.figure()
    plt.plot(t_arr, res_open["c_theta"], label="Open-loop (fixed c_theta)")
    plt.plot(t_arr, res_semi["c_theta"], label="Semi-active c_theta(t)")
    plt.xlabel("t [s]")
    plt.ylabel("c_theta(t) [N·m·s/rad]")
    plt.title("Torsional damping coefficient vs time")
    plt.legend()
    plt.tight_layout()

    # ----------------- Plot: applied tip force -----------------
    plt.figure()
    plt.plot(t_arr, res_open["Fx_tip"], label="Applied tip force Fx(t)")
    plt.xlabel("t [s]")
    plt.ylabel("F_x at tip [N]")
    plt.title("Applied sinusoidal force at tip")
    plt.legend()
    plt.tight_layout()

    # ----------------- Optional: snapshots of deformed shape (semi-active) ---
    snapshots = res_semi["snapshots"]
    mid_idx   = res_semi["midNode"]

    for t_snap in sorted(snapshots.keys()):
        xa, ya = snapshots[t_snap]
        plt.figure()
        plt.plot(xa, ya, "k.-", linewidth=1, label="Ladder")
        plt.plot(xa[mid_idx], ya[mid_idx], "ro", markersize=6,
                 label="Middle node")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"Ladder shape (semi-active) at t = {t_snap} s")
        plt.legend()
        plt.tight_layout()

    plt.show()
