!pip install control -q
import numpy as np
import control as ct
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


def calculate_virtual_signals(y, tau, Ts):
    y = np.asarray(y, dtype=float)
    Td_s = ct.TransferFunction([1], [tau, 1])
    Td_z = ct.sample_system(Td_s, Ts, method='zoh')
    Td_inv = 1 / Td_z

    num = Td_inv.num[0][0]
    den = Td_inv.den[0][0]
    rv_delayed = signal.lfilter(num, den, y)

    shift = len(num) - len(den)
    rv = np.zeros_like(y)
    if shift > 0:
        rv[:-shift] = rv_delayed[shift:]
        rv[-shift:] = rv_delayed[-1]
    elif shift < 0:
        rv[-shift:] = rv_delayed[:shift]
        rv[:-shift] = rv_delayed[0]
    else:
        rv = rv_delayed
    ev = rv - y
    return rv, ev


def calculate_pid_basis(ev, Ts):
    ev = np.asarray(ev, dtype=float)
    phi_p = ev.copy()
    phi_i = np.cumsum(ev) * Ts
    phi_d = np.zeros_like(ev)
    phi_d[1:] = (ev[1:] - ev[:-1]) / Ts
    phi = np.column_stack((phi_p, phi_i, phi_d))
    return phi


def apply_data_filter(u, phi, tau, Ts):
    u = np.asarray(u, dtype=float)
    Td_s = ct.TransferFunction([1], [tau, 1])
    Td_z = ct.sample_system(Td_s, Ts, method='zoh')
    L_z = Td_z * (1 - Td_z)
    num = L_z.num[0][0]
    den = L_z.den[0][0]
    uf = signal.lfilter(num, den, u)
    phi_f = signal.lfilter(num, den, phi, axis=0)
    return uf, phi_f

def calculate_optimal_pid(uf, phi_f):
    uf_flat = np.asarray(uf).ravel()
    phi_f_matrix = np.asarray(phi_f)
    result = lsq_linear(phi_f_matrix, uf_flat, bounds=(0, 10))
    return tuple(result.x)

def vrft_pid_pipeline(u, y, tau, Ts):
    rv, ev = calculate_virtual_signals(y, tau, Ts)
    phi = calculate_pid_basis(ev, Ts)
    uf, phi_f = apply_data_filter(u, phi, tau, Ts)
    kp, ki, kd = calculate_optimal_pid(uf, phi_f)
    return kp, ki, kd

def generate_pid_data(kp, ki, kd, Ts, t_end=50.0):
    P_s = ct.TransferFunction([1], [1, 2, 1])
    P_z = ct.sample_system(P_s, Ts, method='zoh')
    z = ct.TransferFunction([1, 0], [1], dt=Ts)
    C_z = kp + ki * (Ts * z / (z - 1)) + kd * ((z - 1) / (Ts * z))
    T_y = ct.feedback(P_z * C_z, 1)
    T_u = ct.feedback(C_z, P_z)
    t = np.arange(0, t_end, Ts)
    r = np.ones_like(t)
    _, y = ct.forced_response(T_y, T=t, U=r)
    _, u = ct.forced_response(T_u, T=t, U=r)
    return u, y

if __name__ == "__main__":
    Ts = 0.1
    t_end = 50.0
    tau = 1.5

    kp_init = 0.5
    ki_init = 0.2
    kd_init = 0.1

    u_init, y_init = generate_pid_data(kp_init, ki_init, kd_init, Ts, t_end)

    kp_opt, ki_opt, kd_opt = vrft_pid_pipeline(u_init, y_init, tau, Ts)

    u_opt, y_opt = generate_pid_data(kp_opt, ki_opt, kd_opt, Ts, t_end)

    t = np.arange(0, t_end, Ts)
    r = np.ones_like(t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, r, 'k--', label="Reference")
    ax1.plot(t, y_init, 'b-', label=f"Initial (Kp={kp_init:.2f}, Ki={ki_init:.2f}, Kd={kd_init:.2f})")
    ax1.plot(t, y_opt, 'r-', label=f"Optimized (Kp={kp_opt:.2f}, Ki={ki_opt:.2f}, Kd={kd_opt:.2f})")
    ax1.set_title("System Output (y)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, u_init, 'b-', label="Initial Control Signal")
    ax2.plot(t, u_opt, 'r-', label="Optimized Control Signal")
    ax2.set_title("Control Signal (u)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
