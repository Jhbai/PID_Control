import control as ct
import numpy as np

def generate_ph_control_series():
    N = 3000
    dt = 15
    s = ct.tf('s')
    
    P_s = 1.0 / (3.0 * s + 1.0)
    
    Kp = 3.0
    Ki = 0.1
    Kd = 1.0
    C_s = Kp + Ki / (s + 0.1) + Kd * s / (0.3 * s + 1.0)
    
    P_ss = ct.ss(P_s)
    C_ss = ct.ss(C_s)
    
    P_d = ct.sample_system(P_ss, dt)
    Ca_d = ct.sample_system(C_ss, dt)
    Cb_d = ct.sample_system(C_ss, dt)
    
    x_P = np.zeros((P_d.nstates, 1))
    x_Ca = np.zeros((Ca_d.nstates, 1))
    x_Cb = np.zeros((Cb_d.nstates, 1))
    
    ph_trg = []
    acid_rate = []
    base_rate = []
    
    disturbance_state = 1.5
    
    for _ in range(N):
        if np.random.rand() < 0.01:
            disturbance_state = -disturbance_state
            
        d = disturbance_state + np.random.randn() * 0.15
        
        pH_val = (P_d.C @ x_P).item() + 6.25 + np.random.randn() * 0.04
        
        e_a = pH_val - 6.5
        e_b = 6.0 - pH_val
        
        u_a_raw = (Ca_d.C @ x_Ca + Ca_d.D * e_a).item()
        u_b_raw = (Cb_d.C @ x_Cb + Cb_d.D * e_b).item()
        
        u_a = max(0.0, min(u_a_raw, 5.0))
        u_b = max(0.0, min(u_b_raw, 5.0))
        
        u_total = u_b - u_a + d
        
        x_P = P_d.A @ x_P + P_d.B * u_total
        x_Ca = Ca_d.A @ x_Ca + Ca_d.B * e_a
        x_Cb = Cb_d.A @ x_Cb + Cb_d.B * e_b
        
        ph_trg.append(pH_val)
        acid_rate.append(u_a)
        base_rate.append(u_b)
        
    return ph_trg, acid_rate, base_rate

import numpy as np
from scipy.optimize import lsq_linear

import numpy as np
from scipy.optimize import lsq_linear

def vrft_pid_tuning(ph_trg, acid_rate, base_rate):
    y = np.array(ph_trg)
    ua = np.array(acid_rate)
    ub = np.array(base_rate)
    N = len(y)
    dt = 0.2
    tau = 1.0
    alpha = np.exp(-dt / tau)
    
    dya = np.insert(np.diff(y), 0, 0)
    dyb = -dya
    
    eva = (alpha / (1.0 - alpha)) * dya
    evb = (alpha / (1.0 - alpha)) * dyb
    
    L_eva = np.zeros(N)
    L_evb = np.zeros(N)
    L_ua = np.zeros(N)
    L_ub = np.zeros(N)
    
    for i in range(1, N):
        L_eva[i] = alpha * L_eva[i-1] + (1.0 - alpha) * eva[i]
        L_evb[i] = alpha * L_evb[i-1] + (1.0 - alpha) * evb[i]
        L_ua[i] = alpha * L_ua[i-1] + (1.0 - alpha) * ua[i]
        L_ub[i] = alpha * L_ub[i-1] + (1.0 - alpha) * ub[i]
        
    dL_eva = np.insert(np.diff(L_eva), 0, 0)
    dL_evb = np.insert(np.diff(L_evb), 0, 0)
    dL_ua = np.insert(np.diff(L_ua), 0, 0)
    dL_ub = np.insert(np.diff(L_ub), 0, 0)
    
    ddL_eva = np.insert(np.diff(dL_eva), 0, 0)
    ddL_evb = np.insert(np.diff(dL_evb), 0, 0)
    
    Phi_a = np.column_stack((dL_eva, L_eva * dt, ddL_eva / dt))
    Phi_b = np.column_stack((dL_evb, L_evb * dt, ddL_evb / dt))
    
    y_target_a = dL_ua
    y_target_b = dL_ub
    
    valid_a = (ua > 0.05) & (ua < 4.95)
    if np.sum(valid_a) < 10:
        valid_a = np.ones(N, dtype=bool)
        
    valid_b = (ub > 0.05) & (ub < 4.95)
    if np.sum(valid_b) < 10:
        valid_b = np.ones(N, dtype=bool)
        
    prior_a = np.array([8.5, 0.0, 0.7])
    prior_b = np.array([2.7, 0.0, 0.3])
    
    lam_a = 0.05 * np.trace(Phi_a[valid_a].T @ Phi_a[valid_a]) + 1e-3
    lam_b = 0.05 * np.trace(Phi_b[valid_b].T @ Phi_b[valid_b]) + 1e-3
    
    Phi_a_aug = np.vstack((Phi_a[valid_a], np.sqrt(lam_a) * np.eye(3)))
    uf_a_aug = np.concatenate((y_target_a[valid_a], np.sqrt(lam_a) * prior_a))
    
    Phi_b_aug = np.vstack((Phi_b[valid_b], np.sqrt(lam_b) * np.eye(3)))
    uf_b_aug = np.concatenate((y_target_b[valid_b], np.sqrt(lam_b) * prior_b))
    
    res_a = lsq_linear(Phi_a_aug, uf_a_aug, bounds=(0.0, 10.0))
    res_b = lsq_linear(Phi_b_aug, uf_b_aug, bounds=(0.0, 10.0))
    
    return res_a.x.tolist() + res_b.x.tolist()

ph, ac, bs = generate_ph_control_series()
fig, ax = plt.subplots(3, 1, figsize=(24, 9))
ax[0].plot(ph, color="black")
ax[1].plot(ac, color="red")
ax[2].plot(bs, color="blue")
for a in ax:
  a.grid(color="gray", linestyle="--", alpha=.4)
plt.show()
tuned_params = vrft_pid_tuning(ph, ac, bs)
print(tuned_params)
