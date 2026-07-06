import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import lfilter, lfilter_zi, butter, filtfilt

def generate_frit_reference_with_delay(y_out, omega, base_dt, delay_steps, zeta=1.0):
    if omega <= 0 or base_dt <= 0:
        return np.copy(y_out)

    nyq = 0.5 / base_dt
    cutoff = min(omega * 5.0, nyq * 0.8)
    b_butt, a_butt = butter(2, cutoff / nyq, btype='low')
    y_smooth = filtfilt(b_butt, a_butt, y_out)

    w_dt = omega * base_dt
    w2_dt2 = w_dt ** 2

    b0 = 1.0 / w2_dt2 + 2.0 * zeta / w_dt + 1.0
    b1 = -2.0 / w2_dt2 - 2.0 * zeta / w_dt
    b2 = 1.0 / w2_dt2

    b = np.array([b0, b1, b2])
    a = np.array([1.0])

    zi = lfilter_zi(b, a) * y_smooth[0]
    rf_pure, _ = lfilter(b, a, y_smooth, zi=zi)

    if delay_steps > 0:
        rf_advanced = np.full_like(rf_pure, rf_pure[-1])
        rf_advanced[:-delay_steps] = rf_pure[delay_steps:]
        return rf_advanced

    return rf_pure

def dual_pid_gain_scheduled_sim(params, y_out, ph_in, flow_in, base_dt, delay_steps, e_a, e_b):
    (kp_a_sens, kp_a_buf, ki_a, kd_a, max_out_a, cycle_mult_a,
     kp_b_sens, kp_b_buf, ki_b, kd_b, max_out_b, cycle_mult_b,
     f_ph_a, f_flow_a, f_ph_b, f_flow_b) = params

    cycle_dt_a = base_dt * max(1, int(round(cycle_mult_a)))
    cycle_dt_b = base_dt * max(1, int(round(cycle_mult_b)))

    n = len(y_out)
    v_a_sim = np.zeros(n)
    v_b_sim = np.zeros(n)

    u_a_state = 0.0
    e_a_last = 0.0
    e_a_prev = 0.0
    active_a = False
    last_t_a = -np.inf

    u_b_state = 0.0
    e_b_last = 0.0
    e_b_prev = 0.0
    active_b = False
    last_t_b = -np.inf

    start_k = max(2, delay_steps) if delay_steps > 0 else 2

    for k in range(start_k, n):
        current_t = k * base_dt
        current_ph = y_out[k]

        c_oh_in = (10.0 ** (ph_in[k] - 14.0)) * 1e6 if ph_in[k] > 7.0 else 0.0
        c_h_in  = (10.0 ** (-ph_in[k])) * 1e6       if ph_in[k] < 7.0 else 0.0

        ff_a = c_oh_in * f_ph_a * flow_in[k] + f_flow_a if c_oh_in > 0 else 0.0
        ff_b = c_h_in  * f_ph_b * flow_in[k] + f_flow_b if c_h_in > 0 else 0.0

        if current_ph > 6.5:
            active_b = False
            v_b_sim[k] = 0.0
            u_b_state = 0.0

            kp_a = kp_a_sens if current_ph <= 8.5 else kp_a_buf

            if not active_a:
                active_a = True
                e_a_last = e_a[k]
                e_a_prev = e_a[k]
                u_a_state = max(0.0, v_a_sim[k - 1] - ff_a)
                last_t_a = current_t
                v_out = np.clip(u_a_state + ff_a, 0.0, max_out_a)
                v_a_sim[k] = v_out
                u_a_state = max(0.0, v_out - ff_a)

            elif current_t - last_t_a >= cycle_dt_a - 1e-9:
                e_k = e_a[k]
                de_a1 = e_k - e_a_last
                de_a2 = e_k - 2.0 * e_a_last + e_a_prev

                du_a = kp_a * de_a1 + ki_a * e_k * cycle_dt_a + (kd_a / cycle_dt_a) * de_a2

                u_a_state_raw = u_a_state + du_a
                u_a_state = np.clip(u_a_state_raw, 0.0, max_out_a)

                v_out = np.clip(u_a_state + ff_a, 0.0, max_out_a)
                v_a_sim[k] = v_out
                u_a_state = max(0.0, v_out - ff_a)

                e_a_prev = e_a_last
                e_a_last = e_k
                last_t_a = current_t
            else:
                v_out = np.clip(u_a_state + ff_a, 0.0, max_out_a)
                v_a_sim[k] = v_out

        elif current_ph < 6.0:
            active_a = False
            v_a_sim[k] = 0.0
            u_a_state = 0.0

            kp_b = kp_b_sens if current_ph >= 4.5 else kp_b_buf

            if not active_b:
                active_b = True
                e_b_last = e_b[k]
                e_b_prev = e_b[k]
                u_b_state = max(0.0, v_b_sim[k - 1] - ff_b)
                last_t_b = current_t
                v_out = np.clip(u_b_state + ff_b, 0.0, max_out_b)
                v_b_sim[k] = v_out
                u_b_state = max(0.0, v_out - ff_b)

            elif current_t - last_t_b >= cycle_dt_b - 1e-9:
                e_k = e_b[k]
                de_b1 = e_k - e_b_last
                de_b2 = e_k - 2.0 * e_b_last + e_b_prev

                du_b = kp_b * de_b1 + ki_b * e_k * cycle_dt_b + (kd_b / cycle_dt_b) * de_b2

                u_b_state_raw = u_b_state + du_b
                u_b_state = np.clip(u_b_state_raw, 0.0, max_out_b)

                v_out = np.clip(u_b_state + ff_b, 0.0, max_out_b)
                v_b_sim[k] = v_out
                u_b_state = max(0.0, v_out - ff_b)

                e_b_prev = e_b_last
                e_b_last = e_k
                last_t_b = current_t
            else:
                v_out = np.clip(u_b_state + ff_b, 0.0, max_out_b)
                v_b_sim[k] = v_out

        else:
            active_a = False
            active_b = False
            v_a_sim[k] = 0.0
            v_b_sim[k] = 0.0
            u_a_state = 0.0
            u_b_state = 0.0

    return v_a_sim, v_b_sim

def frit_gs_objective(params, v_a_hist, v_b_hist, y_out, ph_in, flow_in, base_dt,
                       delay_steps, e_a, e_b, lambda_smooth=0.1, lambda_consump=0.0):
    v_a_sim, v_b_sim = dual_pid_gain_scheduled_sim(
        params, y_out, ph_in, flow_in, base_dt, delay_steps, e_a, e_b
    )

    if delay_steps > 0:
        end_idx = len(y_out) - delay_steps
        v_a_hist_eval = v_a_hist[delay_steps:end_idx]
        v_b_hist_eval = v_b_hist[delay_steps:end_idx]
        v_a_sim_eval = v_a_sim[delay_steps:end_idx]
        v_b_sim_eval = v_b_sim[delay_steps:end_idx]
    else:
        v_a_hist_eval = v_a_hist
        v_b_hist_eval = v_b_hist
        v_a_sim_eval = v_a_sim
        v_b_sim_eval = v_b_sim

    loss_a_fit = np.mean((v_a_hist_eval - v_a_sim_eval) ** 2)
    loss_b_fit = np.mean((v_b_hist_eval - v_b_sim_eval) ** 2)

    diff_a = np.diff(v_a_sim_eval)
    diff_b = np.diff(v_b_sim_eval)
    
    smoothness_penalty = 0.0
    if len(diff_a) > 0:
        smoothness_penalty = lambda_smooth * (np.mean(diff_a ** 2) + np.mean(diff_b ** 2))

    consump_penalty = lambda_consump * (np.mean(v_a_sim_eval) + np.mean(v_b_sim_eval))

    return loss_a_fit + loss_b_fit + smoothness_penalty + consump_penalty


def optimize_gs_frit(df, base_dt=1.0, delay_steps=0, omega_target=0.015,
                      zeta_a=1.2, zeta_b=1.2, lambda_smooth=2.0, lambda_consump=0.0):
    ph_in = df['src_pH'].to_numpy()
    flow_in = df['src_fiq'].to_numpy()
    v_a_hist = df['acid_pid'].to_numpy()
    v_b_hist = df['base_pid'].to_numpy()
    y_out = df['trg_pH'].to_numpy()

    y_dev_a = y_out - 6.5
    y_dev_b = 6.0 - y_out

    rf_a = generate_frit_reference_with_delay(y_dev_a, omega_target, base_dt, delay_steps, zeta=zeta_a)
    rf_b = generate_frit_reference_with_delay(y_dev_b, omega_target, base_dt, delay_steps, zeta=zeta_b)

    e_a = y_dev_a - rf_a
    e_b = y_dev_b - rf_b

    bounds = [
        (0.0, 50.0), (0.0, 50.0), (0.0, 5.0), (0.0, 2.0), (10.0, 100.0), (1.0, 2.0),
        (0.0, 50.0), (0.0, 50.0), (0.0, 5.0), (0.0, 2.0), (10.0, 100.0), (1.0, 2.0),
        (0.0, 0.0), (0.0, 0.0),
        (0.0, 0.0), (0.0, 0.0)
    ]

    print("開始執行 Differential Evolution 全局最佳化，請稍候 (約 10~20 秒)...")
    
    result = differential_evolution(
        frit_gs_objective,
        bounds=bounds,
        args=(v_a_hist, v_b_hist, y_out, ph_in, flow_in, base_dt, delay_steps,
              e_a, e_b, lambda_smooth, lambda_consump),
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=1e-3,
        disp=True,
        workers=-1
    )

    res = result.x
    return {
        'Acid Valve (Sensitive Zone Kp)': res[0],
        'Acid Valve (Buffer Zone Kp)': res[1],
        'Acid Valve Ki': res[2],
        'Acid Valve Kd': res[3],
        'Acid Valve Max Out': res[4],
        'Acid Valve Cycle Time': base_dt * max(1, int(round(res[5]))),
        'Base Valve (Sensitive Zone Kp)': res[6],
        'Base Valve (Buffer Zone Kp)': res[7],
        'Base Valve Ki': res[8],
        'Base Valve Kd': res[9],
        'Base Valve Max Out': res[10],
        'Base Valve Cycle Time': base_dt * max(1, int(round(res[11]))),
        'Acid Feedforward Conc Multiplier': res[12],
        'Acid Feedforward Constant Bias': res[13],
        'Base Feedforward Conc Multiplier': res[14],
        'Base Feedforward Constant Bias': res[15],
        'Objective Value': result.fun,
    }


if __name__ == '__main__':
    np.random.seed(42)
    n_pts = 600
    base_dt = 1.0
    delay_steps = 0

    src_ph_arr = np.concatenate([np.random.normal(10.5, 0.2, 300), np.random.normal(3.5, 0.2, 300)])
    src_flow_arr = np.random.normal(50.0, 2.0, n_pts)
    
    trg_ph_arr = np.zeros(n_pts)
    trg_ph_arr[:300] = 7.5 + 1.5 * np.sin(np.linspace(0, 10, 300)) + np.random.normal(0, 0.1, 300)
    trg_ph_arr[300:] = 5.0 + 1.0 * np.cos(np.linspace(0, 10, 300)) + np.random.normal(0, 0.1, 300)

    acid_pid_arr = np.zeros(n_pts)
    base_pid_arr = np.zeros(n_pts)

    for i in range(n_pts):
        if trg_ph_arr[i] > 6.5:
            acid_pid_arr[i] = np.clip((trg_ph_arr[i] - 6.5) * 20.0 + np.random.normal(0, 2.0), 0, 100)
        elif trg_ph_arr[i] < 6.0:
            base_pid_arr[i] = np.clip((6.0 - trg_ph_arr[i]) * 20.0 + np.random.normal(0, 2.0), 0, 100)

    df = pd.DataFrame({
        'src_pH': src_ph_arr,
        'src_fiq': src_flow_arr,
        'trg_pH': trg_ph_arr,
        'acid_pid': acid_pid_arr,
        'base_pid': base_pid_arr
    })

    result_params = optimize_gs_frit(df, base_dt=base_dt, delay_steps=delay_steps)

    opt_x = [
        result_params['Acid Valve (Sensitive Zone Kp)'],
        result_params['Acid Valve (Buffer Zone Kp)'],
        result_params['Acid Valve Ki'],
        result_params['Acid Valve Kd'],
        result_params['Acid Valve Max Out'],
        result_params['Acid Valve Cycle Time'] / base_dt,
        result_params['Base Valve (Sensitive Zone Kp)'],
        result_params['Base Valve (Buffer Zone Kp)'],
        result_params['Base Valve Ki'],
        result_params['Base Valve Kd'],
        result_params['Base Valve Max Out'],
        result_params['Base Valve Cycle Time'] / base_dt,
        result_params['Acid Feedforward Conc Multiplier'],
        result_params['Acid Feedforward Constant Bias'],
        result_params['Base Feedforward Conc Multiplier'],
        result_params['Base Feedforward Constant Bias']
    ]

    omega_target = 0.015
    zeta_a = 1.2
    zeta_b = 1.2

    y_dev_a = trg_ph_arr - 6.5
    y_dev_b = 6.0 - trg_ph_arr
    
    rf_a = generate_frit_reference_with_delay(y_dev_a, omega_target, base_dt, delay_steps, zeta_a)
    rf_b = generate_frit_reference_with_delay(y_dev_b, omega_target, base_dt, delay_steps, zeta_b)
    
    e_a = y_dev_a - rf_a
    e_b = y_dev_b - rf_b

    v_a_sim, v_b_sim = dual_pid_gain_scheduled_sim(
        opt_x, trg_ph_arr, src_ph_arr, src_flow_arr, base_dt, delay_steps, e_a, e_b
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    axes[0].plot(src_ph_arr, label='Source pH', color='purple')
    axes[0].set_title('Source pH Profile')
    axes[0].set_ylabel('pH')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    axes[1].plot(trg_ph_arr, label='Target pH', color='green')
    axes[1].axhline(6.5, color='red', linestyle='--', label='Acid Activation Threshold (6.5)')
    axes[1].axhline(6.0, color='blue', linestyle='--', label='Base Activation Threshold (6.0)')
    axes[1].set_title('Target pH Profile')
    axes[1].set_ylabel('pH')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    axes[2].plot(acid_pid_arr, label='Historical Acid PID', color='black', alpha=0.6)
    axes[2].plot(v_a_sim, label='Simulated Acid PID', color='red', linestyle='--')
    axes[2].set_title('Acid Control Signal')
    axes[2].set_ylabel('Valve Output (%)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    axes[3].plot(base_pid_arr, label='Historical Base PID', color='black', alpha=0.6)
    axes[3].plot(v_b_sim, label='Simulated Base PID', color='blue', linestyle='--')
    axes[3].set_title('Base Control Signal')
    axes[3].set_xlabel('Time Steps')
    axes[3].set_ylabel('Valve Output (%)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()
