import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

def generate_frit_reference_with_delay(y_out, target_ph, omega, base_dt, delay_steps):
    """
    建立帶有死區時間(Dead Time)的理想閉迴路二階參考軌跡
    """
    A = np.array([1.0, -2.0, 1.0])
    B = np.array([omega**2 * base_dt**2, 2 * omega**2 * base_dt**2, omega**2 * base_dt**2])
    rf_pure = lfilter(A, B, y_out) + target_ph
    
    # 塞入 Dead Time 修正：將理想軌跡整體向後平移 delay_steps
    if delay_steps > 0:
        rf_delayed = np.zeros_like(rf_pure)
        rf_delayed[delay_steps:] = rf_pure[:-delay_steps]
        rf_delayed[:delay_steps] = target_ph
        return rf_delayed
    return rf_pure

def dual_pid_gain_scheduled_sim(params, y_out, ph_in, flow_in, base_dt, delay_steps):
    """
    動態增益排程與抗時滯的雙迴路 PID 虛擬模擬器
    """
    # 解包優化參數 (酸閥與鹼閥各有獨立的 kp_sens, kp_buf, ki, kd, max_out, cycle_mult)
    (kp_a_sens, kp_a_buf, ki_a, kd_a, max_out_a, cycle_mult_a,
     kp_b_sens, kp_b_buf, ki_b, kd_b, max_out_b, cycle_mult_b,
     f_ph, f_flow) = params
    
    # 計算各自的控制採樣週期
    cycle_dt_a = base_dt * max(1, int(round(cycle_mult_a)))
    cycle_dt_b = base_dt * max(1, int(round(cycle_mult_b)))
    
    n = len(y_out)
    v_a_sim = np.zeros(n)
    v_b_sim = np.zeros(n)
    
    # 根據製程期望，設定酸閥目標 6.5，鹼閥目標 6.0 的動態引導線
    rf_a = generate_frit_reference_with_delay(y_out, 6.5, 0.02, base_dt, delay_steps)
    rf_b = generate_frit_reference_with_delay(y_out, 6.0, 0.02, base_dt, delay_steps)
    
    e_a = y_out - rf_a  # 出口 pH 高於 6.5 時，酸閥的虛擬誤差
    e_b = rf_b - y_out  # 出口 pH 低於 6.0 時，鹼閥的虛擬誤差
    
    u_a_state = 0.0
    u_b_state = 0.0
    last_k_a = 0
    last_k_b = 0
    
    start_k = max(2, delay_steps)
    
    for k in range(start_k, n):
        current_ph = y_out[k]
        
        # 前饋補償計算
        ff_term = f_ph * (ph_in[k] - 7.0) * flow_in[k] + f_flow * flow_in[k]
        
        # === 狀況一：出口 pH 偏高 -> 啟動酸閥控制 (Setpoint = 6.5) ===
        if current_ph > 6.5:
            if (k - last_k_a) * base_dt >= cycle_dt_a:
                # 【Gain Scheduling】: 區分敏感區與強鹼緩衝區
                kp_a = kp_a_sens if current_ph <= 8.5 else kp_a_buf
                
                de_a1 = e_a[k] - e_a[last_k_a]
                de_a2 = e_a[k] - 2 * e_a[last_k_a] + (e_a[last_k_a - 1] if last_k_a > 0 else e_a[0])
                du_a = kp_a * de_a1 + ki_a * e_a[k] * cycle_dt_a + (kd_a / cycle_dt_a) * de_a2
                
                u_a_state = np.clip(u_a_state + du_a + ff_term, 0, max_out_a)
                v_a_sim[k] = u_a_state
                last_k_a = k
            else:
                v_a_sim[k] = v_a_sim[last_k_a]
            
            # 酸閥啟動時，鹼閥強制歸零並重置狀態
            v_b_sim[k] = 0.0
            u_b_state = 0.0
            
        # === 狀況二：出口 pH 偏低 -> 啟動鹼閥控制 (Setpoint = 6.0) ===
        elif current_ph < 6.0:
            if (k - last_k_b) * base_dt >= cycle_dt_b:
                # 【Gain Scheduling】: 區分敏感區與強酸緩衝區
                kp_b = kp_b_sens if current_ph >= 4.5 else kp_b_buf
                
                de_b1 = e_b[k] - e_b[last_k_b]
                de_b2 = e_b[k] - 2 * e_b[last_k_b] + (e_b[last_k_b - 1] if last_k_b > 0 else e_b[0])
                du_b = kp_b * de_b1 + ki_b * e_b[k] * cycle_dt_b + (kd_b / cycle_dt_b) * de_b2
                
                u_b_state = np.clip(u_b_state + du_b - ff_term, 0, max_out_b)
                v_b_sim[k] = u_b_state
                last_k_b = k
            else:
                v_b_sim[k] = v_b_sim[last_k_b]
            
            # 鹼閥啟動時，酸閥強制歸零並重置狀態
            v_a_sim[k] = 0.0
            u_a_state = 0.0
            
        # === 狀況三：處於 6.0 ~ 6.5 死區帶 -> 雙閥皆不動作 ===
        else:
            v_a_sim[k] = 0.0
            v_b_sim[k] = 0.0
            u_a_state = 0.0
            u_b_state = 0.0
            
    return v_a_sim, v_b_sim

def frit_gs_objective(params, v_a_hist, v_b_hist, y_out, ph_in, flow_in, base_dt, delay_steps):
    v_a_sim, v_b_sim = dual_pid_gain_scheduled_sim(params, y_out, ph_in, flow_in, base_dt, delay_steps)
    
    # 1. 模型擬合項 (維持原樣，確保學到系統的非線性物理特性)
    loss_a_fit = np.sum((v_a_hist[delay_steps:] - v_a_sim[delay_steps:]) ** 2)
    loss_b_fit = np.sum((v_b_hist[delay_steps:] - v_b_sim[delay_steps:]) ** 2)
    
    # 2. 新增：用藥成本懲罰項 (L1 Regularization)
    # 取絕對值加總，在物理上精確對應這段時間內的「閥門總耗能/總加藥量」
    lambda_chem = 0.5  # 【這是一個可調權重】如果發現參數變得很遲鈍，調小；如果想更省藥，調大。
    loss_a_chem = np.sum(np.abs(v_a_sim[delay_steps:]))
    loss_b_chem = np.sum(np.abs(v_b_sim[delay_steps:]))
    
    # 總目標函數：兼顧物理模型擬合 與 用藥量最小化
    return (loss_a_fit + loss_b_fit) + lambda_chem * (loss_a_chem + loss_b_chem)

def optimize_gs_frit(df, base_dt=1.0, delay_steps=5):
    ph_in = df['src_pH'].to_numpy()
    flow_in = df['src_fiq'].to_numpy()
    v_a_hist = df['acid_pid'].to_numpy()
    v_b_hist = df['base_pid'].to_numpy()
    y_out = df['trg_pH'].to_numpy()
    
    # 初始猜測值: [酸閥組(6個)], [鹼閥組(6個)], [前饋組(2個)]
    initial_guess = [
        0.5, 2.0, 0.01, 0.005, 50.0, 1.0,  # 酸閥: kp_sens, kp_buf, ki, kd, max_out, cycle_mult
        0.5, 2.0, 0.01, 0.005, 50.0, 1.0,  # 鹼閥: kp_sens, kp_buf, ki, kd, max_out, cycle_mult
        0.01, 0.01                         # 前饋: f_ph, f_flow
    ]
    
    # 設定物理邊界限制
    bounds = [
        (0.0, 10.0), (0.0, 30.0), (0.0, 5.0), (0.0, 2.0), (10.0, 100.0), (1.0, 30.0),
        (0.0, 10.0), (0.0, 30.0), (0.0, 5.0), (0.0, 2.0), (10.0, 100.0), (1.0, 30.0),
        (-5.0, 5.0), (-5.0, 5.0)
    ]
    
    result = minimize(
        frit_gs_objective,
        initial_guess,
        args=(v_a_hist, v_b_hist, y_out, ph_in, flow_in, base_dt, delay_steps),
        method='Powell',
        bounds=bounds,
        options={'maxiter': 5000, 'xtol': 1e-5}
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
        'Feedforward pH Gain': res[12],
        'Feedforward Flow Gain': res[13]
    }


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ==========================================
    # 步驟一：模擬一段包含死區與緩衝效應的真實歷史數據
    # ==========================================
    n_samples = 1200
    base_dt = 1.0
    delay_steps = 8  # 模擬 8 秒的 Dead Time
    
    np.random.seed(42)
    # 源頭擾動：pH 隨時間弦波震盪 + 流量起伏
    src_pH = 4.5 + 2.0 * np.sin(np.linspace(0, 12, n_samples)) + np.random.normal(0, 0.05, n_samples)
    src_fiq = 50.0 + 15 * np.cos(np.linspace(0, 6, n_samples))
    
    acid_pid_hist = np.zeros(n_samples)
    base_pid_hist = np.zeros(n_samples)
    trg_pH_hist = np.full(n_samples, 7.0)  # 出口 pH 歷史紀錄
    
    # 模擬舊控制器的運作 (故意調得很爛，有震盪且反應慢)
    for k in range(20, n_samples):
        # 控制器因為死區，只能看到 8 秒前的源頭變化
        if src_pH[k - delay_steps] < 5.5:
            base_pid_hist[k] = np.clip(45.0 + np.random.normal(0, 2), 0, 100)
        elif src_pH[k - delay_steps] > 7.5:
            acid_pid_hist[k] = np.clip(25.0 + np.random.normal(0, 1), 0, 100)
            
        # 物理非線性環境：酸鹼中和反應 (引入歷史閥門開度的時滯影響)
        # 模擬緩衝效應：在敏感區(6~8)變化極快，在外圍變慢
        net_effect = (base_pid_hist[k - delay_steps] * 0.08) - (acid_pid_hist[k - delay_steps] * 0.12) \
                     + (src_pH[k - delay_steps] - 7.0) * (src_fiq[k - delay_steps] / 50.0)
                     
        # 狀態方程式更新
        prev_ph = trg_pH_hist[k - 1]
        if 6.0 <= prev_ph <= 8.0:
            # 敏感區：增益極高
            trg_pH_hist[k] = 6.5 + 0.75 * (prev_ph - 6.5) + 0.15 * net_effect + np.random.normal(0, 0.02)
        else:
            # 緩衝區：阻尼大、推不動
            trg_pH_hist[k] = 6.5 + 0.95 * (prev_ph - 6.5) + 0.02 * net_effect + np.random.normal(0, 0.01)

    # 建立 DataFrame 封裝
    df_test = pd.DataFrame({
        'src_pH': src_pH, 'src_fiq': src_fiq,
        'acid_pid': acid_pid_hist, 'base_pid': base_pid_hist, 'trg_pH': trg_pH_hist
    })

    # ==========================================
    # 步驟二：執行 Gain-Scheduled FRIT 參數優化
    # ==========================================
    print(">> Starting Gain-Scheduled FRIT Optimization...")
    best_params_dict = optimize_gs_frit(df_test, base_dt=base_dt, delay_steps=delay_steps)
    
    print("\n====== Optimized Configurations ======")
    for key, val in best_params_dict.items():
        print(f"{key:<35} : {val:.4f}")

    # ==========================================
    # 步驟三：將黃金參數帶入模擬器，提取 v_sim 準備畫圖
    # ==========================================
    # 重新將 dict 轉回 minimize 需要的 list 格式以利帶入 sim 函式
    extracted_params = list(best_params_dict.values())
    # 修正：補上前饋增益與組裝
    f_ph_opt = extracted_params[-2]
    f_flow_opt = extracted_params[-1]
    
    # 依序還原 params list 丟給模擬器
    p_sim = [
        best_params_dict['Acid Valve (Sensitive Zone Kp)'], best_params_dict['Acid Valve (Buffer Zone Kp)'],
        best_params_dict['Acid Valve Ki'], best_params_dict['Acid Valve Kd'], best_params_dict['Acid Valve Max Out'], 1.0, # 採樣倍率固定還原
        best_params_dict['Base Valve (Sensitive Zone Kp)'], best_params_dict['Base Valve (Buffer Zone Kp)'],
        best_params_dict['Base Valve Ki'], best_params_dict['Base Valve Kd'], best_params_dict['Base Valve Max Out'], 1.0,
        f_ph_opt, f_flow_opt
    ]
    
    v_a_sim, v_b_sim = dual_pid_gain_scheduled_sim(p_sim, trg_pH_hist, src_pH, src_fiq, base_dt, delay_steps)

    # ==========================================
    # 步驟四：資料視覺化 (Matplotlib 繪圖)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    time_axis = np.arange(n_samples)

    # Subplot 1: 出口 pH 軌跡
    axes[0].plot(time_axis, trg_pH_hist, color='black', alpha=0.7, label='Historical Target pH')
    axes[0].axhline(6.5, color='red', linestyle='--', alpha=0.5, label='Acid Setpoint (6.5)')
    axes[0].axhline(6.0, color='blue', linestyle='--', alpha=0.5, label='Base Setpoint (6.0)')
    axes[0].set_ylabel('Output pH')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].set_title('Gain-Scheduled FRIT Performance Verification', fontsize=11, fontweight='bold')

    # Subplot 2: 酸閥開度對比 (真實 vs FRIT 模擬)
    axes[1].plot(time_axis, acid_pid_hist, color='salmon', alpha=0.5, label='Hist Acid Valve')
    axes[1].plot(time_axis, v_a_sim, color='red', linestyle='-', alpha=0.9, label='FRIT Sim Acid Valve')
    axes[1].set_ylabel('Acid Valve (%)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: 鹼閥開度對比 (真實 vs FRIT 模擬)
    axes[2].plot(time_axis, base_pid_hist, color='skyblue', alpha=0.5, label='Hist Base Valve')
    axes[2].plot(time_axis, v_b_sim, color='blue', linestyle='-', alpha=0.9, label='FRIT Sim Base Valve')
    axes[2].set_ylabel('Base Valve (%)')
    axes[2].set_xlabel('Time Steps (seconds)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
