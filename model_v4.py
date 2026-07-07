import numpy as np
import pandas as pd
from simple_pid import PID
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import lfilter, lfilter_zi, butter, filtfilt

# ==========================================
# 1. 您所撰寫的 FRIT 虛擬參考軌跡生成函數
# ==========================================
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

# ==========================================
# 2. 建立模擬受控系統 (FOPDT: 一階帶延遲系統)
# ==========================================
class SimulatedPlant:
    def __init__(self, K, T, L, dt):
        """ K: 增益, T: 時間常數, L: 延遲時間, dt: 取樣時間 """
        self.dt = dt
        self.alpha = np.exp(-dt / T)
        self.beta = K * (1 - self.alpha)
        self.delay_steps = max(1, int(L / dt))
        self.u_buffer = np.zeros(self.delay_steps)
        self.y = 0.0

    def update(self, u):
        # 取得延遲後的控制輸入
        u_delayed = self.u_buffer[0]
        # 更新延遲緩衝區
        self.u_buffer = np.roll(self.u_buffer, -1)
        self.u_buffer[-1] = u
        
        # 加入一點高斯白噪音模擬真實感
        noise = np.random.normal(0, 0.02)
        
        # 更新系統狀態
        self.y = self.alpha * self.y + self.beta * u_delayed + noise
        return self.y

# ==========================================
# 3. 實驗操作與 FRIT 最佳化流程
# ==========================================
def run_simulation(Kp, Ki, Kd, plant_params, sim_time, dt, setpoint):
    """ 執行一次系統模擬並回傳數據 (修正版：自定義離散 PID 確保模擬時間正確) """
    plant = SimulatedPlant(**plant_params, dt=dt)
    
    steps = int(sim_time / dt)
    t_data = np.linspace(0, sim_time, steps)
    u_data = np.zeros(steps)
    y_data = np.zeros(steps)
    
    y = 0.0
    integral = 0.0
    prev_error = setpoint - y
    
    for i in range(steps):
        # 1. 計算當前誤差
        error = setpoint - y
        
        # 2. 計算 PID 積分與微分項 (純粹依賴模擬的 dt)
        integral += error * dt
        derivative = (error - prev_error) / dt
        
        # 3. 計算控制輸出 u
        u = Kp * error + Ki * integral + Kd * derivative
        
        # 4. 將控制力送入受控體
        y = plant.update(u)
        
        # 5. 紀錄數據
        u_data[i] = u
        y_data[i] = y
        prev_error = error
        
    return t_data, u_data, y_data

def frit_cost_function(pid_params, rf, y_out, u_out, dt):
    """ FRIT 損失函數：計算使用當前 PID 參數與虛擬參考軌跡時，與實際控制力的 MSE """
    Kp, Ki, Kd = pid_params
    
    u_sim = np.zeros_like(u_out)
    integral = 0.0
    prev_error = 0.0
    
    for i in range(len(y_out)):
        error = rf[i] - y_out[i]
        integral += error * dt
        derivative = (error - prev_error) / dt if i > 0 else 0.0
        
        u_sim[i] = Kp * error + Ki * integral + Kd * derivative
        prev_error = error
        
    # 計算擬合控制力與實際控制力的均方誤差 (MSE)
    return np.mean((u_out - u_sim)**2)

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    # --- 系統與模擬參數 ---
    DT = 0.1           # 取樣時間 (sec)
    SIM_TIME = 100.0    # 模擬總時間 (sec)
    SETPOINT = 10.0    # 目標值
    
    # 建立一個真實受控體: 增益=1.5, 時間常數=2.0, 延遲=0.5
    PLANT_PARAMS = {"K": 1.5, "T": 2.0, "L": 0.5}
    DELAY_STEPS = int(PLANT_PARAMS["L"] / DT)

    # --- 步驟 A：收集初始數據 (使用一組保守/反應慢的 PID) ---
    print("Step A: 收集初始數據...")
    INITIAL_PID = (0.3, 0.1, 0.0)
    t, u_initial, y_initial = run_simulation(*INITIAL_PID, PLANT_PARAMS, SIM_TIME, DT, SETPOINT)

    # --- 步驟 B：生成 FRIT 虛擬參考軌跡 ---
    print("Step B: 計算虛擬參考軌跡...")
    # 設定我們期望的系統響應速度 (omega 越大反應越快)
    DESIRED_OMEGA = 1.5 
    rf = generate_frit_reference_with_delay(
        y_out=y_initial, 
        omega=DESIRED_OMEGA, 
        base_dt=DT, 
        delay_steps=DELAY_STEPS, 
        zeta=1.0
    )

    # --- 步驟 C：FRIT 最佳化 ---
    print("Step C: 執行差分進化算法尋找最佳 PID 參數...")
    bounds = [(0.01, 5.0), (0.01, 5.0), (0.0, 1.0)] # (Kp, Ki, Kd) 的搜尋範圍
    result = differential_evolution(
        frit_cost_function, 
        bounds, 
        args=(rf, y_initial, u_initial, DT),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-4
    )
    
    best_kp, best_ki, best_kd = result.x
    print(f"最佳化完成！\n找到的 PID 參數: Kp={best_kp:.3f}, Ki={best_ki:.3f}, Kd={best_kd:.3f}")

    # --- 步驟 D：驗證調適結果 ---
    print("Step D: 使用新的 PID 參數驗證系統...")
    _, u_tuned, y_tuned = run_simulation(best_kp, best_ki, best_kd, PLANT_PARAMS, SIM_TIME, DT, SETPOINT)

    # --- 繪圖比較 ---
    plt.figure(figsize=(12, 8))
    
    # 子圖 1: 輸出響應 (Process Variable)
    plt.subplot(2, 1, 1)
    plt.axhline(SETPOINT, color='k', linestyle='--', label='Setpoint')
    plt.plot(t, y_initial, label=f'Initial PID {INITIAL_PID}', alpha=0.7)
    plt.plot(t, y_tuned, label=f'Tuned PID ({best_kp:.2f}, {best_ki:.2f}, {best_kd:.2f})', linewidth=2)
    plt.plot(t, rf, 'r:', label='Fictitious Reference (rf)', alpha=0.6)
    plt.title('System Response (FRIT Tuning)')
    plt.ylabel('Output (y)')
    plt.legend()
    plt.grid(True)

    # 子圖 2: 控制力 (Control Output)
    plt.subplot(2, 1, 2)
    plt.plot(t, u_initial, label='Initial Control (u)', alpha=0.7)
    plt.plot(t, u_tuned, label='Tuned Control (u)', linewidth=2)
    plt.xlabel('Time (sec)')
    plt.ylabel('Control Effort (u)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
