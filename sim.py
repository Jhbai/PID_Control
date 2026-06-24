import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from greybox_cstr_simulator import GreyBoxCSTR # 匯入我們剛建立的灰盒模型

class PIDController:
    """標準的 PID 控制器，包含積分抗飽和 (Anti-windup) 機制"""
    def __init__(self, kp, ki, kd, cycle_time, out_min=-100.0, out_max=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.cycle_time = cycle_time
        
        self.out_min = out_min
        self.out_max = out_max
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, current_value):
        # 誤差定義：設定值 - 當前值
        # 假設 Setpoint=7, Current=5, Error=+2 -> 需要加鹼來提升 pH -> 輸出正值
        error = setpoint - current_value
        
        # P 項
        p_term = self.kp * error
        
        # I 項 (乘上 cycle_time 作為積分時間步長)
        self.integral += error * self.cycle_time
        i_term = self.ki * self.integral
        
        # D 項
        d_term = self.kd * (error - self.prev_error) / self.cycle_time
        self.prev_error = error
        
        # 總輸出 MV
        mv = p_term + i_term + d_term
        
        # 輸出限制 (Clamp) 與積分抗飽和 (Anti-windup)
        if mv > self.out_max:
            mv = self.out_max
            self.integral -= error * self.cycle_time # 退回積分
        elif mv < self.out_min:
            mv = self.out_min
            self.integral -= error * self.cycle_time
            
        return mv

def generate_pwm_signals(mv_percent, cycle_time):
    """
    將 PID 的百分比輸出轉換為 0/1 的 FBK 序列。
    邏輯： Output * cycle_time -> 取下整數秒數 -> 轉換為 0/1 陣列
    """
    # 限制輸入在 -100 到 100 之間
    mv_percent = max(min(mv_percent, 100.0), -100.0)
    
    # 計算作動秒數 (取下整數)
    active_seconds = int((abs(mv_percent) / 100.0) * cycle_time)
    
    # 初始化全 0 的序列
    acid_pwm = np.zeros(cycle_time, dtype=np.float32)
    base_pwm = np.zeros(cycle_time, dtype=np.float32)
    
    if active_seconds > 0:
        if mv_percent < 0:
            # MV 為負：需要降低 pH，啟動酸閥
            acid_pwm[:active_seconds] = 1.0
        elif mv_percent > 0:
            # MV 為正：需要提升 pH，啟動鹼閥
            base_pwm[:active_seconds] = 1.0
            
    return acid_pwm, base_pwm

class CSTREnvironment:
    """模擬器環境封裝，負責維護狀態與歷史特徵佇列"""
    def __init__(self, model, window_size, initial_pH):
        self.model = model
        self.model.eval() # 模擬時使用 Eval 模式
        self.window_size = window_size
        
        # 初始化物理狀態
        Kw = 1e-14
        H_plus = 10**(-initial_pH)
        self.current_W = torch.tensor([[H_plus - Kw / H_plus]], dtype=torch.float32)
        
        # 初始化歷史閥門佇列 (全關閉狀態)
        self.acid_history = deque([0.0]*window_size, maxlen=window_size)
        self.base_history = deque([0.0]*window_size, maxlen=window_size)

    def step(self, acid_fbk, base_fbk, current_F_in, current_W_in):
        """執行單秒的模擬步進"""
        # 1. 更新歷史佇列
        self.acid_history.append(acid_fbk)
        self.base_history.append(base_fbk)
        
        # 2. 轉換為 Model 預期的 Tensor 形狀: (Batch=1, 2, window_size)
        acid_tensor = torch.tensor(self.acid_history, dtype=torch.float32).view(1, 1, -1)
        base_tensor = torch.tensor(self.base_history, dtype=torch.float32).view(1, 1, -1)
        valve_history_t = torch.cat([acid_tensor, base_tensor], dim=1)
        
        F_in_t = torch.tensor([[current_F_in]], dtype=torch.float32)
        W_in_t = torch.tensor([[current_W_in]], dtype=torch.float32)
        
        with torch.no_grad():
            # 取得物理約束參數
            max_flow_acid, max_flow_base, W_acid, W_base = self.model.get_constrained_physical_params()
            
            # 透過 NN 取得有效流量權重
            alpha_acid, alpha_base = self.model.nn_block(valve_history_t, F_in_t)
            
            F_acid_eff = alpha_acid * max_flow_acid
            F_base_eff = alpha_base * max_flow_base
            
            # 執行隱式物理積分
            self.current_W = self.model.implicit_physics_step(
                self.current_W, F_in_t, W_in_t, F_acid_eff, F_base_eff, W_acid, W_base
            )
            
            # 轉換為 pH
            current_pH = self.model.w_to_ph(self.current_W)
            
        return current_pH.item()

def run_simulation():
    # --- 參數設定 ---
    WINDOW_SIZE = 300      # 歷史觀察窗 (秒)
    CYCLE_TIME = 10        # PID 計算週期與 PWM 週期 (秒)
    SIMULATION_TIME = 3600 # 總模擬時間 1 小時 (秒)
    SETPOINT = 7.0         # 目標 pH 值
    
    # 建立模型與環境 (這裡使用未訓練的初始權重，你之後可載入訓練好的 .pt)
    model = GreyBoxCSTR(window_size=WINDOW_SIZE, V_tank=100.0, dt=1.0)
    env = CSTREnvironment(model, window_size=WINDOW_SIZE, initial_pH=5.0)
    
    # 建立 PID 控制器
    pid = PIDController(kp=15.0, ki=0.5, kd=2.0, cycle_time=CYCLE_TIME)
    
    # 干擾設定 (流量與濃度)
    F_in_constant = 2.0
    W_in_constant = 10**(-6.0) - 10**(-8.0) # 假設原水是微酸性 (pH=6)
    
    # --- 紀錄用的列表 ---
    log_time = []
    log_pH = []
    log_mv = []
    log_acid_fbk = []
    log_base_fbk = []
    
    current_pH = 5.0
    
    print("開始閉迴路控制模擬...")
    
    # --- 主迴圈 ---
    # 外層迴圈：以 Cycle Time 為單位執行 PID
    for cycle_idx in range(SIMULATION_TIME // CYCLE_TIME):
        # 1. 讀取當下 pH，計算 PID 輸出
        mv = pid.compute(setpoint=SETPOINT, current_value=current_pH)
        
        # 2. 將 PID 輸出轉換為 0/1 的 PWM 陣列
        acid_seq, base_seq = generate_pwm_signals(mv, CYCLE_TIME)
        
        # 內層迴圈：將 PWM 序列逐秒餵給 CSTR 執行物理模擬
        for step_in_cycle in range(CYCLE_TIME):
            global_sec = cycle_idx * CYCLE_TIME + step_in_cycle
            
            # 加入微小的流量雜訊模擬真實世界
            noisy_F_in = F_in_constant + np.random.normal(0, 0.05)
            
            # 執行單秒模擬
            current_pH = env.step(
                acid_fbk=acid_seq[step_in_cycle], 
                base_fbk=base_seq[step_in_cycle], 
                current_F_in=noisy_F_in, 
                current_W_in=W_in_constant
            )
            
            # 紀錄數據
            log_time.append(global_sec)
            log_pH.append(current_pH)
            log_mv.append(mv)
            log_acid_fbk.append(acid_seq[step_in_cycle])
            log_base_fbk.append(base_seq[step_in_cycle])

    print("模擬完成，開始繪圖...")
    
    # --- 繪圖 (仿造你提供的介面風格) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. pH 響應圖
    axes[0].plot(log_time, log_pH, color='purple', label='Actual Output pH')
    axes[0].axhline(SETPOINT, color='red', linestyle='--', label='Target pH')
    axes[0].axhline(SETPOINT + 0.5, color='blue', linestyle=':', label='Safe Zone Max')
    axes[0].axhline(SETPOINT - 0.5, color='blue', linestyle=':', label='Safe Zone Min')
    axes[0].axhspan(SETPOINT - 0.5, SETPOINT + 0.5, facecolor='green', alpha=0.1)
    axes[0].set_ylabel('pH Value')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. PID 連續輸出百分比
    axes[1].plot(log_time, log_mv, color='blue', label='PID MV (%)')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_ylim(-105, 105)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 閥門真實開關動作 (State 0/1)
    # 將酸鹼閥畫在不同高度以利觀察
    acid_plot = np.array(log_acid_fbk) * 0.8
    base_plot = np.array(log_base_fbk) * -0.8
    axes[2].step(log_time, acid_plot, color='red', label='Acid Valve FBK (0/1)', where='post')
    axes[2].step(log_time, base_plot, color='blue', label='Base Valve FBK (0/1)', where='post')
    axes[2].set_ylabel('Valve Action')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_yticks([-0.8, 0, 0.8])
    axes[2].set_yticklabels(['Base ON', 'OFF', 'Acid ON'])
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
