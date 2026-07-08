import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# 1. 準備模擬數據 (實務上這會是你收集到的 sensor data)
# ==========================================
# 假設我們在時間 0~20 秒內取樣 200 點
T = np.linspace(0, 20, 200)

# 假設受控體 (Plant) 為一個未知的二階系統 P(s) = 1 / (s^2 + 2s + 1)
# 這裡只是為了「產生」實驗數據 u0, y0
s = ct.tf('s')
P = 1 / (s**2 + 2*s + 1)

# 給定一段任意激發訊號當作 u0 (控制力)
u0 = np.sin(0.5 * T) + 0.5 * np.sin(1.2 * T)
# 模擬出系統的輸出 y0
_, y0 = ct.forced_response(P, T, u0)


# ==========================================
# 2. 定義 FRIT 的期望閉迴路模型 Td
# ==========================================
# 我們希望調校後的閉迴路系統表現像是一個一階系統 1 / (0.5s + 1)
Td = 1 / (0.5 * s + 1)


# ==========================================
# 3. 定義計算 Fictitious Reference 與誤差的函數
# ==========================================
def frit_cost(rho):
    Kp, Ki = rho
    
    # 建立 PI 控制器 C(s) = (Kp*s + Ki) / s
    C = (Kp * s + Ki) / s
    
    # 計算 C 的反函數 C^-1
    # 對 PI 來說 C^-1(s) = s / (Kp*s + Ki)
    C_inv = 1 / C
    
    # --- 計算 Fictitious Reference (虛擬參考命令) ---
    # 將 u0 餵入 C^-1，計算 u_filtered = C^-1 * u0
    _, u_filt = ct.forced_response(C_inv, T, u0)
    
    # r_tilde = C^-1 * u0 + y0
    r_tilde = u_filt + y0
    
    # --- 計算期望輸出與誤差 ---
    # 將虛擬參考命令 r_tilde 餵入期望模型 Td，得到預測輸出 y_sim
    _, y_sim = ct.forced_response(Td, T, r_tilde)
    
    # 計算 FRIT 損失函數 J = Σ (y0 - y_sim)^2
    cost = np.sum((y0 - y_sim)**2)
    return cost

# ==========================================
# 4. 最佳化尋找最佳參數
# ==========================================
# 初始猜測參數 rho_init = [Kp, Ki]
rho_init = [1.0, 1.0]

# 使用 SciPy 進行最佳化，給定邊界避免 Kp, Ki 小於等於 0 導致發散
res = minimize(frit_cost, rho_init, method='L-BFGS-B', bounds=[(0.01, None), (0.01, None)])

Kp_opt, Ki_opt = res.x
print(f"最佳化結果: Kp = {Kp_opt:.4f}, Ki = {Ki_opt:.4f}")

# ==========================================
# 5. 繪圖驗證與觀察 Fictitious Reference
# ==========================================
# 用最佳參數再算一次 Fictitious Reference
C_opt = (Kp_opt * s + Ki_opt) / s
C_inv_opt = 1 / C_opt
_, u_filt_opt = ct.forced_response(C_inv_opt, T, u0)
r_tilde_opt = u_filt_opt + y0
_, y_sim_opt = ct.forced_response(Td, T, r_tilde_opt)

# 畫圖比較
plt.figure(figsize=(10, 6))

# 子圖 1: 輸出響應貼合度
plt.subplot(2, 1, 1)
plt.plot(T, y0, label='Actual Output Data ($y_0$)', lw=2)
plt.plot(T, y_sim_opt, '--', label='Ideal Output ($T_d \cdot \\tilde{r}$)', lw=2)
plt.title('FRIT Optimization Result')
plt.legend()
plt.grid()

# 子圖 2: Fictitious Reference 與實際輸出的關係
plt.subplot(2, 1, 2)
plt.plot(T, r_tilde_opt, label='Fictitious Reference ($\\tilde{r}$)', color='orange')
plt.plot(T, y0, label='Actual Output Data ($y_0$)', color='blue', alpha=0.5)
plt.title('Calculated Fictitious Reference')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
