import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicRTDFilter(nn.Module):
    """
    動態滯留時間分佈濾波器 (Dynamic Residence Time Distribution Filter)。
    專門處理 PWM (0/1) 閥門訊號。利用 HyperNetwork 動態生成 Softmax 權重，
    保證質量絕對守恆，並根據當下流速 F_in 動態調整延遲分佈。
    """
    def __init__(self, window_size):
        super(DynamicRTDFilter, self).__init__()
        self.window_size = window_size
        
        # HyperNetwork: 根據流量 F_in 預測延遲權重
        # Acid 與 Base 走各自的 MLP，因為管線長度可能不同
        self.acid_delay_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, window_size),
            nn.Softmax(dim=-1) # 關鍵：保證權重和為 1，質量絕對守恆
        )
        
        self.base_delay_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, window_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, valve_history, current_F_in):
        """
        valve_history: (Batch, 2, window_size) -> 0是酸閥, 1是鹼閥 (純 0/1 序列)
        current_F_in: (Batch, 1) -> 當下流量
        """
        # 取出酸鹼的 0/1 歷史序列
        acid_fbk_seq = valve_history[:, 0, :] # (Batch, window_size)
        base_fbk_seq = valve_history[:, 1, :] # (Batch, window_size)
        
        # 根據當下流量 F_in，動態生成延遲分佈 (Attention Weights)
        acid_weights = self.acid_delay_net(current_F_in) # (Batch, window_size)
        base_weights = self.base_delay_net(current_F_in) # (Batch, window_size)
        
        # 點積計算當下的有效連續等效係數 (0~1 之間)
        # 物理意義：這段管線裡過去 W 步打進來的 0/1 訊號，經過擴散後，現在這瞬間流進了多少
        alpha_acid = torch.sum(acid_weights * acid_fbk_seq, dim=1, keepdim=True)
        alpha_base = torch.sum(base_weights * base_fbk_seq, dim=1, keepdim=True)
        
        return alpha_acid, alpha_base

class GreyBoxCSTR(nn.Module):
    """
    完整的灰盒 Simulator 架構，結合 NN 與可微物理公式。
    所有未知的物理常數皆轉為可學習參數 (Learnable Parameters)。
    """
    def __init__(self, window_size, V_tank, dt):
        super(GreyBoxCSTR, self).__init__()
        self.nn_block = DynamicRTDFilter(window_size)
        
        # 已知的物理常數 (不可訓練)
        self.V_tank = V_tank
        self.dt = dt
        
        # 未知的物理常數轉為可學習參數 (Learnable Parameters)
        # 初始化賦予合理的猜測值 (取 log 或小數值皆可，因為後續會過 softplus)
        self.raw_max_flow_acid = nn.Parameter(torch.tensor(0.1))
        self.raw_max_flow_base = nn.Parameter(torch.tensor(0.1))
        self.raw_W_acid = nn.Parameter(torch.tensor(0.1))
        self.raw_W_base = nn.Parameter(torch.tensor(0.1)) # 預設為正數，在 forward 階段翻轉極性

    def get_constrained_physical_params(self):
        """
        利用 softplus 施加物理約束，防止模型在訓練過程中推導出負流量或極性錯誤的濃度。
        softplus(x) = ln(1 + exp(x))，保證輸出 > 0 且平滑可導。
        """
        max_flow_acid = F.softplus(self.raw_max_flow_acid)
        max_flow_base = F.softplus(self.raw_max_flow_base)
        W_acid = F.softplus(self.raw_W_acid)         # 強酸當量濃度必須 > 0
        W_base = -F.softplus(self.raw_W_base)        # 強鹼當量濃度必須 < 0
        
        return max_flow_acid, max_flow_base, W_acid, W_base

    def implicit_physics_step(self, W_current, F_in, W_in, F_acid_eff, F_base_eff, W_acid, W_base):
        """
        向後歐拉法 (Backward Euler) 的解析解，處理數值剛性。
        此處的 W_acid 與 W_base 來自模型自己訓練學習到的物理參數。
        """
        F_out = F_in + F_acid_eff + F_base_eff
        
        # 分子: 上一刻的質量 + dt * 這一刻注進來的總質量
        numerator = W_current + (self.dt / self.V_tank) * (F_in * W_in + F_acid_eff * W_acid + F_base_eff * W_base)
        # 分母: 1 + dt * (流出率)
        denominator = 1 + (self.dt / self.V_tank) * F_out
        
        W_next = numerator / denominator
        return W_next

    def w_to_ph(self, W):
        """可微的當量濃度轉 pH 函數 (解二次方程式 [H+]^2 - W[H+] - Kw = 0)"""
        Kw = 1e-14
        H_plus = (W + torch.sqrt(W**2 + 4 * Kw)) / 2.0
        # 加上極小值防止 log(0)
        pH = -torch.log10(H_plus + 1e-12)
        return pH

    def forward(self, initial_W, F_in_seq, W_in_seq, valve_history_seq):
        """
        Rollout 運算：模型在此進行多步自迴歸模擬
        initial_W: shape (Batch, 1), t=0 的初始當量濃度
        _seq: shape (Batch, Horizon, ...) 未來 H 步的外部控制與干擾序列
        """
        Batch_size = initial_W.shape[0]
        Horizon = F_in_seq.shape[1]
        
        W_pred_seq = []
        pH_pred_seq = []
        
        W_current = initial_W
        
        # 取得加上物理約束後的學習參數
        max_flow_acid, max_flow_base, W_acid, W_base = self.get_constrained_physical_params()
        
        # 進行多步預測 (Rollout)
        for t in range(Horizon):
            F_in_t = F_in_seq[:, t, :]
            W_in_t = W_in_seq[:, t, :]
            # 取得 t 時刻對應的歷史 Sliding Window
            valve_history_t = valve_history_seq[:, t, :, :] 
            
            # 1. 透過動態延遲濾波器預測有效閥門係數 (將 PWM 轉為連續有效量)
            alpha_acid, alpha_base = self.nn_block(valve_history_t, F_in_t)
            
            # 計算有效流量 (學習到的最大流量 * 等效係數)
            F_acid_eff = alpha_acid * max_flow_acid
            F_base_eff = alpha_base * max_flow_base
            
            # 2. 透過物理層更新狀態
            W_next = self.implicit_physics_step(W_current, F_in_t, W_in_t, F_acid_eff, F_base_eff, W_acid, W_base)
            
            # 3. 轉為 pH 值
            pH_next = self.w_to_ph(W_next)
            
            W_pred_seq.append(W_next)
            pH_pred_seq.append(pH_next)
            
            # 將預測的狀態作為下一時刻的輸入 (Autoregressive)
            W_current = W_next
            
        return torch.cat(pH_pred_seq, dim=1) # 回傳形狀: (Batch, Horizon)

# ---------------------------------------------------------
# 訓練迴圈中的 Loss 計算範例 (Pseudo-code)
# ---------------------------------------------------------
"""
# 初始化模型時，不需再輸入 W_acid_source, W_base_source
model = GreyBoxCSTR(window_size=300, V_tank=100.0, dt=1.0)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_data in dataloader:
        W_init = batch_data['initial_W']           # (Batch, 1)
        F_in_seq = batch_data['F_in_seq']          # (Batch, 100, 1)
        W_in_seq = batch_data['W_in_seq']          # (Batch, 100, 1)
        valve_hist = batch_data['valve_hist_seq']  # (Batch, 100, 2, 300)
        pH_true = batch_data['pH_true_seq']        # (Batch, 100)
        
        # 1. 模型 Rollout 模擬 (不用再輸入 max_valve_flow)
        pH_pred = model(W_init, F_in_seq, W_in_seq, valve_hist)
        
        # 2. 計算 Simulation Error
        loss = criterion(pH_pred, pH_true)
        
        # 3. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # (選擇性) 梯度裁剪防止物理層初期爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
    # 訓練過程中，你可以隨時印出這些參數，檢視模型學習到的物理真實數值
    if epoch % 10 == 0:
        with torch.no_grad():
            f_a, f_b, w_a, w_b = model.get_constrained_physical_params()
            print(f"Learned Acid Flow: {f_a.item():.4f}, Base Flow: {f_b.item():.4f}")
            print(f"Learned Acid Conc: {w_a.item():.4f}, Base Conc: {w_b.item():.4f}")
"""
