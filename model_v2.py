import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicRTDFilter(nn.Module):
    """
    動態滯留時間分佈濾波器 (Dynamic Residence Time Distribution Filter) - Bucket 版本。
    處理聚合後的閥門訊號 (每個 bucket 紀錄該區間內的總開啟秒數)。
    """
    def __init__(self, num_buckets, bucket_size):
        super(DynamicRTDFilter, self).__init__()
        self.num_buckets = num_buckets
        self.bucket_size = bucket_size # 每個 bucket 代表的物理時間 (秒)
        
        # HyperNetwork: 根據流量 F_in 預測這 num_buckets 個區塊的延遲權重
        self.acid_delay_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_buckets),
            nn.Softmax(dim=-1) # 保證權重和為 1，質量絕對守恆
        )
        
        self.base_delay_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_buckets),
            nn.Softmax(dim=-1)
        )

    def forward(self, valve_history, current_F_in):
        """
        valve_history: (Batch, 2, num_buckets) -> 裡面裝的是 0 ~ bucket_size 的累積秒數
        current_F_in: (Batch, 1) -> 當下流量
        """
        # 取出酸鹼的累積秒數歷史序列
        acid_bucket_seq = valve_history[:, 0, :] # (Batch, num_buckets)
        base_bucket_seq = valve_history[:, 1, :] # (Batch, num_buckets)
        
        # 動態生成延遲分佈 (Attention Weights)
        acid_weights = self.acid_delay_net(current_F_in) # (Batch, num_buckets)
        base_weights = self.base_delay_net(current_F_in) # (Batch, num_buckets)
        
        # 點積計算：算出「當下這個時間步，等效釋放了幾秒的藥劑」
        effective_acid_seconds = torch.sum(acid_weights * acid_bucket_seq, dim=1, keepdim=True)
        effective_base_seconds = torch.sum(base_weights * base_bucket_seq, dim=1, keepdim=True)
        
        # 【關鍵微調】：將等效秒數除以 bucket_size，轉回 0~1 的佔空比 (Duty Cycle) alpha
        alpha_acid = effective_acid_seconds / self.bucket_size
        alpha_base = effective_base_seconds / self.bucket_size
        
        return alpha_acid, alpha_base


class GreyBoxCSTR(nn.Module):
    """
    完整的灰盒 Simulator 架構，結合 NN 與可微物理公式。
    所有未知的物理常數皆轉為可學習參數 (Learnable Parameters)。
    """
    def __init__(self, num_buckets, bucket_size, V_tank, dt):
        super(GreyBoxCSTR, self).__init__()
        # 使用更新後的 Bucket RTD 濾波器
        self.nn_block = DynamicRTDFilter(num_buckets, bucket_size)
        
        # 已知的物理常數
        self.V_tank = V_tank
        self.dt = dt
        
        # 未知的物理常數轉為可學習參數
        self.raw_max_flow_acid = nn.Parameter(torch.tensor(0.1))
        self.raw_max_flow_base = nn.Parameter(torch.tensor(0.1))
        self.raw_W_acid = nn.Parameter(torch.tensor(0.1))
        self.raw_W_base = nn.Parameter(torch.tensor(0.1))

    def get_constrained_physical_params(self):
        """利用 softplus 施加物理約束，確保物理意義合理性"""
        max_flow_acid = F.softplus(self.raw_max_flow_acid)
        max_flow_base = F.softplus(self.raw_max_flow_base)
        W_acid = F.softplus(self.raw_W_acid)         # 強酸 > 0
        W_base = -F.softplus(self.raw_W_base)        # 強鹼 < 0
        return max_flow_acid, max_flow_base, W_acid, W_base

    def implicit_physics_step(self, W_current, F_in, W_in, F_acid_eff, F_base_eff, W_acid, W_base):
        """向後歐拉法解析解"""
        F_out = F_in + F_acid_eff + F_base_eff
        numerator = W_current + (self.dt / self.V_tank) * (F_in * W_in + F_acid_eff * W_acid + F_base_eff * W_base)
        denominator = 1 + (self.dt / self.V_tank) * F_out
        W_next = numerator / denominator
        return W_next

    def w_to_ph(self, W):
        """可微當量濃度轉 pH 函數"""
        Kw = 1e-14
        H_plus = (W + torch.sqrt(W**2 + 4 * Kw)) / 2.0
        pH = -torch.log10(H_plus + 1e-12)
        return pH

    def forward(self, initial_W, F_in_seq, W_in_seq, valve_history_seq):
        """
        Rollout 運算
        注意：這裡的序列長度 Horizon 必須與你的降採樣頻率一致。
        """
        Batch_size = initial_W.shape[0]
        Horizon = F_in_seq.shape[1]
        
        W_pred_seq = []
        pH_pred_seq = []
        W_current = initial_W
        
        max_flow_acid, max_flow_base, W_acid, W_base = self.get_constrained_physical_params()
        
        for t in range(Horizon):
            F_in_t = F_in_seq[:, t, :]
            W_in_t = W_in_seq[:, t, :]
            valve_history_t = valve_history_seq[:, t, :, :] 
            
            # alpha 已經是 0~1 的有效佔空比
            alpha_acid, alpha_base = self.nn_block(valve_history_t, F_in_t)
            
            # 佔空比 * 最大瞬時流率 = 有效平均流率
            F_acid_eff = alpha_acid * max_flow_acid
            F_base_eff = alpha_base * max_flow_base
            
            W_next = self.implicit_physics_step(W_current, F_in_t, W_in_t, F_acid_eff, F_base_eff, W_acid, W_base)
            pH_next = self.w_to_ph(W_next)
            
            W_pred_seq.append(W_next)
            pH_pred_seq.append(pH_next)
            W_current = W_next
            
        return torch.cat(pH_pred_seq, dim=1)
