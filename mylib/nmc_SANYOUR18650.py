import numpy as np 
from mylib.degradation_model import BatteryDegradationModel

class nmc_SANYOUR18650(BatteryDegradationModel):
    """
    半经验电池退化模型实现，基于 Nájera et al. (2023) Table 4 中 SANYO UR18650 W 参数。
    同时考虑日历衰减和循环衰减。
    """
    def __init__(self, degradation_scalar: float = 1, label: str = "NMC SANYO UR18650W"):  
        # 内部状态
        self.states = {
            'qLoss_cal': np.array([0.]),   # 日历衰减累积
            'qLoss_cyc': np.array([0.]),   # 循环衰减累积
        }
        # 输出
        self.outputs = {
            'q': np.array([1.]),          # 相对容量
            'q_cal': np.array([1.]),      # 日历剩余容量
            'q_cyc': np.array([1.]),      # 循环剩余容量
        }
        # 应力源历史
        self.stressors = {
            'delta_t_days': np.array([np.nan]),
            't_days': np.array([0]),
            'delta_efc': np.array([np.nan]),
            'efc': np.array([0]),
            't_secs': np.array([0.]),     # 时间序列（秒）
            'TdegK': np.array([np.nan]),  # 温度（K）
            'soc': np.array([np.nan]),    # 荷电状态
            'Ua': np.array([np.nan]),     # 阳极电位
            'dod': np.array([np.nan]),    # 放电深度
            'I': np.array([np.nan]),      # 电流（A）
            'delta_ah': np.array([np.nan]),
        }
        # 速率历史
        self.rates = {
            'q_cal': np.array([np.nan]),
            'q_cyc': np.array([np.nan]),
        }
        # 实验有效范围
        self.experimental_range = {
            'temperature': [273.15, 323.15],  # K
            'soc': [0, 1],
            'crate': [0, 2],
        }
        self._degradation_scalar = degradation_scalar
        self._label = label

    @property
    def cap(self):
        return 3000  # 典型 18650 容量 mAh

    @property
    def _params_life(self):
        # 来自论文 Table 4: SANYO UR18650 W
        # 单位系统确认：
        # 温度 T：开尔文 (K)
        # 时间 t：天 (days) 
        # AH（安时）：Ah
        # C-rate：无量纲（1 C = 额定容量/小时）
        # 指数系数 h 对应开尔文；d 对应 K⁻¹；e 无量纲
        # 模型输出 Q_loss：百分比 (%)
        return {
            'a': 1.2918e-7,    # 循环衰减系数 [1/(K²·Ah)]
            'b': -7.6878e-5,   # 循环衰减系数 [1/(K·Ah)]
            'c': 0.0114,       # 循环衰减系数 [1/Ah]
            'd': -6.7149e-3,   # C-rate温度系数 [K⁻¹]
            'e': 2.3467,       # C-rate系数 [无量纲]
            'f': 154.9601,     # 日历衰减基础系数
            'g': 0.6898,       # SOC指数系数 [无量纲]
            'h': -2.9467e3,    # 温度指数系数 [K]
            'z': 0.5,          # 时间指数 [无量纲]
        }

    def update_rates(self, stressors):
        # 直接计算退化增量，不做速率归一化
        t = stressors['t_secs']
        T = stressors['TdegK']
        soc = stressors['soc']
        
        # 处理电流参数：如果不存在，则从 Crate 估算
        if 'I' in stressors and not np.isnan(stressors['I']).all():
            I = stressors['I']
        else:
            # 从 Crate 估算电流，假设电池容量为 3Ah
            Crate = stressors.get('Crate', 1.0)  # 默认 1C
            if np.isnan(Crate):
                Crate = 1.0  # 如果 Crate 是 NaN，使用默认值
            if isinstance(Crate, (int, float)):
                I = np.full_like(soc, Crate * self.cap / 1000)  # 转换为 A
            else:
                I = Crate * self.cap / 1000
        
        # 确保 Crate 存在且有效
        if 'Crate' in stressors and not np.isnan(stressors['Crate']):
            Crate = stressors['Crate']
        else:
            # 从电流估算 Crate
            if hasattr(I, '__len__'):
                Crate = np.mean(np.abs(I)) / (self.cap / 1000)
            else:
                Crate = np.abs(I) / (self.cap / 1000)
        
        # 参数
        p = self._params_life
        
        # 日历衰减增量：按论文公式积分
        # 单位确认：时间 t 使用天 (days)，温度 T 使用开尔文 (K)
        t_days = t / (24 * 3600)  # 秒 → 天转换
        
        # 计算时间增量（仅当前步长的贡献）
        if len(t_days) > 1:
            # 取最后一个时间步长
            dt_days = t_days[-1] - t_days[-2]
            t_current = t_days[-1]
        else:
            dt_days = t_days[0]  # 第一步
            t_current = t_days[0]
        
        # 确保有合理的时间值
        if t_current <= 0:
            t_current = dt_days if dt_days > 0 else 1e-6
        
        # 按论文公式计算日历老化：f * exp(g*SOC) * exp(h/T) * t^z
        # 这里计算当前时间步的日历老化增量
        T_avg = np.mean(T)
        soc_avg = np.mean(soc)
        
        cal_factor = p['f'] * np.exp(p['g'] * soc_avg) * np.exp(p['h'] / T_avg)
        
        # 日历老化增量：使用幂函数的差分形式
        if t_current > dt_days:
            # t^z - (t-dt)^z 的近似
            dq_cal = cal_factor * (t_current**p['z'] - (t_current - dt_days)**p['z']) / 100.0
        else:
            # 第一步或很小的时间步
            dq_cal = cal_factor * (t_current**p['z']) / 100.0
        
        # 循环衰减增量：瞬时损伤密度乘以安时增量
        # 处理 delta_ah：优先使用 delta_ah，如果不存在则使用 delta_efc
        if 'delta_ah' in stressors and not np.isnan(stressors['delta_ah']):
            delta_ah = stressors['delta_ah']
        else:
            # 从 delta_efc 估算 delta_ah，假设电池容量为 3Ah
            delta_efc = stressors.get('delta_efc', 0.0)
            if np.isnan(delta_efc):
                delta_efc = 0.0
            delta_ah = delta_efc * self.cap / 1000  # 转换为 Ah
        
        # 循环衰减：按论文公式 (aT^2 + bT + c) * exp((dT + e) * Crate) * ΔAh
        # 单位确认：T(K), Crate(无量纲), ΔAh(Ah), 输出为百分比
        T_avg = np.mean(T)
        damage_density = (p['a'] * T_avg**2 + p['b'] * T_avg + p['c']) * np.exp((p['d'] * T_avg + p['e']) * Crate)
        dq_cyc = damage_density * delta_ah / 100.0  # 转换为小数形式(0-1)
        
        # 确保退化值为正且合理
        dq_cal = max(0, dq_cal)
        dq_cyc = max(0, abs(dq_cyc))  # 取绝对值确保为正
        
        # 存储增量（不是速率）
        self.rates['q_cal'] = np.append(self.rates['q_cal'], dq_cal)
        self.rates['q_cyc'] = np.append(self.rates['q_cyc'], dq_cyc)

    def update_states(self, stressors):
        # 直接累加退化增量，不再乘以时间
        # 获取最新增量
        dq_cal = self.rates['q_cal'][-1] * self._degradation_scalar
        dq_cyc = self.rates['q_cyc'][-1] * self._degradation_scalar
        
        # 累计
        self.states['qLoss_cal'] = np.append(self.states['qLoss_cal'], self.states['qLoss_cal'][-1] + dq_cal)
        self.states['qLoss_cyc'] = np.append(self.states['qLoss_cyc'], self.states['qLoss_cyc'][-1] + dq_cyc)

    def update_outputs(self, stressors):
        # 基于状态计算输出
        loss_cal = self.states['qLoss_cal'][-1]
        loss_cyc = self.states['qLoss_cyc'][-1]
        q_cal = 1 - loss_cal
        q_cyc = 1 - loss_cyc
        q = min(q_cal, q_cyc)
        # 存储
        for key, val in zip(['q', 'q_cal', 'q_cyc'], [q, q_cal, q_cyc]):
            self.outputs[key] = np.append(self.outputs[key], val)