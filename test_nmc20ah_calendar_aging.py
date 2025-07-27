#!/usr/bin/env python3
"""
测试 NMC 20Ah 电池日历老化模型
重点验证日历老化是否达到文章中的水平：600天到96%容量
"""

import numpy as np
import matplotlib.pyplot as plt
from mylib.nmc20Ah import nmc_20Ah

def test_nmc20ah_calendar_aging():
    """测试 NMC 20Ah 电池的日历老化"""
    
    # 创建电池模型
    battery = nmc_20Ah()
    
    # 仿真参数
    total_days = 1000
    steps_per_day = 1  # 每天1步，简化计算
    total_steps = total_days * steps_per_day
    
    # 日历老化测试条件（参考文章）
    temperature_C = 25  # 25°C 室温
    temperature_K = temperature_C + 273.15
    soc = 0.5  # 50% SOC
    
    print(f"=== NMC 20Ah 日历老化测试 ===")
    print(f"温度: {temperature_C}°C ({temperature_K}K)")
    print(f"SOC: {soc*100}%")
    print(f"测试天数: {total_days} 天")
    print(f"步长: {24/steps_per_day:.1f} 小时/步")
    
    # 时间序列
    time_days = np.linspace(0, total_days, total_steps + 1)
    time_seconds = time_days * 24 * 3600
    
    # 存储结果
    capacity_history = [1.0]  # 初始容量100%
    calendar_loss_history = [0.0]
    days_history = [0]
    
    # 逐步仿真
    for i in range(total_steps):
        # 当前时间步的应力条件
        dt_seconds = 24 * 3600 / steps_per_day  # 时间步长（秒）
        current_time = time_seconds[i+1]
        
        stressors = {
            't_secs': np.array([current_time]),
            'TdegK': np.array([temperature_K]),
            'soc': np.array([soc]),
            'delta_efc': np.array([0.0]),  # 纯日历老化，无循环
            'delta_ah': np.array([0.0]),   # 纯日历老化，无安时积累
            'Crate': np.array([0.0])       # 静置状态
        }
        
        # 更新电池状态
        battery.update_battery_state(stressors)
        
        # 记录结果
        current_capacity = battery.outputs['q'][-1]
        calendar_loss = battery.states['qLoss_cal'][-1]
        
        capacity_history.append(current_capacity)
        calendar_loss_history.append(calendar_loss)
        days_history.append(time_days[i+1])
        
        # 定期输出进度
        if (i+1) % 100 == 0 or time_days[i+1] in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            print(f"第 {time_days[i+1]:4.0f} 天: 容量 = {current_capacity:.3f} ({current_capacity*100:.1f}%), 日历损失 = {calendar_loss:.4f}")
    
    # 关键检查点
    print("\n=== 关键检查点 ===")
    days_check = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for day in days_check:
        if day <= total_days:
            idx = int(day * steps_per_day)
            if idx < len(capacity_history):
                cap = capacity_history[idx]
                print(f"第 {day:3d} 天: {cap:.3f} ({cap*100:.1f}%)")
    
    # 检查是否达到文章预期（600天到96%）
    day_600_idx = int(600 * steps_per_day)
    if day_600_idx < len(capacity_history):
        cap_600 = capacity_history[day_600_idx]
        print(f"\n文章对比:")
        print(f"600天容量: {cap_600:.3f} ({cap_600*100:.1f}%) - 期望: ~96%")
        if cap_600 > 0.98:
            print("⚠️  日历老化过慢，需要增加参数")
        elif cap_600 < 0.94:
            print("⚠️  日历老化过快，需要减小参数")
        else:
            print("✅ 日历老化速率合理")
    
    return days_history, capacity_history, calendar_loss_history

def plot_results(days, capacity, calendar_loss):
    """绘制结果"""
    plt.rcParams['font.size'] = 12
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 容量衰减曲线
    ax1.plot(days, np.array(capacity) * 100, 'b-', linewidth=2, label='总容量')
    ax1.axhline(y=96, color='r', linestyle='--', alpha=0.7, label='期望 600天: 96%')
    ax1.axvline(x=600, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('时间 (天)')
    ax1.set_ylabel('容量 (%)')
    ax1.set_title('NMC 20Ah 日历老化测试')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(90, 100.5)
    
    # 日历损失累积
    ax2.plot(days, np.array(calendar_loss) * 100, 'g-', linewidth=2, label='日历损失')
    ax2.set_xlabel('时间 (天)')
    ax2.set_ylabel('日历损失 (%)')
    ax2.set_title('日历老化损失累积')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1000)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行测试
    days, capacity, calendar_loss = test_nmc20ah_calendar_aging()
    
    # 绘制结果
    plot_results(days, capacity, calendar_loss)
    
    print("\n=== 测试完成 ===")
