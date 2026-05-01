import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

# ================= 全局图表格式 =================
font = "Times New Roman"
plt.rcParams["font.family"] = font
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = font
plt.rcParams["mathtext.it"] = font
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# ================= NARMA 目标生成函数 =================
def generate_narma_target(u, order=10):
    """
    根据输入序列 u(t) 递归生成 NARMA-n 目标序列 y(t)
    标准公式: y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i)) + 1.5*u(t-n+1)*u(t) + 0.1
    """
    length = len(u)
    y = np.zeros(length)

    # 初始的几个步长用简单的随机或0填充，因为历史数据不足
    for t in range(order, length - 1):
        sum_y = np.sum(y[t - order + 1 : t + 1])
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum_y + 1.5 * u[t - order + 1] * u[t] + 0.1

    return y


# ================= 误差评估函数 (NMSE) =================
def calculate_nmse(target, prediction):
    """计算归一化均方误差 (NMSE), 越接近 0 越好"""
    mse = np.mean((target - prediction) ** 2)
    var_target = np.var(target)
    if var_target == 0:
        return 0
    return mse / var_target


# ================= 数据预处理 =================
def load_and_preprocess_narma(filepath, v_min=1.0, v_max=2.0, node_point=50):
    print(f"\n正在读取 NARMA 数据: {filepath.split('/')[-1]}...")
    df = pd.read_csv(filepath)

    time_step_total = len(df) // node_point
    data_num = time_step_total * node_point
    df = df.iloc[:data_num]

    voltage = df["voltage"].values
    current = df["current"].values

    # 提取每个脉冲的电压幅值
    voltage_pulse = voltage[node_point // 2 :: node_point]

    # 把电压反向映射回理论的输入信号 u(t) ∈ [0, 0.5]
    u_signal = (voltage_pulse - v_min) / (v_max - v_min) * 0.5
    u_signal = np.clip(u_signal, 0, 0.5)  # 确保边界安全

    # 构建状态矩阵并进行全局归一化
    VN = current.reshape(time_step_total, node_point)
    VN_normalized = VN / np.max(np.abs(current))

    return VN_normalized, u_signal, time_step_total


# ================= 核心 NARMA 评估引擎 =================
def evaluate_narma_task(VN, u_signal, order=10, selected_nodes=10, train_ratio=0.6):
    print(f"\n--- 启动 NARMA-{order} 性能评估 (使用 {selected_nodes} 个 Node) ---")

    # 1. 生成真实的理论目标信号 y(t)
    target_y = generate_narma_target(u_signal, order=order)

    # 2. 截取器件特征状态
    states_x = VN[:, :selected_nodes]

    # 3. 划分训练集和测试集 (剔除开头数据不足的 order 步)
    total_steps = len(u_signal)
    train_end = int(total_steps * train_ratio)

    # 保证不使用前面未经完全迭代的噪声数据
    X_train = states_x[order:train_end]
    Y_train = target_y[order:train_end]

    X_test = states_x[train_end:]
    Y_test = target_y[train_end:]

    # 4. 岭回归训练
    b_candidates = np.logspace(-6, 2, 9)
    ridge = RidgeCV(alphas=b_candidates, cv=5, fit_intercept=True)
    ridge.fit(X_train, Y_train)

    # 5. 预测与评估
    Y_test_pred = ridge.predict(X_test)
    nmse_test = calculate_nmse(Y_test, Y_test_pred)

    print(f"  > 最佳正则化参数 b: {ridge.alpha_:.1e}")
    print(f"  > 测试集 NMSE: {nmse_test:.4f} (越接近0越好)")

    # 6. 绘制重构轨迹图 (这是论文里最具说服力的图)
    plot_steps = min(100, len(Y_test))  # 只取前100步画图展示细节

    plt.figure(figsize=(10, 4))
    plt.plot(range(plot_steps), Y_test[:plot_steps], "o-", color="black", linewidth=2, label="Target (True NARMA)")
    plt.plot(range(plot_steps), Y_test_pred[:plot_steps], "s--", color="red", linewidth=2, alpha=0.8, label=f"Output (NMSE = {nmse_test:.3f})")

    plt.xlabel("Time Step (Pulse Number)", fontsize=16)
    plt.ylabel(f"NARMA-{order} Value", fontsize=16)
    plt.title(f"NARMA-{order} Task Reconstruction (Test Set)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return nmse_test


# ================= 运行示例 =================
if __name__ == "__main__":
    # 注意：你需要换成你跑的新 NARMA 实验数据
    filepath = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/PtNSTO_05wt_D02/PtNSTO_05wt_D020_1uC_1ms_2_0_sqr_1M_stm.csv"

    try:
        # 这里 V_min 和 V_max 要填入你实际打脉冲时的最低和最高电压！！！
        VN_norm, u_signal, _ = load_and_preprocess_narma(filepath, v_min=0.0, v_max=2.0, node_point=10)

        # 你可以像上一个代码一样写个循环扫最佳节点数，为了演示，直接设定为之前跑出来的 10
        optimal_nodes = 10

        # 挑战 NARMA-10
        nmse_result = evaluate_narma_task(VN_norm, u_signal, order=2, selected_nodes=optimal_nodes, train_ratio=0.7)

    except Exception as e:
        print(f"出错: {e}")
