import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import RidgeCV

# ================= 全局图表格式设置 (学术标准) =================
font = "Times New Roman"
plt.rcParams["font.family"] = font
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = font
plt.rcParams["mathtext.it"] = font
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# ================= Part 0: 数据加载与全局归一化 =================
def load_and_preprocess_data(filepath, width=1e-3, node_point=50):
    print(f"\n[GLOBAL 模式] 正在读取数据: {filepath.split('/')[-1]}...")
    df = pd.read_csv(filepath)

    time_step_total = len(df) // node_point
    data_num = time_step_total * node_point
    df = df.iloc[:data_num]

    voltage = df["voltage"].values
    current = df["current"].values

    # 提取输入信号: 根据脉冲中间点的电压判定 (0 或 1)
    voltage_x = voltage[node_point // 2 :: node_point]
    signal_y = np.where(voltage_x > 1.0, 1, 0)

    # 重构电流矩阵并进行 GLOBAL 归一化 (保留器件电导漂移记忆)
    VN = current.reshape(time_step_total, node_point)
    global_max = np.max(np.abs(current))
    VN_normalized = VN / global_max

    return VN_normalized, signal_y, time_step_total


# ================= Part 1: Node 数量优化 (特征窗口提取) =================
def explore_node_relationship(VN, signal_y, t_delay=1, train_ratio=0.6):
    print(f"\n--- [Part 1] 特征窗口优化扫描 (T_delay = {t_delay}) ---")
    train_data_num = int(len(signal_y) * train_ratio)
    node_test_list = [2, 5, 10, 15, 20, 30, 40, 50]

    train_cc, test_cc = [], []
    b_candidates = np.logspace(-6, 2, 9)

    for nodes in node_test_list:
        states_x = VN[:, :nodes]
        # 严格的时间因果律对齐：用当前状态(states_x[t_delay:])预测过去信号(signal_y[:-t_delay])
        target_y = signal_y[:-t_delay]
        input_x = states_x[t_delay:]

        X_train, Y_train = input_x[:train_data_num], target_y[:train_data_num]
        X_test, Y_test = input_x[train_data_num:], target_y[train_data_num:]

        ridge = RidgeCV(alphas=b_candidates, cv=5, fit_intercept=True)
        ridge.fit(X_train, Y_train)

        cc_train = np.corrcoef(Y_train, ridge.predict(X_train))[0, 1]
        cc_test = np.corrcoef(Y_test, ridge.predict(X_test))[0, 1]

        train_cc.append(cc_train)
        test_cc.append(cc_test)
        print(f"  Nodes = {nodes:<2} | Train CC = {cc_train:.4f} | Test CC = {cc_test:.4f}")

    # 找出泛化性能(Test CC)最好的节点数
    best_idx = np.argmax(test_cc)
    best_nodes = node_test_list[best_idx]

    # 画图验证
    plt.figure(figsize=(6, 4.5))
    plt.plot(node_test_list, train_cc, "s--", color="navy", markersize=8, linewidth=2, label="Train CC")
    plt.plot(node_test_list, test_cc, "o-", color="firebrick", markersize=8, linewidth=2, label="Test CC")
    plt.plot(best_nodes, test_cc[best_idx], marker="*", markersize=15, color="gold", markeredgecolor="black")

    plt.xlabel("Number of Virtual Nodes", fontsize=16)
    plt.ylabel("Correlation Coefficient (CC)", fontsize=16)
    plt.xticks(node_test_list, fontsize=12)
    plt.yticks(fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.title("Feature Window Optimization", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return best_nodes


# ================= Part 2: 完整 STM 性能与总 MC 计算 =================
def evaluate_stm_capacity(VN, signal_y, selected_nodes=10, max_delay=6, train_ratio=0.6):
    print(f"\n--- [Part 2] STM 记忆性能终极评估 (锁定前 {selected_nodes} 个 Node) ---")
    train_data_num = int(len(signal_y) * train_ratio)
    states_x = VN[:, :selected_nodes]

    delay_list = list(range(1, max_delay + 1))
    CC_test_list = []
    MC_total = 0
    b_candidates = np.logspace(-6, 2, 9)

    for t_delay in delay_list:
        target_y = signal_y[:-t_delay]
        input_x = states_x[t_delay:]

        X_train, Y_train = input_x[:train_data_num], target_y[:train_data_num]
        X_test, Y_test = input_x[train_data_num:], target_y[train_data_num:]

        ridge = RidgeCV(alphas=b_candidates, cv=5, fit_intercept=True)
        ridge.fit(X_train, Y_train)

        cc_test = np.corrcoef(Y_test, ridge.predict(X_test))[0, 1]
        cc_test = max(0, cc_test) if not np.isnan(cc_test) else 0

        CC_test_list.append(cc_test)
        MC_total += cc_test**2
        print(f"  T_delay = {t_delay:<2} | Best b = {ridge.alpha_:<7.1e} | Test CC = {cc_test:.4f}")

    print(f"\n  >>> 实验结论: 最终 Memory Capacity (MC) = {MC_total:.4f} <<<")

    # 绘制经典的 STM 衰减曲线
    plt.figure(figsize=(6, 4.5))
    plt.plot(delay_list, CC_test_list, marker="o", markersize=8, color="darkgreen", linewidth=2, label=f"Total MC = {MC_total:.2f}")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")

    plt.xlabel("Delay Step ($T_{delay}$)", fontsize=16)
    plt.ylabel("Test CC", fontsize=16)
    plt.xticks(delay_list, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.1, 1.05)
    plt.title("Short-Term Memory Capacity", fontsize=16)
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    plt.show()


# ================= 启动程序 =================
if __name__ == "__main__":
    # 替换为你要分析的 CSV 文件路径
    folder_name = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/PtNSTO_05wt_D02/"
    file_name = "PtNSTO_05wt_D020_1uC_1ms_2_0_sqr_1M_stm.csv"
    filepath = folder_name + file_name

    try:
        # 1. 读取数据并全局归一化
        VN_norm, signal_y, total_pulses = load_and_preprocess_data(filepath, node_point=50)

        # 2. 自动寻找最佳截取节点数 (防止包含冗余噪声导致过拟合)
        optimal_nodes = explore_node_relationship(VN_norm, signal_y, t_delay=1, train_ratio=0.6)

        # 3. 计算真正的 MC 分数
        evaluate_stm_capacity(VN_norm, signal_y, selected_nodes=optimal_nodes, max_delay=6, train_ratio=0.6)

    except Exception as e:
        print(f"\n执行出错，请检查文件路径或数据格式。错误信息: {e}")
