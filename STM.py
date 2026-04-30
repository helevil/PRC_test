import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import RidgeCV

# ================= 全局图表格式设置 =================
font = "Times New Roman"
plt.rcParams["font.family"] = font
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = font
plt.rcParams["mathtext.it"] = font
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# ================= Part 0: 数据加载与预处理 =================
def load_and_preprocess_data(filepath, width=1e-3, node_point=50, normalize_mode="per_pulse", show_plot=True):
    print(f"\n[{normalize_mode.upper()} 模式] 正在读取数据: {filepath.split('/')[-1]}...")
    df = pd.read_csv(filepath)

    total_rows = len(df)
    time_step_total = total_rows // node_point
    data_num = time_step_total * node_point

    df = df.iloc[:data_num]
    voltage = df["voltage"].values
    current = df["current"].values
    time_data = df["time"].values

    voltage_x = voltage[node_point // 2 :: node_point]
    signal_y = np.where(voltage_x > 1.0, 1, 0)

    VN = current.reshape(time_step_total, node_point)
    VN_normalized = np.zeros_like(VN, dtype=float)

    if normalize_mode == "per_pulse":
        for i in range(time_step_total):
            pulse_max = np.max(np.abs(VN[i]))
            if pulse_max > 0:
                VN_normalized[i] = VN[i] / pulse_max
            else:
                VN_normalized[i] = VN[i]
    elif normalize_mode == "global":
        global_max = np.max(np.abs(current))
        VN_normalized = VN / global_max

    if show_plot:
        plot_pulses = min(8, time_step_total)
        plot_points = plot_pulses * node_point
        t_plot = time_data[:plot_points]
        v_plot = voltage[:plot_points]
        c_plot = VN_normalized.ravel()[:plot_points]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        color_v = "tab:blue"
        ax1.set_xlabel("Time (s)", fontsize=16)
        ax1.set_ylabel("Input Voltage (V)", color=color_v, fontsize=16)
        ax1.plot(t_plot, v_plot, color=color_v, linewidth=2, label="Voltage")
        ax1.tick_params(axis="y", labelcolor=color_v, labelsize=14)
        ax1.tick_params(axis="x", labelsize=14)
        ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))

        ax2 = ax1.twinx()
        color_c = "tab:red"
        ax2.set_ylabel(f"Normalized Current ({normalize_mode})", color=color_c, fontsize=16)
        ax2.plot(t_plot, c_plot, color=color_c, linewidth=2, label="Current")
        ax2.tick_params(axis="y", labelcolor=color_c, labelsize=14)

        plt.title("Device Physical Response", fontsize=16, pad=15)
        fig.tight_layout()
        plt.show()

    return VN_normalized, signal_y, time_step_total


# ================= Part 1: Node 数量优化 =================
def explore_node_relationship(VN, signal_y, t_delay=1, train_ratio=0.6):
    print(f"\n--- [Part 1] Node 数量优化 (T_delay = {t_delay}) ---")
    time_step_total = len(signal_y)
    train_data_num = int(time_step_total * train_ratio)

    node_test_list = [2, 5, 10, 15, 20, 30, 40, 50]
    train_cc_results, test_cc_results = [], []
    b_candidates = np.logspace(-6, 2, 9)

    for nodes in node_test_list:
        states_x = VN[:, :nodes]

        # 【致命错误已修复】：正确的STM时间对齐，用当前状态预测过去的信号
        target_y = signal_y[:-t_delay]
        input_x = states_x[t_delay:]

        X_train, Y_train = input_x[:train_data_num], target_y[:train_data_num]
        X_test, Y_test = input_x[train_data_num:], target_y[train_data_num:]

        ridge = RidgeCV(alphas=b_candidates, cv=5, fit_intercept=True)
        ridge.fit(X_train, Y_train)

        Y_train_pred = ridge.predict(X_train)
        Y_test_pred = ridge.predict(X_test)

        cc_train = np.corrcoef(Y_train, Y_train_pred)[0, 1]
        cc_test = np.corrcoef(Y_test, Y_test_pred)[0, 1]

        train_cc_results.append(cc_train)
        test_cc_results.append(cc_test)
        print(f"  Nodes = {nodes:<2} | Train CC = {cc_train:.4f} | Test CC = {cc_test:.4f}")

    plt.figure(figsize=(6, 4.5))
    plt.plot(node_test_list, train_cc_results, "s--", color="navy", markersize=8, linewidth=2, label="Train CC")
    plt.plot(node_test_list, test_cc_results, "o-", color="firebrick", markersize=8, linewidth=2, label="Test CC")

    best_idx = np.argmax(test_cc_results)
    best_nodes = node_test_list[best_idx]
    best_cc = test_cc_results[best_idx]

    plt.plot(best_nodes, best_cc, marker="*", markersize=15, color="gold", markeredgecolor="black")
    plt.annotate(f"Optimal = {best_nodes}\nCC = {best_cc:.2f}", xy=(best_nodes, best_cc), xytext=(best_nodes + 5, best_cc - 0.15), arrowprops=dict(facecolor="black", arrowstyle="->"), fontsize=13)

    plt.xlabel("Number of Virtual Nodes", fontsize=16)
    plt.ylabel("Correlation Coefficient", fontsize=16)
    plt.xticks(node_test_list, fontsize=12)
    plt.yticks(fontsize=14)
    plt.ylim(-0.1, 1.05)
    plt.title("Feature Extraction Optimization", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return best_nodes


# ================= Part 2: 完整 STM 性能计算 =================
def evaluate_stm_capacity(VN, signal_y, selected_nodes=10, max_delay=10, train_ratio=0.6):
    print(f"\n--- [Part 2] STM 记忆性能分析 (使用前 {selected_nodes} 个 Node) ---")
    time_step_total = len(signal_y)
    train_data_num = int(time_step_total * train_ratio)
    states_x = VN[:, :selected_nodes]

    delay_list = list(range(1, max_delay + 1))
    CC_test_list = []
    MC_total = 0
    b_candidates = np.logspace(-6, 2, 9)

    for t_delay in delay_list:

        # 【致命错误已修复】：正确的STM时间对齐，用当前状态预测过去的信号
        target_y = signal_y[:-t_delay]
        input_x = states_x[t_delay:]

        X_train, Y_train = input_x[:train_data_num], target_y[:train_data_num]
        X_test, Y_test = input_x[train_data_num:], target_y[train_data_num:]

        ridge = RidgeCV(alphas=b_candidates, cv=5, fit_intercept=True)
        ridge.fit(X_train, Y_train)
        Y_test_pred = ridge.predict(X_test)

        cc_test = np.corrcoef(Y_test, Y_test_pred)[0, 1]
        cc_test = max(0, cc_test) if not np.isnan(cc_test) else 0

        CC_test_list.append(cc_test)
        MC_total += cc_test**2
        print(f"  T_delay = {t_delay:<2} | Best b = {ridge.alpha_:<7.1e} | Test CC = {cc_test:.4f}")

    print(f"\n  >>> 最终 Memory Capacity (MC) = {MC_total:.4f}")

    plt.figure(figsize=(6, 4.5))
    plt.plot(delay_list, CC_test_list, marker="o", markersize=8, color="darkgreen", linewidth=2, label=f"Test CC (MC = {MC_total:.2f})")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")

    plt.xlabel("Delay Step ($t_{delay}$)", fontsize=16)
    plt.ylabel("Correlation Coefficient (CC)", fontsize=16)
    plt.xticks(delay_list, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.1, 1.05)
    plt.title("Short-Term Memory (STM) Capacity", fontsize=16)
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    plt.show()


# ================= 执行引擎 =================
if __name__ == "__main__":
    folder_name = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/PtNSTO_05wt_D02/"
    file_name = "PtNSTO_05wt_D020_1uC_1ms_2_0_sqr_1M_stm.csv"
    filepath = folder_name + file_name

    try:
        # 你可以尝试修改为 normalize_mode='global' or "per_pulse" 看哪种物理意义更符合你的器件
        VN_norm, signal_y, total_pulses = load_and_preprocess_data(filepath, width=1e-3, node_point=50, normalize_mode="global", show_plot=True)

        optimal_nodes = explore_node_relationship(VN_norm, signal_y, t_delay=1, train_ratio=0.6)

        evaluate_stm_capacity(VN_norm, signal_y, selected_nodes=optimal_nodes, max_delay=6, train_ratio=0.6)

    except Exception as e:
        print(f"出错: {e}")
