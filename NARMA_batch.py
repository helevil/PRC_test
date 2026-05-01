import os
import glob
import re  # 引入正则表达式库
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


# ================= 辅助函数：从文件名自动提取电压 =================
def extract_voltages_from_filename(filename):
    """
    智能提取电压：支持从 _15_-15_ 自动换算为 1.5V 和 -1.5V
    """
    pattern = r"_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_sqr"
    match = re.search(pattern, filename)

    if match:
        v1 = float(match.group(1))
        v2 = float(match.group(2))

        # 💡 【核心修复补丁】：
        # 如果从文件名读到的电压绝对值大于等于 5，我们就默认它是省略了小数点，自动除以 10
        # 这样 15 会自动变成 1.5，20 会变成 2.0，而原本写了 2 的就不受影响。
        if abs(v1) >= 5:
            v1 = v1 / 10.0
        if abs(v2) >= 5:
            v2 = v2 / 10.0

        v_max = max(v1, v2)
        v_min = min(v1, v2)
        return v_min, v_max
    else:
        return None, None


# ================= 核心计算与预处理函数 =================
def generate_narma_target(u, order=10):
    length = len(u)
    y = np.zeros(length)
    for t in range(order, length - 1):
        sum_y = np.sum(y[t - order + 1 : t + 1])
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum_y + 1.5 * u[t - order + 1] * u[t] + 0.1
    return y


def calculate_nmse(target, prediction):
    mse = np.mean((target - prediction) ** 2)
    var_target = np.var(target)
    if var_target == 0:
        return 0
    return mse / var_target


def load_and_preprocess_narma(filepath, v_min, v_max, node_point=50):
    df = pd.read_csv(filepath)
    time_step_total = len(df) // node_point
    data_num = time_step_total * node_point
    df = df.iloc[:data_num]

    voltage = df["voltage"].values
    current = df["current"].values

    voltage_pulse = voltage[node_point // 2 :: node_point]

    # 根据自动读取的 v_min 和 v_max 进行动态映射！
    u_signal = (voltage_pulse - v_min) / (v_max - v_min) * 0.5
    u_signal = np.clip(u_signal, 0, 0.5)

    VN = current.reshape(time_step_total, node_point)
    VN_normalized = VN / np.max(np.abs(current))

    return VN_normalized, u_signal, time_step_total


def evaluate_narma_silent(VN, u_signal, order, selected_nodes, file_prefix, output_dir, v_min, v_max, train_ratio=0.7):
    target_y = generate_narma_target(u_signal, order=order)
    states_x = VN[:, :selected_nodes]

    total_steps = len(u_signal)
    train_end = int(total_steps * train_ratio)

    X_train, Y_train = states_x[order:train_end], target_y[order:train_end]
    X_test, Y_test = states_x[train_end:], target_y[train_end:]

    b_candidates = np.logspace(-6, 2, 9)
    ridge = RidgeCV(alphas=b_candidates, cv=5, fit_intercept=True)
    ridge.fit(X_train, Y_train)

    Y_test_pred = ridge.predict(X_test)
    nmse_test = calculate_nmse(Y_test, Y_test_pred)

    plot_steps = min(100, len(Y_test))
    plt.figure(figsize=(10, 4))
    plt.plot(range(plot_steps), Y_test[:plot_steps], "o-", color="black", linewidth=2, label="Target (True)")
    plt.plot(range(plot_steps), Y_test_pred[:plot_steps], "s--", color="red", linewidth=2, alpha=0.8, label=f"Output (NMSE={nmse_test:.3f})")

    plt.xlabel("Time Step (Pulse Number)", fontsize=16)
    plt.ylabel(f"NARMA-{order} Value", fontsize=16)
    # 在标题上自动打上识别出的电压标签
    plt.title(f"NARMA-{order} ({v_min}V to {v_max}V): {file_prefix}", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_prefix}_NARMA{order}_NMSE_{nmse_test:.3f}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return round(nmse_test, 4), ridge.alpha_


# ================= 主批处理控制器 =================
def batch_process_narma_auto_voltage(input_dir, output_dir, order, selected_nodes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"❌ 在 {input_dir} 中没有找到任何 CSV 文件！")
        return

    print(f"===== 🚀 开始全自动解析批量处理 (NARMA-{order}) =====")
    summary_data = []

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        file_prefix = filename.replace(".csv", "")
        print(f"\n[{idx}/{len(csv_files)}] 处理文件: {filename}")

        try:
            # 💡 核心升级：自动读取电压
            v_min, v_max = extract_voltages_from_filename(filename)

            if v_min is None or v_max is None:
                print(f"  ⚠️ 警告: 无法从文件名识别电压，跳过此文件。请确保文件名包含类似 '_2_0_sqr' 的格式。")
                continue

            print(f"  ⚡ 成功识别电压范围: {v_min}V 到 {v_max}V", end=" | ")

            # 使用识别出的电压进行计算
            VN_norm, u_signal, total_pulses = load_and_preprocess_narma(filepath, v_min, v_max, node_point=50)
            nmse, best_b = evaluate_narma_silent(VN_norm, u_signal, order, selected_nodes, file_prefix, output_dir, v_min, v_max)

            print(f"✅ NMSE = {nmse:.4f}")

            summary_data.append({"Filename": filename, "V_Min": v_min, "V_Max": v_max, "Total_Pulses": total_pulses, "NARMA_Task": f"NARMA-{order}", "Nodes_Used": selected_nodes, "Best_Alpha(b)": best_b, "NMSE": nmse})  # 记录到总表里，方便后续用 Excel 做筛选  # 记录到总表里

        except Exception as e:
            print(f"\n  ❌ 处理出错: {e}")

    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="NMSE", ascending=True)
        csv_out_path = os.path.join(output_dir, f"00_NARMA{order}_AutoBatch_Summary.csv")
        summary_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 完美收工！自动电压总表已保存！")


# ================= 执行引擎 =================
if __name__ == "__main__":

    INPUT_DIR = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/PtNSTO_05wt_D02/"
    OUTPUT_DIR = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/NARMA_AutoBatch_Results/PtNSTO_05wt_D02/"

    NARMA_ORDER = 2
    OPTIMAL_NODES = 10

    # 启动自动解析电压的批处理
    batch_process_narma_auto_voltage(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, order=NARMA_ORDER, selected_nodes=OPTIMAL_NODES)
