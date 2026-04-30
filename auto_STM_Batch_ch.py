import os
import glob
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


# ================= Part 0: 数据加载与全局归一化 =================
def load_and_preprocess_data(filepath, node_point=50):
    df = pd.read_csv(filepath)
    time_step_total = len(df) // node_point
    data_num = time_step_total * node_point
    df = df.iloc[:data_num]

    voltage = df["voltage"].values
    current = df["current"].values

    voltage_x = voltage[node_point // 2 :: node_point]
    signal_y = np.where(voltage_x > 1.0, 1, 0)

    VN = current.reshape(time_step_total, node_point)
    global_max = np.max(np.abs(current))
    VN_normalized = VN / global_max

    return VN_normalized, signal_y, time_step_total


# ================= Part 1: Node 数量优化 =================
def explore_node_relationship(VN, signal_y, file_prefix, output_dir, t_delay=1, train_ratio=0.6):
    train_data_num = int(len(signal_y) * train_ratio)
    node_test_list = [2, 5, 10, 15, 20, 30, 40, 50]

    train_cc, test_cc = [], []
    b_candidates = np.logspace(-6, 2, 9)

    for nodes in node_test_list:
        states_x = VN[:, :nodes]
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

    best_idx = np.argmax(test_cc)
    best_nodes = node_test_list[best_idx]

    # 画图并保存 (不弹出窗口)
    plt.figure(figsize=(6, 4.5))
    plt.plot(node_test_list, train_cc, "s--", color="navy", markersize=8, linewidth=2, label="Train CC")
    plt.plot(node_test_list, test_cc, "o-", color="firebrick", markersize=8, linewidth=2, label="Test CC")
    plt.plot(best_nodes, test_cc[best_idx], marker="*", markersize=15, color="gold", markeredgecolor="black")

    plt.xlabel("Number of Virtual Nodes", fontsize=16)
    plt.ylabel("Correlation Coefficient (CC)", fontsize=16)
    plt.ylim(0.0, 1.05)
    plt.title(f"Feature Optimization ({file_prefix})", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    # 自动保存图表
    save_path = os.path.join(output_dir, f"{file_prefix}_01_Node_Opt.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 释放内存

    return best_nodes


# ================= Part 2: 完整 STM 性能与总 MC 计算 =================
def evaluate_stm_capacity(VN, signal_y, best_nodes, file_prefix, output_dir, max_delay=6, train_ratio=0.6):
    train_data_num = int(len(signal_y) * train_ratio)
    states_x = VN[:, :best_nodes]

    delay_list = list(range(1, max_delay + 1))
    CC_test_list = []
    MC_total = 0
    b_candidates = np.logspace(-6, 2, 9)

    # 用于记录各个 delay 的 CC 成绩
    delay_results_dict = {}

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
        delay_results_dict[f"Delay_{t_delay}_CC"] = round(cc_test, 4)

    # 画图并保存
    plt.figure(figsize=(6, 4.5))
    plt.plot(delay_list, CC_test_list, marker="o", markersize=8, color="darkgreen", linewidth=2, label=f"Total MC = {MC_total:.2f}")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")

    plt.xlabel("Delay Step ($T_{delay}$)", fontsize=16)
    plt.ylabel("Test CC", fontsize=16)
    plt.xticks(delay_list, fontsize=14)
    plt.ylim(-0.1, 1.05)
    plt.title(f"STM Capacity ({file_prefix})", fontsize=14)
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_prefix}_02_STM_Curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 释放内存

    return round(MC_total, 4), delay_results_dict


# ================= 核心！批量处理引擎 =================
def batch_process_folder(input_folder, output_folder, node_point=50, max_delay=6):
    # 确保输出文件夹存在，如果不存在则自动创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹下所有的 csv 文件
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if len(csv_files) == 0:
        print(f"在 {input_folder} 中没有找到任何 CSV 文件！")
        return

    print(f"===== 🚀 开始批量处理，共找到 {len(csv_files)} 个文件 =====")

    # 准备一个空列表，用来收集所有文件的数据
    summary_data_list = []

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        file_prefix = filename.replace(".csv", "")  # 去掉后缀用于图表命名
        print(f"[{idx}/{len(csv_files)}] 正在处理: {filename} ...")

        try:
            # 1. 读取并全局归一化
            VN_norm, signal_y, total_pulses = load_and_preprocess_data(filepath, node_point)

            # 2. 获取最佳节点，同时把图保存到 output_folder
            optimal_nodes = explore_node_relationship(VN_norm, signal_y, file_prefix, output_folder)

            # 3. 获取 MC 和各 delay 分数，同时保存记忆衰减图
            mc_total, delay_results = evaluate_stm_capacity(VN_norm, signal_y, optimal_nodes, file_prefix, output_folder, max_delay)

            # 4. 把这一行数据打包成字典
            row_data = {"Filename": filename, "Total_Pulses": total_pulses, "Optimal_Nodes": optimal_nodes, "Total_MC": mc_total}
            # 把各个 Delay 的成绩拼接到字典后面
            row_data.update(delay_results)

            # 加入总汇列表
            summary_data_list.append(row_data)

        except Exception as e:
            print(f"  ❌ 处理文件 {filename} 时出错: {e}")

    print("==================================================")
    print("数据处理完毕！正在生成汇总报告...")

    # 将收集到的所有数据转成 pandas 的 DataFrame，并导出为 CSV 总表
    summary_df = pd.DataFrame(summary_data_list)
    summary_csv_path = os.path.join(output_folder, "00_Batch_Summary_Report.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print(f"🎉 批量任务完成！汇总报告及图表已保存至:\n -> {output_folder}")


# ================= 启动程序 =================
if __name__ == "__main__":

    # 🔴 第一步：把包含你要处理的所有 CSV 的文件夹路径写在这里
    # 注意：确保这个文件夹里全是你想处理的 PRC 测试数据
    INPUT_DIR = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/PtNSTO_05wt_D02/"

    # 🔴 第二步：创建一个你想要保存结果的新文件夹路径
    # 建议不要和输入文件夹混在一起，专门建个 Results 文件夹
    OUTPUT_DIR = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/Batch_Results/"

    # 启动批量处理 (默认 node_point=50, max_delay=6，可按需修改)
    batch_process_folder(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR, node_point=50, max_delay=6)
