import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import RidgeCV

# ================= Global Plot Format Settings =================
font = "Times New Roman"
plt.rcParams["font.family"] = font
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = font
plt.rcParams["mathtext.it"] = font
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# ================= Part 0: Data Loading and Global Normalization =================
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


# ================= Part 1: Node Count Optimization =================
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

    # Draw and save the plot (no pop-up window)
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

    # Auto-save the plot
    save_path = os.path.join(output_dir, f"{file_prefix}_01_Node_Opt.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Release memory

    return best_nodes


# ================= Part 2: Complete STM Performance and Total MC Calculation =================
def evaluate_stm_capacity(VN, signal_y, best_nodes, file_prefix, output_dir, max_delay=6, train_ratio=0.6):
    train_data_num = int(len(signal_y) * train_ratio)
    states_x = VN[:, :best_nodes]

    delay_list = list(range(1, max_delay + 1))
    CC_test_list = []
    MC_total = 0
    b_candidates = np.logspace(-6, 2, 9)

    # Record CC scores for each delay
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

    # Draw and save the plot
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
    plt.close()  # Release memory

    return round(MC_total, 4), delay_results_dict


# ================= Core! Batch Processing Engine =================
def batch_process_folder(input_folder, output_folder, node_point=50, max_delay=6):
    # Ensure output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all csv files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if len(csv_files) == 0:
        print(f"No CSV files found in {input_folder}!")
        return

    print(f"===== 🚀 Starting batch processing, found {len(csv_files)} files =====")

    # Prepare an empty list to collect data from all files
    summary_data_list = []

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        file_prefix = filename.replace(".csv", "")  # Remove suffix for chart naming
        print(f"[{idx}/{len(csv_files)}] Processing: {filename} ...")

        try:
            # 1. Load and perform global normalization
            VN_norm, signal_y, total_pulses = load_and_preprocess_data(filepath, node_point)

            # 2. Find optimal nodes and save plots to output_folder
            optimal_nodes = explore_node_relationship(VN_norm, signal_y, file_prefix, output_folder)

            # 3. Get MC and delay scores, save memory decay plots
            mc_total, delay_results = evaluate_stm_capacity(VN_norm, signal_y, optimal_nodes, file_prefix, output_folder, max_delay)

            # 4. Pack this row of data into a dictionary
            row_data = {"Filename": filename, "Total_Pulses": total_pulses, "Optimal_Nodes": optimal_nodes, "Total_MC": mc_total}
            # Append each Delay score to the dictionary
            row_data.update(delay_results)

            # Add to summary list
            summary_data_list.append(row_data)

        except Exception as e:
            print(f"  ❌ Error processing file {filename}: {e}")

    print("==================================================")
    print("Data processing complete! Generating summary report...")

    # Convert all collected data to pandas DataFrame and export as CSV summary table
    summary_df = pd.DataFrame(summary_data_list)
    summary_csv_path = os.path.join(output_folder, "00_Batch_Summary_Report.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print(f"🎉 Batch processing complete! Summary report and plots saved to:\n -> {output_folder}")


# ================= Launch Program =================
if __name__ == "__main__":

    # 🔴 Step 1: Enter the path of the folder containing all CSV files to process
    # Note: Ensure this folder contains only the PRC test data you want to process
    INPUT_DIR = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/PtNSTO_005wt_D14/"

    # 🔴 Step 2: Create a new folder path where you want to save results
    # It's recommended to create a separate Results folder instead of mixing with input folder
    OUTPUT_DIR = "/Users/yumengzheng/Desktop/DATA/6.STO界面/learning/STM/zheng/Batch_Results/PtNSTO_005wt_D14_2/"

    # Start batch processing (default node_point=50, max_delay=6, modifiable as needed)
    batch_process_folder(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR, node_point=50, max_delay=6)
