import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def generate_narma_input(length=2500, u_min=0.0, u_max=0.5, seed=42):
    """Generate a random NARMA input sequence."""
    rng = np.random.default_rng(seed)
    return rng.uniform(u_min, u_max, length)


def generate_narma_output(u, order=10):
    """Generate the NARMA output sequence from input u."""
    length = len(u)
    y = np.zeros(length)

    if order == 10:
        for k in range(order, length - 1):
            y[k + 1] = 0.3 * y[k] + 0.05 * y[k] * np.sum(y[k - order + 1 : k + 1]) + 1.5 * u[k - order + 1] * u[k] + 0.1
    elif order == 20:
        for k in range(order, length - 1):
            y[k + 1] = 0.2 * y[k] + 0.004 * y[k] * np.sum(y[k - order + 1 : k + 1]) + 1.5 * u[k - order + 1] * u[k] + 0.001
    else:
        raise ValueError("Only order 10 or 20 are supported for the standard NARMA benchmark.")

    return y


def load_prc_data(
    filepath,
    node_point=50,
    normalize_mode="global",
    baseline=0.0,
    scale=1.0,
    show_plot=False,
):
    """Load physical PRC measurement CSV and return normalized states plus the encoded input u."""
    df = pd.read_csv(filepath)
    total_rows = len(df)
    if total_rows % node_point != 0:
        print(f"Warning: total rows {total_rows} not divisible by node_point {node_point}. Truncating to full pulses.")
    time_step_total = total_rows // node_point
    df = df.iloc[: time_step_total * node_point]

    voltage = df["voltage"].values
    current = df["current"].values
    time_data = df["time"].values if "time" in df.columns else np.arange(len(voltage))

    voltage_x = voltage[node_point // 2 :: node_point]
    u = (voltage_x - baseline) / scale
    u = np.clip(u, 0.0, 1.0)

    VN = current.reshape(time_step_total, node_point)
    VN_normalized = np.zeros_like(VN, dtype=float)

    if normalize_mode == "per_pulse":
        for i in range(time_step_total):
            pulse_max = np.max(np.abs(VN[i]))
            VN_normalized[i] = VN[i] / pulse_max if pulse_max > 0 else VN[i]
    elif normalize_mode == "global":
        global_max = np.max(np.abs(current))
        VN_normalized = VN / global_max
    else:
        raise ValueError("normalize_mode must be 'global' or 'per_pulse'.")

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

        ax2 = ax1.twinx()
        color_c = "tab:red"
        ax2.set_ylabel(f"Normalized Current ({normalize_mode})", color=color_c, fontsize=16)
        ax2.plot(t_plot, c_plot, color=color_c, linewidth=2, label="Current")
        ax2.tick_params(axis="y", labelcolor=color_c, labelsize=14)

        plt.title("Physical PRC Response", fontsize=16, pad=15)
        fig.tight_layout()
        plt.show()

    return VN_normalized, u, time_step_total


def evaluate_narma(VN, y_target, selected_nodes=20, train_ratio=0.6, alphas=None):
    """Evaluate a NARMA readout on reservoir states VN."""
    if alphas is None:
        alphas = np.logspace(-6, 2, 9)

    num_steps = min(len(VN), len(y_target))
    X = VN[:num_steps, :selected_nodes]
    Y = y_target[:num_steps]

    train_size = int(num_steps * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    model = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    cc = np.corrcoef(Y_test, Y_pred)[0, 1]
    mse = np.mean((Y_test - Y_pred) ** 2)
    nrmse = np.sqrt(mse) / np.std(Y_test)

    return {
        "alpha": model.alpha_,
        "cc": cc,
        "nrmse": nrmse,
        "mse": mse,
        "y_true": Y_test,
        "y_pred": Y_pred,
    }


def plot_narma_results(y_true, y_pred, order, selected_nodes, cc, nrmse):
    plt.figure(figsize=(8, 4.5))
    plt.plot(y_true, color="navy", linewidth=1.5, label="True Output")
    plt.plot(y_pred, color="orangered", linewidth=1.5, alpha=0.8, label="Predicted Output")
    plt.xlabel("Time Step", fontsize=16)
    plt.ylabel("NARMA Output", fontsize=16)
    plt.title(f"NARMA{order} Task: {selected_nodes} Nodes, CC={cc:.4f}, NRMSE={nrmse:.4f}", fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def run_narma_synthetic(order=10, length=2500, selected_nodes=30, train_ratio=0.6, seed=42, show_plot=False):
    print(f"\n--- Synthetic NARMA{order} benchmark with {selected_nodes} nodes ---")
    u = generate_narma_input(length=length, seed=seed)
    y = generate_narma_output(u, order=order)
    VN = build_random_reservoir_states(u, num_nodes=max(selected_nodes, 50), seed=seed)
    results = evaluate_narma(VN, y, selected_nodes=selected_nodes, train_ratio=train_ratio)
    print(f"Best alpha = {results['alpha']:.2e}")
    print(f"Test CC = {results['cc']:.4f}")
    print(f"Test NRMSE = {results['nrmse']:.4f}")
    if show_plot:
        plot_narma_results(results["y_true"], results["y_pred"], order, selected_nodes, results["cc"], results["nrmse"])


def run_narma_prc(
    filepath,
    order=10,
    selected_nodes=30,
    train_ratio=0.6,
    node_point=50,
    normalize_mode="global",
    baseline=0.0,
    scale=1.0,
    show_plot=False,
):
    print(f"\n--- PRC NARMA{order} benchmark using physical response ---")
    VN, u, _ = load_prc_data(
        filepath,
        node_point=node_point,
        normalize_mode=normalize_mode,
        baseline=baseline,
        scale=scale,
        show_plot=show_plot,
    )
    y = generate_narma_output(u, order=order)
    results = evaluate_narma(VN, y, selected_nodes=selected_nodes, train_ratio=train_ratio)
    print(f"Best alpha = {results['alpha']:.2e}")
    print(f"Test CC = {results['cc']:.4f}")
    print(f"Test NRMSE = {results['nrmse']:.4f}")
    if show_plot:
        plot_narma_results(results["y_true"], results["y_pred"], order, selected_nodes, results["cc"], results["nrmse"])


def build_random_reservoir_states(u, num_nodes=50, spectral_radius=0.95, input_scale=0.1, leak_rate=0.2, seed=0):
    """Create synthetic reservoir states from input sequence u."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((num_nodes, num_nodes))
    W /= max(abs(np.linalg.eigvals(W)))
    W *= spectral_radius
    Win = rng.standard_normal((num_nodes, 1)) * input_scale
    x = np.zeros((len(u), num_nodes))
    state = np.zeros(num_nodes)

    for t in range(len(u)):
        pre_activation = W @ state + Win.flatten() * u[t]
        state = (1 - leak_rate) * state + leak_rate * np.tanh(pre_activation)
        x[t] = state

    return x


def parse_arguments():
    parser = argparse.ArgumentParser(description="NARMA task benchmark for synthetic and PRC physical response")
    parser.add_argument("--mode", choices=["synthetic", "prc"], default="prc", help="Benchmark mode")
    parser.add_argument("--file", type=str, help="CSV file path for PRC data")
    parser.add_argument("--order", type=int, default=10, choices=[10, 20], help="NARMA order")
    parser.add_argument("--nodes", type=int, default=30, help="Number of reservoir nodes to use")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Train/test split ratio")
    parser.add_argument("--node_point", type=int, default=50, help="Time points per pulse for physical measurement")
    parser.add_argument("--normalize", choices=["global", "per_pulse"], default="global", help="Normalization mode for physical response")
    parser.add_argument("--baseline", type=float, default=0.0, help="Voltage baseline for mapping to NARMA input")
    parser.add_argument("--scale", type=float, default=1.0, help="Voltage scale for mapping to NARMA input")
    parser.add_argument("--length", type=int, default=2500, help="Synthetic sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic input")
    parser.add_argument("--show_plot", action="store_true", help="Show response and prediction plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "synthetic":
        run_narma_synthetic(
            order=args.order,
            length=args.length,
            selected_nodes=args.nodes,
            train_ratio=args.train_ratio,
            seed=args.seed,
            show_plot=args.show_plot,
        )
    else:
        if not args.file:
            raise ValueError("Please specify --file for PRC mode.")
        run_narma_prc(
            filepath=args.file,
            order=args.order,
            selected_nodes=args.nodes,
            train_ratio=args.train_ratio,
            node_point=args.node_point,
            normalize_mode=args.normalize,
            baseline=args.baseline,
            scale=args.scale,
            show_plot=args.show_plot,
        )
