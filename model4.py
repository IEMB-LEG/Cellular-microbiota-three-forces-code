import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False


def calculate_braycurtis_distances(n_max=10, base_step=0.65):
    generations = np.arange(1, n_max + 1)
    generation_diffs = np.arange(1, n_max + 1)

    if n_max >= 1:
        g = generations.astype(float)  # 1..n_max
        b = float(base_step)

        A = np.log((1.0 - b) / b)          # ln(1/b - 1)
        B = np.log(1.0 / 0.99 - 1.0)       # ln(1/0.95 - 1)
        if n_max == 1:
            alpha = 1.0
            gamma = 1.0
        else:
            alpha = (A - B) / (n_max - 1)
            gamma = 1.0 + A / alpha

        dist_initial_mean = 1.0 / (1.0 + np.exp(-alpha * (g - gamma)))
        dist_initial_mean = np.clip(dist_initial_mean, 0.0, 1.0)

    dist_initial_std =  0.02 * generation_diffs + 0.01 
    dist_initial_lower = np.maximum(dist_initial_mean - 1.96 * dist_initial_std, 0)
    dist_initial_upper = np.minimum(dist_initial_mean + 1.96 * dist_initial_std, 1)

    dist_between_mean = np.zeros_like(generation_diffs, dtype=float)
    dist_between_mean[0] = base_step

    if n_max > 1:
       Aamp = 0.05     
       omega = np.pi/4   
       phi = 0.0
       dist_between_mean = base_step + Aamp * np.sin(omega * (generation_diffs - 1) + phi)
       dist_between_mean = np.clip(dist_between_mean, 0.0, 1.0)

    dist_between_std = 0.02 * generation_diffs + 0.01
    dist_between_lower = np.maximum(dist_between_mean - 1.96 * dist_between_std, 0)
    dist_between_upper = np.minimum(dist_between_mean + 1.96 * dist_between_std, 1)

    return {
        "generations": generations,
        "dist_initial": {"mean": dist_initial_mean, "lower": dist_initial_lower, "upper": dist_initial_upper},
        "gen_diffs": generation_diffs,
        "dist_between": {"mean": dist_between_mean, "lower": dist_between_lower, "upper": dist_between_upper}
    }


def plot_horn_plots(distance_data, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    gen = distance_data["generations"]
    init_dist = distance_data["dist_initial"]

    ax1.fill_between(
        x=gen, y1=init_dist["lower"], y2=init_dist["upper"],
        color="#2E86AB", alpha=0.3
    )
    ax1.plot(
        gen, init_dist["mean"], "o-", color="#2E86AB",
        markersize=8, linewidth=2.5, markerfacecolor="#A23B72"
    )
    ax1.scatter(
        1, init_dist["mean"][0], color="#F18F01", s=150, zorder=5
    )
    ax1.axhline(y=1, color="#C73E1D", linestyle="--", linewidth=2)

    ax1.set_title("Simulated Distance Between Each Generation and the First Generation", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("n", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Bray-CurtisD istance", fontsize=12, fontweight="bold")
    ax1.set_xlim(0.5, len(gen) + 0.5)
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    diffs = distance_data["gen_diffs"]
    between_dist = distance_data["dist_between"]

    ax2.fill_between(
        x=diffs, y1=between_dist["lower"], y2=between_dist["upper"],
        color="#00A896", alpha=0.3
    )
    ax2.plot(
        diffs, between_dist["mean"], "s-", color="#00A896",
        markersize=8, linewidth=2.5, markerfacecolor="#02353C"
    )
    ax2.scatter(
        1, between_dist["mean"][0], color="#F18F01", s=150, zorder=5
    )
    ax2.axhline(y=1, color="#C73E1D", linestyle="--", linewidth=2)

    ax2.set_title("Simulated Distance of Each Generation", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("n", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Bray-Curtis Distance", fontsize=12, fontweight="bold")
    ax2.set_xlim(0.5, len(diffs) + 0.5)
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    plt.legend() 

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"图表已保存至：{output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Bray-Curtis Dynamic")
    parser.add_argument("--max-gen", "-n", type=int, default=10, help="The max generation")
    parser.add_argument("--base-step", "-d", type=float, default=0.65, help="The original distance")
    parser.add_argument("--output", "-o", type=str, default="microbial_succession_plots.pdf", help="Output path")
    args = parser.parse_args()

    if args.base_step != 0.65:
        print("Original distance is 0.65")
        args.base_step = 0.65

    distance_data = calculate_braycurtis_distances(args.max_gen, args.base_step)
    
    plot_horn_plots(distance_data, args.output)


if __name__ == "__main__":
    main()
