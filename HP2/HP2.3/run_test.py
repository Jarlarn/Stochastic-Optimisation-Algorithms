import argparse
from typing import List, Dict, Any

import matplotlib.pyplot as plt

from truck_model import Truck
from nn_controller import create_controller_from_chromosome
from slopes import SLOPE_LENGTH


def plot_simulation_results(
    history: Dict[str, List[float]], title: str = "Truck Simulation Results"
):
    fig, axs = plt.subplots(5, 1, figsize=(12, 18))
    x = history["position"]

    axs[0].plot(x, history["slope_angle"], "r-")
    axs[0].set_ylabel("Slope angle [deg]")
    axs[0].set_title("Slope angle α vs x")
    axs[0].grid(True)
    axs[1].plot(x, history["pedal"], "g-")
    axs[1].set_ylabel("Pedal [0-1]")
    axs[1].grid(True)
    axs[2].plot(x, history["gear"], "b-")
    axs[2].set_ylabel("Gear")
    axs[2].grid(True)
    axs[3].plot(x, history["velocity"], "k-")
    axs[3].set_ylabel("Velocity [m/s]")
    axs[3].grid(True)
    axs[4].plot(x, history["brake_temp"], "m-")
    axs[4].axhline(y=750.0, color="r", linestyle="--", label="Tmax")
    axs[4].set_ylabel("Brake Temp [K]")
    axs[4].legend()
    axs[4].grid(True)

    for ax in axs:
        ax.set_xlabel("Horizontal distance x [m]")
    fig.tight_layout()
    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(top=0.95)
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Run best chromosome on a slope and plot results"
    )
    p.add_argument(
        "--slope", type=int, default=1, help="Slope index to run (default: 1)"
    )
    p.add_argument(
        "--data-set", type=int, default=3, help="Data set index (default: 3 = test set)"
    )
    p.add_argument(
        "--best-py",
        default="best_chromosome.py",
        help="Python module with best chromosome",
    )
    p.add_argument(
        "--save-plot", action="store_true", help="Save plot instead of showing"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run the simulation on all slope indexes for the selected data set",
    )
    args = p.parse_args()

    # Sanity check: sample slope angle at start/mid/end to detect constant/default slope
    from slopes import get_slope_angle

    # Load chromosome from Python module only
    import importlib.util, os

    if not os.path.exists(args.best_py):
        print(
            f"Error: {args.best_py} not found. Please run the optimizer to produce best_chromosome.py"
        )
        return
    spec = importlib.util.spec_from_file_location("best_chromosome", args.best_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    chromosome = mod.CHROMOSOME
    ni = getattr(mod, "NI", 3)
    nh = getattr(mod, "NH", 6)
    no = getattr(mod, "NO", 2)
    print(f"Loaded best chromosome from {args.best_py}")

    # Create truck and controller once
    truck = Truck(mass=20000.0, base_engine_brake_coeff=3000.0, max_brake_temp=750.0)
    controller = create_controller_from_chromosome(chromosome, ni, nh, no)

    # Determine which slopes to run
    if args.all:
        if args.data_set == 1:
            slope_indices = range(1, 11)  # 10 training slopes
        else:
            slope_indices = range(1, 6)  # 5 validation/test slopes
    else:
        slope_indices = [args.slope]

    for slope_idx in slope_indices:
        # sample slope to detect errors
        a0 = get_slope_angle(0.0, slope_idx, args.data_set)
        a_mid = get_slope_angle(SLOPE_LENGTH / 2.0, slope_idx, args.data_set)
        a_end = get_slope_angle(SLOPE_LENGTH, slope_idx, args.data_set)
        print(
            f"Slope {slope_idx} sampled angles (deg) at x=0,mid,end: {a0:.3f}, {a_mid:.3f}, {a_end:.3f}"
        )
        if abs(a0 - a_mid) < 1e-6 and abs(a0 - a_end) < 1e-6:
            print(
                f"Warning: sampled slope angles appear constant for slope {slope_idx} — check slope definition."
            )

        # Initialize truck state per assignment before each run
        truck.reset(position=0.0, velocity=20.0, gear=7, tb_total=500.0)

        print(
            f"Running slope {slope_idx}, data_set {args.data_set} (max_distance={SLOPE_LENGTH}) ..."
        )
        history = truck.simulate(
            controller=controller,
            slope_index=slope_idx,
            data_set_index=args.data_set,
            max_distance=SLOPE_LENGTH,
            max_time=3600.0,
        )

        # Report summary
        final_distance = history["position"][-1] if history["position"] else 0.0
        max_v = max(history["velocity"]) if history["velocity"] else 0.0
        max_tb = max(history["brake_temp"]) if history["brake_temp"] else 0.0
        print(
            f"Slope {slope_idx} -> Distance: {final_distance:.2f} m, Max v: {max_v:.2f} m/s, Max Tb: {max_tb:.2f} K"
        )

        fig = plot_simulation_results(
            history, title=f"Slope {slope_idx} - Dataset {args.data_set}"
        )
        if args.save_plot:
            out = f"simulation_slope{slope_idx}_ds{args.data_set}.png"
            fig.savefig(out)
            print(f"Saved plot to {out}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
