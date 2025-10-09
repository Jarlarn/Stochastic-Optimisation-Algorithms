import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional

from truck_model import Truck
from nn_controller import create_controller_from_chromosome


def load_best_chromosome(filename: str = "best_chromosome.json") -> Dict[str, Any]:
    """Load the best chromosome from a file"""
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def plot_simulation_results(
    history: Dict[str, List[float]], title: str = "Truck Simulation Results"
):
    """Plot the results of a simulation"""
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # Plot position vs time
    axs[0].plot(history["time"], history["position"])
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Position [m]")
    axs[0].set_title("Position vs Time")
    axs[0].grid(True)

    # Plot velocity vs time
    axs[1].plot(history["time"], history["velocity"])
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].set_title("Velocity vs Time")
    axs[1].grid(True)

    # Plot brake temperature vs time
    axs[2].plot(history["time"], history["brake_temp"])
    axs[2].axhline(y=750.0, color="r", linestyle="--", label="Max Temp")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Brake Temperature [°C]")
    axs[2].set_title("Brake Temperature vs Time")
    axs[2].legend()
    axs[2].grid(True)

    # Plot gear and pedal pressure vs time
    ax3a = axs[3]
    ax3a.plot(history["time"], history["pedal"], "g-", label="Pedal Pressure")
    ax3a.set_xlabel("Time [s]")
    ax3a.set_ylabel("Pedal Pressure [0-1]")
    ax3a.set_ylim([0, 1.1])
    ax3a.grid(True)

    ax3b = ax3a.twinx()
    ax3b.plot(history["time"], history["gear"], "b-", label="Gear")
    ax3b.set_ylabel("Gear")
    ax3b.set_ylim([0, 11])

    lines1, labels1 = ax3a.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3b.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Add slope angle to the velocity plot
    ax1b = axs[1].twinx()
    ax1b.plot(history["time"], history["slope_angle"], "r--", label="Slope Angle")
    ax1b.set_ylabel("Slope Angle [degrees]")

    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)

    return fig


def test_controller(
    chromosome: List[float],
    ni: int,
    nh: int,
    no: int,
    slope_index: int = 0,
    data_set_index: int = 0,
    max_distance: float = 10000.0,
    auto_gear: bool = False,
    save_plot: bool = False,
) -> Dict[str, List[float]]:
    """Test a controller on a specific slope and data set"""
    # Create truck and controller
    truck = Truck(
        mass=20000.0, base_engine_brake_coeff=3000.0, max_brake_temp=750.0
    )  # Fixed Cb value
    controller = create_controller_from_chromosome(chromosome, ni, nh, no)

    truck.reset(
        position=0.0, velocity=20.0, gear=7, tb_total=500.0
    )  # Set initial state

    # Run simulation
    history = truck.simulate(
        controller=controller,
        slope_index=slope_index,
        data_set_index=data_set_index,
        max_distance=max_distance,
        auto_gear=auto_gear,
        max_time=3600.0,
    )

    # Calculate performance metrics
    max_velocity = max(history["velocity"])
    max_brake_temp = max(history["brake_temp"])
    avg_velocity = sum(history["velocity"]) / len(history["velocity"])
    final_distance = history["position"][-1]
    simulation_time = history["time"][-1]

    # Print summary
    print(f"Simulation Summary (Slope {slope_index}, Dataset {data_set_index}):")
    print(f"  Distance traveled: {final_distance:.2f} m")
    print(f"  Maximum velocity: {max_velocity:.2f} m/s")
    print(f"  Average velocity: {avg_velocity:.2f} m/s")
    print(f"  Maximum brake temperature: {max_brake_temp:.2f} °C")
    print(f"  Simulation time: {simulation_time:.2f} s")

    # Plot results
    title = f"Truck Simulation - Slope {slope_index}, Dataset {data_set_index}"
    fig = plot_simulation_results(history, title)

    if save_plot:
        plt.savefig(
            f"simulation_results_slope{slope_index}_dataset{data_set_index}.png"
        )
    else:
        plt.show()

    return history


def main():
    # Try to load from best_chromosome.py first (direct import)
    try:
        import best_chromosome

        ni = getattr(best_chromosome, "NI", 3)
        nh = getattr(best_chromosome, "NH", 6)
        no = getattr(best_chromosome, "NO", 2)
        sigmoid_c = getattr(best_chromosome, "SIGMOID_C", 2.0)
        chromosome = best_chromosome.CHROMOSOME
        print(
            f"Loaded chromosome from best_chromosome.py with ni={ni}, nh={nh}, no={no}"
        )
    except ImportError:
        # Fall back to JSON file
        try:
            data = load_best_chromosome()
            chromosome = data["chromosome"]
            ni = data.get("ni", 3)
            nh = data.get("nh", 6)
            no = data.get("no", 2)
            sigmoid_c = data.get("sigmoid_c", 2.0)
            print(
                f"Loaded chromosome from best_chromosome.json with ni={ni}, nh={nh}, no={no}"
            )
            print(f"Best fitness from optimization: {data.get('fitness', 'N/A')}")
        except FileNotFoundError:
            print("Error: Neither best_chromosome.py nor best_chromosome.json found.")
            return

    # Test parameters from assignment
    truck = Truck(
        mass=20000.0,  # 20000 kg
        base_engine_brake_coeff=3000.0,  # Was 2000.0, should be 3000.0
        max_brake_temp=750.0,
    )

    # Test on all test set slopes
    print("\nRunning tests on test set slopes...")
    for slope_index in range(1, 6):  # Test slopes 1-5
        test_controller(
            chromosome=chromosome,
            ni=ni,
            nh=nh,
            no=no,
            slope_index=slope_index,
            data_set_index=3,  # Test set (data_set_index=3)
            max_distance=1000.0,
            auto_gear=False,  # Use NN's gear change logic
            save_plot=True,
        )
