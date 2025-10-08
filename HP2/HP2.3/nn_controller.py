import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional, Callable
from run_encoding_decoding_test import decode_chromosome

# Default weight range used for encoding/decoding
W_MAX = 5.0

# Constants for truck control (matching assignment specs)
V_MAX = 25.0  # m/s
V_MIN = 1.0  # m/s
ALPHA_MAX = 10.0  # degrees
T_MAX = 750.0  # K

# Sigmoid parameter c ∈ [1,3] as specified in the assignment
SIGMOID_C = 2.0  # You can adjust this within the [1,3] range


class NeuralNetworkController:
    """Neural Network controller for truck braking system"""

    def __init__(
        self,
        ni: int = 3,  # Changed to 3 inputs per assignment specs
        nh: int = 6,
        no: int = 2,  # Two outputs: brake pedal pressure and gear change
        chromosome: Optional[List[float]] = None,
        w_max: float = W_MAX,
        sigmoid_c: float = SIGMOID_C,
    ):
        """
        Initialize a neural network controller.

        Args:
            ni: Number of input neurons (3: normalized velocity, slope, temperature)
            nh: Number of hidden neurons
            no: Number of outputs (2: pedal pressure and gear change)
            chromosome: Optional chromosome to initialize weights
            w_max: Maximum absolute weight value for decoding
            sigmoid_c: Sigmoid activation parameter c ∈ [1,3]
        """
        self.ni = ni
        self.nh = nh
        self.no = no
        self.sigmoid_c = sigmoid_c

        # Initialize weights randomly or from chromosome
        if chromosome is not None:
            self.w_i_h, self.w_h_o = decode_chromosome(chromosome, ni, nh, no, w_max)
        else:
            # Initialize with small random weights
            self.w_i_h = np.random.uniform(-0.5, 0.5, (nh, ni + 1))
            self.w_h_o = np.random.uniform(-0.5, 0.5, (no, nh + 1))

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with parameter c
        f(x) = 1/(1+e^(-c*x))
        """
        # Clip to prevent overflow
        x_clipped = np.clip(x, -20.0 / self.sigmoid_c, 20.0 / self.sigmoid_c)
        return 1.0 / (1.0 + np.exp(-self.sigmoid_c * x_clipped))

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Run the neural network forward pass

        Args:
            inputs: List of input values (length ni)

        Returns:
            List of output values (length no)
        """
        # Ensure inputs has the correct length
        if len(inputs) != self.ni:
            raise ValueError(f"Expected {self.ni} inputs, got {len(inputs)}")

        # Add bias input
        x = np.ones(self.ni + 1)
        x[1:] = inputs  # x[0] is the bias (1.0)

        # Hidden layer
        h_in = np.dot(self.w_i_h, x)
        h_out = np.ones(self.nh + 1)  # Add bias
        h_out[1:] = self.activate(h_in)  # h_out[0] is the bias (1.0)

        # Output layer
        o_in = np.dot(self.w_h_o, h_out)
        o_out = self.activate(o_in)

        return o_out.tolist()

    def control(
        self,
        position: float,
        velocity: float,
        brake_temp: float,
        slope_angle: float,
        gear: int,
    ) -> Tuple[float, int]:
        """
        Control function interface for use with truck simulation.

        Args:
            position: Current position [m]
            velocity: Current velocity [m/s]
            brake_temp: Current brake temperature [K]
            slope_angle: Current slope angle [degrees]
            gear: Current gear

        Returns:
            Tuple of (brake_pedal_pressure, gear_change)
            where gear_change is -1 (down), 0 (no change), or 1 (up)
        """
        # Normalize inputs as specified in assignment:
        # v/vmax, α/αmax, Tb/Tmax
        norm_velocity = velocity / V_MAX
        norm_slope = slope_angle / ALPHA_MAX
        norm_temp = brake_temp / T_MAX

        # Feed inputs to neural network (3 inputs as specified)
        nn_inputs = [norm_velocity, norm_slope, norm_temp]
        nn_outputs = self.forward(nn_inputs)

        # Interpret outputs
        # First output: brake pedal pressure [0,1]
        brake_pedal = max(0.0, min(1.0, nn_outputs[0]))

        # Second output: gear change
        # Map to [-1, 0, 1] for (down, no change, up)
        gear_out = nn_outputs[1]
        # Threshold into 3 regions
        if gear_out < 0.33:
            gear_change = -1  # Down-shift
        elif gear_out > 0.67:
            gear_change = 1  # Up-shift
        else:
            gear_change = 0  # No change

        return brake_pedal, gear_change


def create_controller_from_chromosome(
    chromosome: List[float],
    ni: int = 3,  # Changed to 3 inputs
    nh: int = 6,
    no: int = 2,
    w_max: float = W_MAX,
    sigmoid_c: float = SIGMOID_C,
) -> Callable:
    """
    Create a controller function from a chromosome.

    Args:
        chromosome: List of weights
        ni: Number of inputs (3: velocity, slope, temperature)
        nh: Number of hidden neurons
        no: Number of outputs (2: brake pedal, gear change)
        w_max: Maximum absolute weight value for decoding
        sigmoid_c: Sigmoid activation parameter

    Returns:
        Controller function that takes truck state and returns control signals
    """
    nn = NeuralNetworkController(ni, nh, no, chromosome, w_max, sigmoid_c)

    def controller(position, velocity, brake_temp, slope_angle, gear):
        return nn.control(position, velocity, brake_temp, slope_angle, gear)

    return controller


def test_nn_controller():
    """Test the neural network controller"""
    # Create a random chromosome
    ni, nh, no = 3, 6, 2  # 3 inputs as specified
    w_i_h_size = nh * (ni + 1)
    w_h_o_size = no * (nh + 1)
    chromosome = np.random.uniform(
        0, 1, w_i_h_size + w_h_o_size
    ).tolist()  # [0,1] range

    # Create controller
    nn = NeuralNetworkController(ni, nh, no, chromosome)

    # Test with some inputs
    position = 100.0
    velocity = 20.0
    brake_temp = 500.0
    slope_angle = 5.0
    gear = 3

    pedal, gear_change = nn.control(position, velocity, brake_temp, slope_angle, gear)

    print(
        f"Test inputs: velocity={velocity}, brake_temp={brake_temp}, slope_angle={slope_angle}, gear={gear}"
    )
    print(f"Control outputs: pedal={pedal:.3f}, gear_change={gear_change}")
    print(
        f"Normalized inputs: v/vmax={velocity/V_MAX:.3f}, α/αmax={slope_angle/ALPHA_MAX:.3f}, Tb/Tmax={brake_temp/T_MAX:.3f}"
    )


if __name__ == "__main__":
    test_nn_controller()
