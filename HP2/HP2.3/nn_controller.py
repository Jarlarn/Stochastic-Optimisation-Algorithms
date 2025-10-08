import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from run_encoding_decoding_test import decode_chromosome

# Default weight range used for encoding/decoding
W_MAX = 5.0


class NeuralNetworkController:
    """Neural Network controller for truck braking system"""

    def __init__(
        self,
        ni: int,
        nh: int,
        no: int,
        chromosome: Optional[List[float]] = None,
        w_max: float = W_MAX,
    ):
        """
        Initialize a neural network controller.

        Args:
            ni: Number of input neurons
            nh: Number of hidden neurons
            no: Number of output neurons
            chromosome: Optional chromosome to initialize weights
            w_max: Maximum absolute weight value for decoding
        """
        self.ni = ni
        self.nh = nh
        self.no = no

        # Initialize weights randomly or from chromosome
        if chromosome is not None:
            self.w_i_h, self.w_h_o = decode_chromosome(chromosome, ni, nh, no, w_max)
        else:
            # Initialize with small random weights
            self.w_i_h = np.random.uniform(-0.5, 0.5, (nh, ni + 1))
            self.w_h_o = np.random.uniform(-0.5, 0.5, (no, nh + 1))

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        """
        return 1.0 / (1.0 + np.exp(-x))

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
            brake_temp: Current brake temperature [°C]
            slope_angle: Current slope angle [degrees]
            gear: Current gear

        Returns:
            Tuple of (brake_pedal_pressure, recommended_gear)
        """
        # Normalize inputs to reasonable ranges
        norm_velocity = velocity / 30.0  # Normalize to ~0-1 range for typical speeds
        norm_brake_temp = brake_temp / 750.0  # Normalize to max temp
        norm_slope = slope_angle / 20.0  # Normalize to typical max slope
        norm_gear = gear / 10.0  # Normalize gear

        # Feed inputs to neural network
        nn_inputs = [norm_velocity, norm_brake_temp, norm_slope, norm_gear]
        nn_outputs = self.forward(nn_inputs)

        # Interpret outputs
        brake_pedal = max(0.0, min(1.0, nn_outputs[0]))  # Clamp to [0,1]

        # For gear control, map output to gear range
        if self.no > 1:
            gear_output = nn_outputs[1]
            gear = int(round(gear_output * 9)) + 1  # Map [0,1] -> [1,10]
            gear = max(1, min(10, gear))  # Ensure within valid range
        else:
            gear = None  # Use truck's default gear control

        return brake_pedal, gear


def create_controller_from_chromosome(
    chromosome: List[float], ni: int = 4, nh: int = 6, no: int = 2, w_max: float = W_MAX
) -> Callable:
    """
    Create a controller function from a chromosome.

    Args:
        chromosome: List of weights
        ni: Number of inputs (default 4: velocity, temp, slope, gear)
        nh: Number of hidden neurons
        no: Number of outputs (default 2: brake pedal, gear)
        w_max: Maximum absolute weight value for decoding

    Returns:
        Controller function that takes truck state and returns control signals
    """
    nn = NeuralNetworkController(ni, nh, no, chromosome, w_max)

    def controller(position, velocity, brake_temp, slope_angle, gear):
        return nn.control(position, velocity, brake_temp, slope_angle, gear)

    return controller


def test_nn_controller():
    """Test the neural network controller"""
    # Create a random chromosome
    ni, nh, no = 4, 6, 2
    w_i_h_size = nh * (ni + 1)
    w_h_o_size = no * (nh + 1)
    chromosome = np.random.uniform(
        0, 1, w_i_h_size + w_h_o_size
    ).tolist()  # [0,1] range for chromosome

    # Create controller
    nn = NeuralNetworkController(ni, nh, no, chromosome)

    # Test with some inputs
    position = 100.0
    velocity = 20.0
    brake_temp = 300.0
    slope_angle = 5.0
    gear = 3

    pedal, recommended_gear = nn.control(
        position, velocity, brake_temp, slope_angle, gear
    )

    print(
        f"Test inputs: velocity={velocity}, brake_temp={brake_temp}, slope_angle={slope_angle}, gear={gear}"
    )
    print(f"Control outputs: pedal={pedal:.3f}, recommended_gear={recommended_gear}")


if __name__ == "__main__":
    test_nn_controller()
