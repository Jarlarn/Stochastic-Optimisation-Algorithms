import numpy as np
from typing import List, Tuple, Optional, Callable
from run_encoding_decoding_test import decode_chromosome

W_MAX = 10.0
V_MAX = 25.0
V_MIN = 1.0
ALPHA_MAX = 10.0
T_MAX = 750.0
SIGMOID_C = 2.0


class NeuralNetworkController:
    def __init__(
        self,
        ni: int = 3,
        nh: int = 5,
        no: int = 2,
        chromosome: Optional[List[float]] = None,
        w_max: float = W_MAX,
        sigmoid_c: float = SIGMOID_C,
    ):
        self.ni = ni
        self.nh = nh
        self.no = no
        self.sigmoid_c = sigmoid_c
        if chromosome is not None:
            self.w_i_h, self.w_h_o = decode_chromosome(chromosome, ni, nh, no, w_max)
        else:
            self.w_i_h = np.random.uniform(-1, 1, (nh, ni + 1))
            self.w_h_o = np.random.uniform(-1, 1, (no, nh + 1))

    def activate(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -20.0 / self.sigmoid_c, 20.0 / self.sigmoid_c)
        return 1.0 / (1.0 + np.exp(-self.sigmoid_c * x_clipped))

    def forward(self, inputs: List[float]) -> List[float]:
        if len(inputs) != self.ni:
            raise ValueError(f"Expected {self.ni} inputs, got {len(inputs)}")
        x = np.ones(self.ni + 1)
        x[1:] = inputs
        h_in = np.dot(self.w_i_h, x)
        h_out = np.ones(self.nh + 1)
        h_out[1:] = self.activate(h_in)
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
        current_time: float,
    ) -> Tuple[float, int]:
        norm_velocity = velocity / V_MAX
        norm_slope = slope_angle / ALPHA_MAX
        norm_temp = brake_temp / T_MAX
        nn_inputs = [norm_velocity, norm_slope, norm_temp]
        nn_outputs = self.forward(nn_inputs)
        brake_pedal = max(0.0, min(1.0, nn_outputs[0]))
        gear_out = nn_outputs[1]
        down_shift_threshold = 0.4
        up_shift_threshold = 0.6
        if gear_out < down_shift_threshold:
            gear_change = -1
        elif gear_out > up_shift_threshold:
            gear_change = 1
        else:
            gear_change = 0
        if velocity < 1.5 * V_MIN and gear < 10:
            gear_change = 1
        if velocity < 2.0 * V_MIN and gear_change == -1:
            gear_change = 0
        if velocity > 0.85 * V_MAX and gear > 1:
            gear_change = -1
        if not hasattr(self, "last_gear_change_time"):
            self.last_gear_change_time = -float("inf")
        if current_time - self.last_gear_change_time < 2.0:
            gear_change = 0
        else:
            if gear_change != 0:
                self.last_gear_change_time = current_time
        return brake_pedal, gear_change


def create_controller_from_chromosome(
    chromosome: List[float],
    ni: int = 3,
    nh: int = 5,
    no: int = 2,
    w_max: float = W_MAX,
    sigmoid_c: float = SIGMOID_C,
) -> Callable:
    nn = NeuralNetworkController(ni, nh, no, chromosome, w_max, sigmoid_c)

    def controller(position, velocity, brake_temp, slope_angle, gear, current_time):
        return nn.control(
            position, velocity, brake_temp, slope_angle, gear, current_time
        )

    return controller
