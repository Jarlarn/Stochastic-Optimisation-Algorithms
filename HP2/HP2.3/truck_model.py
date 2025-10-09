import math
from enum import IntEnum
from typing import Final, Dict, List, Tuple, Any, Optional
import numpy as np
from slopes import get_slope_angle

GRAVITY: Final = 9.80665
GEAR_FACTORS: Final = (7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1.0)

# Truck model constants
DEFAULT_TEMP_COOLING_TAU = 30.0  # tau time constant for cooling [s] (assignment)
DEFAULT_TEMP_HEATING_CH = 40.0  # Ch heating coefficient [K/s] (assignment)
DEFAULT_AMBIENT_TEMP = 283.0  # Ambient temperature [K] (assignment)
DEFAULT_MAX_BRAKE_TEMP = 750.0  # Maximum allowable brake temp [K] (assignment)
DEFAULT_TIME_STEP = 0.1  # Default simulation time step [s]
MAX_SIMULATION_STEPS = 100000  # Safety limit for simulation steps


class Gear(IntEnum):
    G1 = 1
    G2 = 2
    G3 = 3
    G4 = 4
    G5 = 5
    G6 = 6
    G7 = 7
    G8 = 8
    G9 = 9
    G10 = 10

    @property
    def factor(self) -> float:
        return GEAR_FACTORS[self.value - 1]


def engine_brake_force(base_coeff: float, gear: Gear) -> float:
    return base_coeff * gear.factor


def foundation_brake_force(
    mass: float, pedal: float, brake_temp: float, max_brake_temp: float
) -> float:
    pedal = max(0.0, min(1.0, pedal))
    mg_over_20 = mass * GRAVITY / 20.0
    threshold = max_brake_temp - 100.0
    if brake_temp < threshold:
        return mg_over_20 * pedal
    return mg_over_20 * pedal * math.exp(-(brake_temp - threshold) / 100.0)


def gravity_component(mass: float, slope_angle_deg: float) -> float:
    return mass * GRAVITY * math.sin(math.radians(slope_angle_deg))


class Truck:
    def __init__(
        self,
        mass: float,
        base_engine_brake_coeff: float,
        max_brake_temp: float = DEFAULT_MAX_BRAKE_TEMP,
        temp_cooling_tau: float = DEFAULT_TEMP_COOLING_TAU,
        temp_heating_ch: float = DEFAULT_TEMP_HEATING_CH,
        ambient_temp: float = DEFAULT_AMBIENT_TEMP,
        dt: float = DEFAULT_TIME_STEP,
    ):
        self._mass = float(mass)
        self.base_engine_brake_coeff = float(base_engine_brake_coeff)
        self._gear: Gear = Gear.G1
        self.max_brake_temp = float(max_brake_temp)
        self.temp_cooling_tau = float(temp_cooling_tau)
        self.temp_heating_ch = float(temp_heating_ch)
        self.ambient_temp = float(ambient_temp)

        # State variables
        self.delta_brake_temp = 0.0  # Temperature above ambient (ΔTb)
        self.position = 0.0
        self.velocity = 0.0
        self.dt = dt
        self.time = 0.0

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def gear(self) -> Gear:
        return self._gear

    @property
    def brake_temp(self) -> float:
        """Total brake temperature = ambient + delta"""
        return self.ambient_temp + self.delta_brake_temp

    def set_gear(self, gear) -> None:
        if isinstance(gear, int):
            if not 1 <= gear <= 10:
                raise ValueError("Gear must be 1..10")
            gear = Gear(gear)
        elif not isinstance(gear, Gear):
            raise TypeError("gear must be int or Gear")

        # No-op if same gear
        if gear == self._gear:
            return

        # Accept gear change immediately (no timing constraint)
        self._gear = gear

    def shift_up(self) -> None:
        if self._gear < Gear.G10:
            self._gear = Gear(self._gear + 1)

    def shift_down(self) -> None:
        if self._gear > Gear.G1:
            self._gear = Gear(self._gear - 1)

    def reset(
        self,
        position: float = 0.0,
        velocity: float = 0.0,
        gear: int = 1,
        tb_total=None,
    ) -> None:
        """Reset truck to initial state. tb_total is absolute Tb in K; if None uses ambient."""
        self.position = position
        self.velocity = velocity
        self._gear = Gear(gear) if isinstance(gear, int) else gear
        if tb_total is None:
            self.delta_brake_temp = 0.0
        else:
            # store ΔTb = Tb - Tamb, clamp >= 0
            self.delta_brake_temp = max(0.0, float(tb_total) - self.ambient_temp)
        self.time = 0.0
        # mark last gear change at reset time so next shift must wait min interval
        self.last_gear_change_time = self.time

    # Thin wrappers delegating to pure functions:
    def current_engine_brake(self) -> float:
        return engine_brake_force(self.base_engine_brake_coeff, self._gear)

    def current_service_brake(self, pedal: float) -> float:
        return foundation_brake_force(
            self.mass, pedal, self.brake_temp, self.max_brake_temp
        )

    def slope_angle(self, x: float, slope_index: int, data_set_index: int) -> float:
        return get_slope_angle(x, slope_index, data_set_index)

    def gravity_force(self, x: float, slope_index: int, data_set_index: int) -> float:
        return gravity_component(
            self.mass, self.slope_angle(x, slope_index, data_set_index)
        )

    def net_force(
        self, x: float, slope_index: int, data_set_index: int, pedal: float
    ) -> float:
        """
        Positive direction: downhill.
        F_net = F_gravity - (F_service + F_engine_brake)
        """
        f_g = self.gravity_force(x, slope_index, data_set_index)
        f_sb = self.current_service_brake(pedal)
        f_eb = self.current_engine_brake()
        return f_g - (f_sb + f_eb)

    def update_temperature(self, pedal: float) -> None:
        """Update brake temperature according to eq. 4"""
        if pedal < 0.01:
            # Cooling
            d_temp = -self.delta_brake_temp / self.temp_cooling_tau
        else:
            # Heating
            d_temp = self.temp_heating_ch * pedal

        # Integrate temperature using first-order difference
        self.delta_brake_temp += d_temp * self.dt

        # Temperature can't go below ambient
        self.delta_brake_temp = max(0.0, self.delta_brake_temp)

    def update_state(self, pedal: float, slope_index: int, data_set_index: int) -> None:
        """Update truck state by one time step"""
        # Calculate net force and acceleration
        f_net = self.net_force(self.position, slope_index, data_set_index, pedal)
        acceleration = f_net / self.mass

        # Update velocity using first-order difference
        self.velocity += acceleration * self.dt

        # Update position
        self.position += self.velocity * self.dt

        # Update brake temperature
        self.update_temperature(pedal)

        # Update time
        self.time += self.dt

    def simulate(
        self,
        controller,
        slope_index: int,
        data_set_index: int,
        max_distance: float = 10000.0,
        max_time: float = 3600.0,  # 1 hour max
        auto_gear: bool = False,
        v_min: float = 1.0,  # Minimum allowed velocity (m/s)
        v_max: float = 25.0,  # Maximum allowed velocity (m/s)
    ) -> Dict[str, Any]:
        """
        Simulate truck descent using provided controller function.
        Terminates if constraints are violated.

        Args:
            controller: Function that returns (pedal_pressure, gear) given truck state
            slope_index: Slope profile index
            data_set_index: Data set index
            max_distance: Maximum simulation distance [m]
            max_time: Maximum simulation time [s]
            auto_gear: If True, use automatic gear selection, otherwise use controller gear
            v_min: Minimum allowed velocity [m/s]
            v_max: Maximum allowed velocity [m/s]

        Returns:
            Dict with time series data and fitness metrics
        """
        # Initialize history
        history = {
            "time": [self.time],
            "position": [self.position],
            "velocity": [self.velocity],
            "brake_temp": [self.brake_temp],
            "gear": [self._gear.value],
            "pedal": [0.0],
            "slope_angle": [
                self.slope_angle(self.position, slope_index, data_set_index)
            ],
        }

        step_count = 0
        constraint_violated = False
        termination_reason = "max_distance"

        # Continue until we reach max distance, time, steps, or violate constraints
        while (
            self.position < max_distance
            and self.time < max_time
            and step_count < MAX_SIMULATION_STEPS
        ):
            # Check constraints before controller is called
            if self.velocity > v_max:
                constraint_violated = True
                termination_reason = "v_max_exceeded"
                break

            if self.velocity < v_min and self.time > 5.0:  # Allow some startup time
                constraint_violated = True
                termination_reason = "v_min_violated"
                break

            if self.brake_temp > self.max_brake_temp:
                constraint_violated = True
                termination_reason = "brake_temp_exceeded"
                break

            # Get control inputs from controller
            control_inputs = controller(
                position=self.position,
                velocity=self.velocity,
                brake_temp=self.brake_temp,
                slope_angle=self.slope_angle(
                    self.position, slope_index, data_set_index
                ),
                gear=self._gear.value,
                current_time=self.time,  # Pass current simulation time
            )

            # Unpack control
            if isinstance(control_inputs, tuple) and len(control_inputs) >= 2:
                pedal, gear_change = control_inputs[0], control_inputs[1]
            else:
                pedal = float(control_inputs)
                gear_change = 0  # No change

            # Apply gear control if provided and not using auto gear
            if gear_change is not None and not auto_gear:
                self.apply_gear_change(gear_change)

            # Update state
            self.update_state(pedal, slope_index, data_set_index)

            # Record history
            history["time"].append(self.time)
            history["position"].append(self.position)
            history["velocity"].append(self.velocity)
            history["brake_temp"].append(self.brake_temp)
            history["gear"].append(self._gear.value)
            history["pedal"].append(pedal)
            history["slope_angle"].append(
                self.slope_angle(self.position, slope_index, data_set_index)
            )

            step_count += 1

        # Check if we hit max time
        if self.time >= max_time:
            termination_reason = "max_time"
        elif step_count >= MAX_SIMULATION_STEPS:
            termination_reason = "max_steps"

        # Calculate fitness metrics
        distance_traveled = self.position
        time_elapsed = self.time

        # Calculate average speed (excluding initial zero velocity if present)
        velocities = [v for v in history["velocity"] if v > 0]
        avg_speed = sum(velocities) / len(velocities) if velocities else 0

        # Calculate fitness: F_i = v̄_i * d_i
        fitness = avg_speed * distance_traveled

        # Add metrics to the result
        result = {
            **history,  # Include all history data
            "metrics": {
                "distance_traveled": distance_traveled,
                "time_elapsed": time_elapsed,
                "avg_speed": avg_speed,
                "fitness": fitness,
                "constraint_violated": constraint_violated,
                "termination_reason": termination_reason,
                "completed_slope": self.position >= max_distance,
            },
        }

        return result

    def apply_gear_change(self, gear_change: int) -> None:
        """
        Apply a gear change request from the controller.

        Args:
            gear_change: -1 (down), 0 (no change), 1 (up)
        """
        if gear_change == 0:
            return  # No change requested

        # Apply the requested change immediately (no timing constraint)
        if gear_change > 0 and self._gear.value < 10:  # Shift up
            self._gear = Gear(self._gear.value + 1)
        elif gear_change < 0 and self._gear.value > 1:  # Shift down
            self._gear = Gear(self._gear.value - 1)


def demo():
    truck = Truck(mass=3000, base_engine_brake_coeff=100.0)
    print("Gear Factor EngBrake")
    for g in Gear:
        truck.set_gear(g)
        print(f"{g.value:>4} {g.factor:>6} {truck.current_engine_brake():>8.1f}")

    # Example net force sample
    x = 100.0
    nf = truck.net_force(x, slope_index=1, data_set_index=1, pedal=0.5)
    print("Net force sample:", nf)

    # Simple controller for demo
    def simple_controller(position, velocity, brake_temp, slope_angle, gear):
        # Apply brakes proportional to velocity over threshold
        target_speed = 15.0
        if velocity > target_speed:
            pedal = min(1.0, (velocity - target_speed) / 10.0)
        else:
            pedal = 0.0
        return pedal, gear

    # Run a short simulation
    print("\nRunning simple simulation...")
    truck.reset()
    history = truck.simulate(
        controller=simple_controller,
        slope_index=0,
        data_set_index=0,
        max_distance=1000.0,
    )

    # Print some stats
    print(f"Final position: {history['position'][-1]:.1f} m")
    print(f"Final velocity: {history['velocity'][-1]:.1f} m/s")
    print(f"Max temperature: {max(history['brake_temp']):.1f} °C")
    print(f"Simulation time: {history['time'][-1]:.1f} s")


if __name__ == "__main__":
    demo()
