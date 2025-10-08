import math
from enum import IntEnum
from slopes import get_slope_angle


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
        # Index by (value - 1)
        factors = [7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1.0]
        return factors[self.value - 1]


class Truck:
    def __init__(self, mass: float, base_engine_brake_coeff: float):
        """
        base_engine_brake_coeff corresponds to C_b in the formula.
        """
        self.base_engine_brake_coeff = base_engine_brake_coeff
        self._gear: Gear = Gear.G1  # start in gear 1
        self.mass = mass
        self.max_brake_temp = 200
        self.brake_temp = 0

    @property
    def gear(self) -> Gear:
        return self._gear

    def set_gear(self, gear) -> None:
        """
        Accepts a Gear enum value or an int 1..10.
        """
        if isinstance(gear, int):
            if not 1 <= gear <= 10:
                raise ValueError("Gear must be in 1..10")
            gear = Gear(gear)
        elif not isinstance(gear, Gear):
            raise TypeError("gear must be int or Gear")
        self._gear = gear

    def shift_up(self) -> None:
        self._gear = Gear(min(self._gear + 1, Gear.G10))

    def shift_down(self) -> None:
        self._gear = Gear(max(self._gear - 1, Gear.G1))

    def engine_brake_force(self) -> float:
        """
        Returns F_eb = factor * C_b for the current gear.
        """
        return self._gear.factor * self.base_engine_brake_coeff

    def slope_angle(self, x: float, slope_index: int, data_set_index: int) -> float:
        return get_slope_angle(x, slope_index, data_set_index)

    def braking_force(self, P_p: float) -> float:
        """Compute service brake force F_b.

        P_p: pedal position (0..1).
        """
        g = 9.80665
        mg_over_20 = self.mass * g / 20.0
        # Optional: clamp pedal
        P_p = max(0.0, min(1.0, P_p))
        threshold = self.max_brake_temp - 100
        if self.brake_temp < threshold:
            Fb = mg_over_20 * P_p
        else:
            Fb = mg_over_20 * P_p * math.exp(-(self.brake_temp - threshold) / 100.0)
        return Fb

    def gravitational_force(self, x, slope_index, data_set_index):
        """Gravity component along the slope (positive downhill)."""
        g = 9.80665
        alpha_deg = self.slope_angle(x, slope_index, data_set_index)
        alpha_rad = math.radians(alpha_deg)
        return self.mass * g * math.sin(alpha_rad)


def demo():
    truck = Truck(mass=3000, base_engine_brake_coeff=100.0)
    print("Gear  Factor  EngineBrakeForce")
    for g in Gear:
        truck.set_gear(g)
        print(f"{g.value:>4}  {g.factor:>6}  {truck.engine_brake_force():>16.1f}")
    print(truck.braking_force(0.5))


if __name__ == "__main__":
    demo()
