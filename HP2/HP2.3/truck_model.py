import math
from enum import IntEnum
from typing import Final
from slopes import get_slope_angle

GRAVITY: Final = 9.80665
GEAR_FACTORS: Final = (7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1.0)


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


def service_brake_force(
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
        self, mass: float, base_engine_brake_coeff: float, max_brake_temp: float = 200.0
    ):
        self._mass = float(mass)
        self.base_engine_brake_coeff = float(base_engine_brake_coeff)
        self._gear: Gear = Gear.G1
        self.brake_temp = 0.0
        self.max_brake_temp = float(max_brake_temp)

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def gear(self) -> Gear:
        return self._gear

    def set_gear(self, gear) -> None:
        if isinstance(gear, int):
            if not 1 <= gear <= 10:
                raise ValueError("Gear must be 1..10")
            gear = Gear(gear)
        elif not isinstance(gear, Gear):
            raise TypeError("gear must be int or Gear")
        self._gear = gear

    def shift_up(self) -> None:
        if self._gear < Gear.G10:
            self._gear = Gear(self._gear + 1)

    def shift_down(self) -> None:
        if self._gear > Gear.G1:
            self._gear = Gear(self._gear - 1)

    # Thin wrappers delegating to pure functions:

    def current_engine_brake(self) -> float:
        return engine_brake_force(self.base_engine_brake_coeff, self._gear)

    def current_service_brake(self, pedal: float) -> float:
        return service_brake_force(
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


if __name__ == "__main__":
    demo()
