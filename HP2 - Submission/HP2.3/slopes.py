import math

ALPHA_MAX = 10.0
SLOPE_LENGTH = 1000.0


def get_slope_angle(x: float, slope_index: int, data_set_index: int) -> float:
    x = max(0.0, min(x, SLOPE_LENGTH))
    if data_set_index == 1:
        if slope_index == 1:
            alpha_deg = (
                5.0 + 1.0 * math.sin(x / 100) + 2.0 * math.cos(math.sqrt(2) * x / 50)
            )
        elif slope_index == 2:
            alpha_deg = (
                4.0 + 1.0 * math.sin(x / 110) + 2.0 * math.cos(math.sqrt(2) * x / 60)
            )
        elif slope_index == 3:
            alpha_deg = (
                6.0 + 1.5 * math.sin(x / 80) + 1.5 * math.cos(math.sqrt(7) * x / 60)
            )
        elif slope_index == 4:
            alpha_deg = (
                2.5 + 1.0 * math.sin(x / 90) + 1.5 * math.cos(math.sqrt(2) * x / 80)
            )
        elif slope_index == 5:
            alpha_deg = (
                5.5 + 1.0 * math.sin(x / 120) - 2.0 * math.cos(math.sqrt(5) * x / 70)
            )
        elif slope_index == 6:
            alpha_deg = (
                5.0 + 1.0 * math.sin(x / 140) + 2.0 * math.cos(math.sqrt(2) * x / 100)
            )
        elif slope_index == 7:
            alpha_deg = (
                4.0 + 1.0 * math.sin(x / 80) + 2.0 * math.cos(math.sqrt(2) * x / 110)
            )
        elif slope_index == 8:
            alpha_deg = 7.0 + 0.5 * math.sin(x / 200)
        elif slope_index == 9:
            alpha_deg = (
                2.0 + 1.0 * math.sin(x / 160) + 1.0 * math.cos(math.sqrt(2) * x / 130)
            )
        elif slope_index == 10:
            alpha_deg = (
                5.0 + 2.0 * math.sin(x / 50) + 2.0 * math.cos(math.sqrt(2) * x / 100)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )
    elif data_set_index == 2:
        if slope_index == 1:
            alpha_deg = (
                5.0 - 1.0 * math.sin(x / 100) + 2.0 * math.cos(math.sqrt(3) * x / 50)
            )
        elif slope_index == 2:
            alpha_deg = (
                4.0 + 1.0 * math.sin(x / 120) - 1.5 * math.cos(math.sqrt(3) * x / 60)
            )
        elif slope_index == 3:
            alpha_deg = (
                5.2 - 1.2 * math.sin(x / 90) + 1.4 * math.cos(math.sqrt(3) * x / 70)
            )
        elif slope_index == 4:
            alpha_deg = (
                5.0 + 0.6 * math.sin(x / 110) - 1.0 * math.cos(math.sqrt(3) * x / 80)
            )
        elif slope_index == 5:
            alpha_deg = (
                4.0 + 2.0 * math.sin(x / 50) + 2.0 * math.cos(math.sqrt(5) * x / 50)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )
    elif data_set_index == 3:
        if slope_index == 1:
            alpha_deg = (
                5.0 - 1.0 * math.sin(x / 100) + 2.0 * math.cos(math.sqrt(7) * x / 50)
            )
        elif slope_index == 2:
            alpha_deg = (
                4.2 + 1.2 * math.sin(x / 120) - 1.6 * math.cos(math.sqrt(7) * x / 60)
            )
        elif slope_index == 3:
            alpha_deg = (
                4.2 + 1.8 * math.sin(x / 90) + 2.0 * math.cos(math.sqrt(7) * x / 70)
            )
        elif slope_index == 4:
            alpha_deg = (
                4.0 - 1.0 * math.sin(x / 110) + 2.0 * math.cos(math.sqrt(7) * x / 80)
            )
        elif slope_index == 5:
            alpha_deg = (
                4.0
                + (x / 1000)
                + 1.0 * math.sin(x / 70)
                + 1.0 * math.cos(math.sqrt(7) * x / 100)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )
    else:
        raise ValueError(f"Invalid data_set_index {data_set_index}")
    return alpha_deg
