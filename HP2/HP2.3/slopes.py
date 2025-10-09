import math

ALPHA_MAX = 10.0  # degrees (maximum allowed slope angle)
SLOPE_LENGTH = 1000.0  # horizontal length of every slope [m]


def get_slope_angle(x: float, slope_index: int, data_set_index: int) -> float:
    """
    Return slope angle in degrees (downhill positive).

    x is interpreted on the interval [0, SLOPE_LENGTH]; values outside are clamped.
    Raises ValueError if the slope_index and data_set_index combination is invalid.
    """
    # Clamp x to the canonical slope length so every slope is 1000 m long
    if x < 0.0:
        x = 0.0
    elif x > SLOPE_LENGTH:
        x = SLOPE_LENGTH

    # Validate slope_index and data_set_index
    if data_set_index == 0:  # Default slopes
        if slope_index == 0:
            alpha_deg = 3.5 + math.sin(x / 200)
        elif slope_index == 1:
            alpha_deg = 4.0 + 2.0 * math.sin(x / 300)
        else:
            alpha_deg = 3.0 + math.cos(x / 150)

    elif data_set_index == 1:  # Training set (10 slopes)
        if slope_index == 1:
            alpha_deg = 4 + math.sin(x / 100) + math.cos(math.sqrt(2) * x / 50)
        elif slope_index == 2:
            alpha_deg = 3.5 + math.sin(x / 120) + math.cos(x / 80)
        elif slope_index == 3:
            alpha_deg = 5.0 + math.sin(x / 90) + 0.5 * math.cos(x / 70)
        elif slope_index == 4:
            alpha_deg = 4.5 - 0.5 * math.sin(x / 110) + math.cos(x / 60)
        elif slope_index == 5:
            alpha_deg = 3.8 + math.sin(x / 130) - 0.6 * math.cos(x / 90)
        elif slope_index == 6:
            alpha_deg = 4.2 + 0.7 * math.sin(x / 140) + 0.3 * math.cos(x / 100)
        elif slope_index == 7:
            alpha_deg = 5.2 - 0.4 * math.sin(x / 150) + 0.8 * math.cos(x / 110)
        elif slope_index == 8:
            alpha_deg = 3.6 + 0.9 * math.sin(x / 160) - 0.2 * math.cos(x / 120)
        elif slope_index == 9:
            alpha_deg = 4.8 - 0.3 * math.sin(x / 170) + 0.5 * math.cos(x / 130)
        elif slope_index == 10:
            alpha_deg = 3 + 2 * math.sin(x / 50) + math.cos(math.sqrt(2) * x / 100)
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    elif data_set_index == 2:  # Validation set (5 slopes)
        if slope_index == 1:
            alpha_deg = 6 - math.sin(x / 100) + math.cos(math.sqrt(3) * x / 50)
        elif slope_index == 2:
            alpha_deg = 5.5 - 0.5 * math.sin(x / 120) + 0.8 * math.cos(x / 80)
        elif slope_index == 3:
            alpha_deg = 4.8 + 0.7 * math.sin(x / 90) - 0.3 * math.cos(x / 70)
        elif slope_index == 4:
            alpha_deg = 5.2 - 0.4 * math.sin(x / 110) - 0.5 * math.cos(x / 60)
        elif slope_index == 5:
            alpha_deg = 5 + math.sin(x / 50) + math.cos(math.sqrt(5) * x / 50)
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    elif data_set_index == 3:  # Test set (5 slopes)
        if slope_index == 1:
            alpha_deg = 6 - math.sin(x / 100) + math.cos(math.sqrt(7) * x / 50)
        elif slope_index == 2:
            alpha_deg = 5.8 + 0.6 * math.sin(x / 130) - 0.4 * math.cos(x / 90)
        elif slope_index == 3:
            alpha_deg = 6.2 - 0.5 * math.sin(x / 140) + 0.7 * math.cos(x / 100)
        elif slope_index == 4:
            alpha_deg = 5.5 + 0.8 * math.sin(x / 150) - 0.2 * math.cos(x / 110)
        elif slope_index == 5:
            alpha_deg = (
                4 + (x / 1000) + math.sin(x / 70) + math.cos(math.sqrt(7) * x / 100)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    else:
        raise ValueError(f"Invalid data_set_index {data_set_index}")

    # Ensure the angle is a sensible positive degree value (downhill), maximum 10 degrees
    return max(0.5, min(ALPHA_MAX, alpha_deg))
