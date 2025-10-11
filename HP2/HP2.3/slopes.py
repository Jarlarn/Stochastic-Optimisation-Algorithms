import math

ALPHA_MAX = 10.0  # degrees (maximum allowed slope angle)
SLOPE_LENGTH = 1000.0  # horizontal length of every slope [m]


def get_slope_angle(x: float, slope_index: int, data_set_index: int) -> float:
    """
    Return slope angle in degrees (downhill positive).

    x is interpreted on the interval [0, SLOPE_LENGTH]; values outside are clamped.
    Raises ValueError if the slope_index and data_set_index combination is invalid.
    Only sin and cos functions are used for slope shapes.
    """
    # Clamp x to the canonical slope length so every slope is 1000 m long
    if x < 0.0:
        x = 0.0
    elif x > SLOPE_LENGTH:
        x = SLOPE_LENGTH

    # Training set slopes (10 slopes)
    if data_set_index == 1:
        if slope_index == 1:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.2 * math.sin(x / 120) + 0.15 * math.cos(x / 80)
            )
        elif slope_index == 2:
            alpha_deg = ALPHA_MAX * (
                0.25 + 0.25 * math.sin(x / 200) + 0.2 * math.cos(x / 60)
            )
        elif slope_index == 3:
            alpha_deg = ALPHA_MAX * (
                0.35 + 0.2 * math.sin(x / 90) + 0.1 * math.cos(x / 150)
            )
        elif slope_index == 4:
            alpha_deg = ALPHA_MAX * (
                0.2 + 0.3 * math.sin(x / 70) + 0.2 * math.cos(x / 110)
            )
        elif slope_index == 5:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.15 * math.sin(x / 50) + 0.25 * math.cos(x / 130)
            )
        elif slope_index == 6:
            alpha_deg = ALPHA_MAX * (
                0.28 + 0.22 * math.sin(x / 100) + 0.18 * math.cos(x / 90)
            )
        elif slope_index == 7:
            alpha_deg = ALPHA_MAX * (
                0.32 + 0.18 * math.sin(x / 80) + 0.2 * math.cos(x / 140)
            )
        elif slope_index == 8:
            alpha_deg = ALPHA_MAX * (
                0.27 + 0.23 * math.sin(x / 60) + 0.15 * math.cos(x / 120)
            )
        elif slope_index == 9:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.2 * math.sin(x / 150) + 0.2 * math.cos(x / 100)
            )
        elif slope_index == 10:
            alpha_deg = ALPHA_MAX * (
                0.25 + 0.25 * math.sin(x / 170) + 0.2 * math.cos(x / 80)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    # Validation set slopes (5 slopes)
    elif data_set_index == 2:
        if slope_index == 1:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.2 * math.sin(x / 110) + 0.15 * math.cos(x / 90)
            )
        elif slope_index == 2:
            alpha_deg = ALPHA_MAX * (
                0.25 + 0.25 * math.sin(x / 130) + 0.2 * math.cos(x / 70)
            )
        elif slope_index == 3:
            alpha_deg = ALPHA_MAX * (
                0.35 + 0.2 * math.sin(x / 80) + 0.1 * math.cos(x / 150)
            )
        elif slope_index == 4:
            alpha_deg = ALPHA_MAX * (
                0.2 + 0.3 * math.sin(x / 60) + 0.2 * math.cos(x / 110)
            )
        elif slope_index == 5:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.15 * math.sin(x / 100) + 0.25 * math.cos(x / 130)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    # Test set slopes (5 slopes)
    elif data_set_index == 3:
        if slope_index == 1:
            alpha_deg = ALPHA_MAX * (
                0.28 + 0.22 * math.sin(x / 120) + 0.18 * math.cos(x / 90)
            )
        elif slope_index == 2:
            alpha_deg = ALPHA_MAX * (
                0.32 + 0.18 * math.sin(x / 80) + 0.2 * math.cos(x / 140)
            )
        elif slope_index == 3:
            alpha_deg = ALPHA_MAX * (
                0.27 + 0.23 * math.sin(x / 60) + 0.15 * math.cos(x / 120)
            )
        elif slope_index == 4:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.2 * math.sin(x / 150) + 0.2 * math.cos(x / 100)
            )
        elif slope_index == 5:
            alpha_deg = ALPHA_MAX * (
                0.25 + 0.25 * math.sin(x / 170) + 0.2 * math.cos(x / 80)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    else:
        raise ValueError(f"Invalid data_set_index {data_set_index}")

    return alpha_deg
