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

    # Training set slopes (10 slopes)
    if data_set_index == 1:
        if slope_index == 1:
            alpha_deg = ALPHA_MAX * (
                0.25 + 0.35 * (x / SLOPE_LENGTH) + 0.15 * math.sin(x / 150)
            )
        elif slope_index == 2:
            alpha_deg = ALPHA_MAX * (
                0.6 - 0.4 * (x / SLOPE_LENGTH) + 0.2 * math.sin(x / 60)
            )
        elif slope_index == 3:
            alpha_deg = ALPHA_MAX * (0.5 + 0.35 * math.sin(2 * math.pi * x / 400))
        elif slope_index == 4:
            alpha_deg = ALPHA_MAX * (
                0.4 + 0.25 * math.sin(x / 80) + 0.2 * math.cos(math.sqrt(2) * x / 150)
            )
        elif slope_index == 5:
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.35 * math.exp(-(((x - 500.0) / 150.0) ** 2))
            )
        elif slope_index == 6:
            alpha_deg = ALPHA_MAX * (
                0.45 + 0.25 * math.sin(2.5 * x / 50) + 0.1 * math.cos(x / 30)
            )
        elif slope_index == 7:
            alpha_deg = ALPHA_MAX * (
                0.35 + 0.3 * (math.sin(x / 300) ** 3) + 0.15 * math.cos(x / 120)
            )
        elif slope_index == 8:
            alpha_deg = ALPHA_MAX * (
                0.2 + 0.35 * math.sin(x / 90) + 0.25 * (x / SLOPE_LENGTH)
            )
        elif slope_index == 9:
            alpha_deg = ALPHA_MAX * (
                0.4 + 0.2 * math.sin(x / 110) + 0.2 * math.sin(x / 40)
            )
        elif slope_index == 10:
            alpha_deg = ALPHA_MAX * (
                0.25
                + 0.25 * math.exp(-(((x - 300.0) / 120.0) ** 2))
                + 0.25 * math.exp(-(((x - 700.0) / 120.0) ** 2))
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    # Validation set slopes (5 slopes) - complementary, still depend on x and are within ALPHA_MAX
    elif data_set_index == 2:
        if slope_index == 1:
            # slow upward trend with medium wiggles
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.4 * (x / SLOPE_LENGTH) + 0.2 * math.sin(x / 70)
            )
        elif slope_index == 2:
            # predominantly sinusoidal with varying amplitude
            alpha_deg = ALPHA_MAX * (
                0.45 + 0.35 * math.sin(2 * math.pi * x / 300) * math.cos(x / 200)
            )
        elif slope_index == 3:
            # two localized bumps (different widths)
            alpha_deg = ALPHA_MAX * (
                0.2
                + 0.35 * math.exp(-(((x - 250.0) / 80.0) ** 2))
                + 0.25 * math.exp(-(((x - 650.0) / 140.0) ** 2))
            )
        elif slope_index == 4:
            # gentle oscillation plus a small decreasing ramp
            alpha_deg = ALPHA_MAX * (
                0.5 - 0.25 * (x / SLOPE_LENGTH) + 0.25 * math.sin(x / 55)
            )
        elif slope_index == 5:
            # asymmetric slow variation using a shifted sine^2
            alpha_deg = ALPHA_MAX * (
                0.35 + 0.3 * (math.sin((x + 120) / 220) ** 2) + 0.15 * math.cos(x / 95)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    # Test set slopes (5 slopes) - novel shapes to evaluate generalization
    elif data_set_index == 3:
        if slope_index == 1:
            # low baseline with medium-frequency modulation
            alpha_deg = ALPHA_MAX * (
                0.28 + 0.32 * math.sin(x / 85) + 0.25 * math.cos(math.sqrt(7) * x / 50)
            )
        elif slope_index == 2:
            # slowly increasing with a late bump
            alpha_deg = ALPHA_MAX * (
                0.3
                + 0.35 * (x / SLOPE_LENGTH)
                + 0.2 * math.exp(-(((x - 800.0) / 80.0) ** 2))
            )
        elif slope_index == 3:
            # alternating small/large ripples (challenging)
            alpha_deg = ALPHA_MAX * (
                0.4 + 0.2 * math.sin(x / 40) + 0.2 * math.sin(x / 180)
            )
        elif slope_index == 4:
            # single broad bump near start
            alpha_deg = ALPHA_MAX * (
                0.3 + 0.45 * math.exp(-(((x - 150.0) / 180.0) ** 2))
            )
        elif slope_index == 5:
            # mix of slow trend and high-frequency detail
            alpha_deg = ALPHA_MAX * (
                0.35 + 0.25 * (x / SLOPE_LENGTH) + 0.2 * math.sin(3 * x / 60)
            )
        else:
            raise ValueError(
                f"Invalid slope_index {slope_index} for data_set_index {data_set_index}"
            )

    else:
        raise ValueError(f"Invalid data_set_index {data_set_index}")

    return alpha_deg
