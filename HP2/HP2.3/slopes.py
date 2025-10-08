import math

# This file provides the FORMAT you should use for the
# slopes in HP2.3. x denotes the horizontal distance
# travelled (by the truck) on a given slope, and
# alpha measures the slope angle at distance x


def get_slope_angle(x, slope_index, data_set_index):
    # Default value for any undefined combinations
    alpha = 4.0  # reasonable default downhill angle

    if data_set_index == 0:  # Added support for data_set_index=0
        if slope_index == 0:
            alpha = 3.5 + math.sin(x / 200)
        elif slope_index == 1:
            alpha = 4.0 + 2.0 * math.sin(x / 300)
        else:
            alpha = 3.0 + math.cos(x / 150)

    elif data_set_index == 1:  # Training
        if slope_index == 1:
            alpha = 4 + math.sin(x / 100) + math.cos(math.sqrt(2) * x / 50)
        elif slope_index == 2:
            alpha = 3.5 + math.sin(x / 120) + math.cos(x / 80)
        elif slope_index == 3:
            alpha = 5.0 + math.sin(x / 90) + 0.5 * math.cos(x / 70)
        elif slope_index == 4:
            alpha = 4.5 - 0.5 * math.sin(x / 110) + math.cos(x / 60)
        elif slope_index == 5:
            alpha = 3.8 + math.sin(x / 130) - 0.6 * math.cos(x / 90)
        elif slope_index == 6:
            alpha = 4.2 + 0.7 * math.sin(x / 140) + 0.3 * math.cos(x / 100)
        elif slope_index == 7:
            alpha = 5.2 - 0.4 * math.sin(x / 150) + 0.8 * math.cos(x / 110)
        elif slope_index == 8:
            alpha = 3.6 + 0.9 * math.sin(x / 160) - 0.2 * math.cos(x / 120)
        elif slope_index == 9:
            alpha = 4.8 - 0.3 * math.sin(x / 170) + 0.5 * math.cos(x / 130)
        elif slope_index == 10:
            alpha = 3 + 2 * math.sin(x / 50) + math.cos(math.sqrt(2) * x / 100)

    elif data_set_index == 2:  # Validation
        if slope_index == 1:
            alpha = 6 - math.sin(x / 100) + math.cos(math.sqrt(3) * x / 50)
        elif slope_index == 2:
            alpha = 5.5 - 0.5 * math.sin(x / 120) + 0.8 * math.cos(x / 80)
        elif slope_index == 3:
            alpha = 4.8 + 0.7 * math.sin(x / 90) - 0.3 * math.cos(x / 70)
        elif slope_index == 4:
            alpha = 5.2 - 0.4 * math.sin(x / 110) - 0.5 * math.cos(x / 60)
        elif slope_index == 5:
            alpha = 5 + math.sin(x / 50) + math.cos(math.sqrt(5) * x / 50)

    elif data_set_index == 3:  # Test
        if slope_index == 1:
            alpha = 6 - math.sin(x / 100) + math.cos(math.sqrt(7) * x / 50)
        elif slope_index == 2:
            alpha = 5.8 + 0.6 * math.sin(x / 130) - 0.4 * math.cos(x / 90)
        elif slope_index == 3:
            alpha = 6.2 - 0.5 * math.sin(x / 140) + 0.7 * math.cos(x / 100)
        elif slope_index == 4:
            alpha = 5.5 + 0.8 * math.sin(x / 150) - 0.2 * math.cos(x / 110)
        elif slope_index == 5:
            alpha = 4 + (x / 1000) + math.sin(x / 70) + math.cos(math.sqrt(7) * x / 100)

    # Ensure the angle is positive (downhill)
    return max(0.5, alpha)
