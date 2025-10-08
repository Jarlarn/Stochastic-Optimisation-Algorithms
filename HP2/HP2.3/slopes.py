import math

# This file provides the FORMAT you should use for the
# slopes in HP2.3. x denotes the horizontal distance
# travelled (by the truck) on a given slope, and
# alpha measures the slope angle at distance x
#
# iSlope denotes the slope index (i.e. 1,2,..10 for the
# training set etc.)
# iDataSet determines whether the slope under consideration
# belongs to the training set (data_set_index = 1), validation
# set (data_set_index = 2), or the test set (data_set_index = 3).
#
# Note that the slopes given below are just EXAMPLES.
# Please feel free to implement your own slopes below,
# as long as they fulfil the criteria given in HP2.3.
#
# You may remove the comments above and below, as they
# (or at least some of them) violate the coding standard
#  a bit. :)
# The comments have been added as a clarification of the
# problem that should be solved!).


def get_slope_angle(x, slope_index, data_set_index):

    if data_set_index == 1:  # Training
        if slope_index == 1:
            alpha = (
                4 + math.sin(x / 100) + math.cos(math.sqrt(2) * x / 50)
            )  # You may modify this!

        #
        # Here, insert the next 8 training set slopes (defined by you)
        #

        elif slope_index == 10:
            alpha = 3 + 2 * math.sin(x / 50) + math.cos(math.sqrt(2) * x / 100)
            # You may modify this!

    elif data_set_index == 2:  # Validation
        if slope_index == 1:
            alpha = 6 - math.sin(x / 100) + math.cos(math.sqrt(3) * x / 50)

        #
        # Here, insert the next 3 validation set slopes (defined by you)
        #

        elif slope_index == 5:
            alpha = 5 + math.sin(x / 50) + math.cos(math.sqrt(5) * x / 50)
            # You may modify this!

    elif data_set_index == 3:  # Test
        if slope_index == 1:
            alpha = (
                6 - math.sin(x / 100) + math.cos(math.sqrt(7) * x / 50)
            )  # You may modify this!

        #
        # Here, insert the next 3 test set slopes (defined by you)
        #

        elif slope_index == 5:
            alpha = (
                4 + (x / 1000) + math.sin(x / 70) + math.cos(math.sqrt(7) * x / 100)
            )  # You may modify this!

    return alpha
