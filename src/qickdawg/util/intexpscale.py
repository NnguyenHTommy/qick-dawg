import numpy as np


def int_exp_scale(start, end, scaling_factor):
    '''
    Function that generates a exponentially scaled array from 'start' to
    'end' by 'scaling_factor'
    ---------------------------------------------------------------------
    Parameters
        start
            int value for the start of the array
        end
            int value for the end of the array
        scaling_factor
            str that must be '3/2', '5/4', '9/8', '17/16'
            that determines the scaling between one point in the 
            array and the next

    The final point is the first value at or above end, and can overshoot it.
    Raises ValueError if the integer recurrence cannot advance toward end.
    '''

    if scaling_factor not in ('3/2', '5/4', '9/8', '17/16'):
        raise ValueError("scaling_factor must be '3/2', '5/4', '9/8', or '17/16'")

    points = [start]

    numerator, denominator = scaling_factor.split('/')
    numerator = int(numerator)
    denominator = int(denominator)

    shift = int(np.log2(denominator))
    while points[-1] < end:
        x = int(points[-1])
        next_x = x + (x >> shift)
        if next_x <= x:
            raise ValueError(
                "Exponential sweep cannot advance from start={} with factor={}".format(
                    start, scaling_factor))
        points.append(next_x)

    return np.array(points)
