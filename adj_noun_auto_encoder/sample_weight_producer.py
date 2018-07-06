from config import *
import math


class WeightTypes(object):
    COUNT = "count"
    UNIFORM = "uniform"
    LOG_COUNT = "log"


def get_weight(count):
    weight = 1.0
    if SAMPLE_WEIGHT_TYPE == WeightTypes.COUNT:
        weight = float(count)
    elif SAMPLE_WEIGHT_TYPE == WeightTypes.UNIFORM:
        weight = 1.0
    elif SAMPLE_WEIGHT_TYPE == WeightTypes.LOG_COUNT:
        weight = math.log(count, 10)
    return weight


