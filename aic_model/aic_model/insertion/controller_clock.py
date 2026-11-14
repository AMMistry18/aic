"""Physical motion clock, independent of the overall wall-clock deadline."""
import os
import time


def control_time(policy,wall_time=None):
    if os.environ.get('AIC_INSERT_CONTROLLER_SIM_TIME','0')=='1':
        return policy._parent_node.get_clock().now().nanoseconds*1e-9
    return (time.monotonic if wall_time is None else wall_time)()
