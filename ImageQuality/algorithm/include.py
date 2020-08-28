import cv2
import numpy as np
import math
import os
import time
def calculate_function_run_time_ms(func):
    def call_fun(*args, **kwargs):
        start_time = time.time()
        f = func(*args, **kwargs)
        end_time = time.time()
        print('%s() run time：%s ms' % (func.__name__, int(1000*(end_time - start_time))))
        return f,1000*(end_time - start_time)
    return call_fun



