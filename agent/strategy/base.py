''' Strategies are balancing exploitation and exploration. '''
import math

class BaseStratregy():
    def __init__(self, start, end, decay):
        # Basic input validation
        if (start < 0) | (end < 0) | (decay < 0) :
            raise ValueError("Only positive arguments accepted")
        elif start < end:
            raise ValueError("End must not be greater than start")

        self.start = start
        self.end = end
        self.decay = decay
        self.current_step = 0 
        
    def get_decayed_rate(self):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * self.current_step * self.decay)