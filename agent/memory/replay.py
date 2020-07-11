from collections import deque
import random 

# Create class to store the Experience objects
class StandardReplayMemory():
    def __init__(self, capacity, min_size_for_sampling):
        self.memory = deque(maxlen=capacity)
        self.min_size = min_size_for_sampling
        
    def push(self, experience):
            self.memory.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide(self):
        return len(self.memory) >= self.min_size


