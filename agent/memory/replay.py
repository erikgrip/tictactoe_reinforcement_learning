from collections import deque
import random 

# Create class to store the Experience objects
class StandardReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience):
            self.memory.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide(self, min_size_to_train):
        return len(self.memory) >= min_size_to_train
    
    def size(self):
        return len(self.memory)


