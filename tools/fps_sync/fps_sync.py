"""
source: https://stackoverflow.com/questions/43761004/fps-how-to-divide-count-by-time-function-to-determine-fps
"""
import time
import collections

class fps_sync:
    def __init__(self,avarageof=10000):
        self.frametimestamps = collections.deque(maxlen=avarageof)
        self.frametimestamps.append(time.time())

    # def __call__(self):
    #     self.frametimestamps.append(time.time())
    #     if(len(self.frametimestamps) > 1):
    #         return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
    #     else:
    #         return 0.0

    def update(self):
        self.frametimestamps.append(time.time())

    def get_FPS(self):
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0
        
    