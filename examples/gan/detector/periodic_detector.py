"""
Detectors that triggers re-training on periodic basis.
"""
import numpy as np

from framework.retraining.detector import Detector


class PeriodicDetector(Detector):
    def __init__(self, period: int):
        self.period = period    
        self.start_timestamp = None
        self.current_verdict = False

    def new_data(self, new_data):
        if len(new_data) == 0:
            return
        timestamp = new_data["Timestamp"].values[0]
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        elif(timestamp - self.start_timestamp > np.timedelta64(self.period, 'D')):
            self.current_verdict = True
    
    def reset(self):
        self.current_verdict = False
        self.start_timestamp = None

    def verdict(self):
        return self.current_verdict
