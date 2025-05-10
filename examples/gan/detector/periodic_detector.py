"""
Detectors that triggers re-training on periodic basis.
"""
import logging
import numpy as np

from auto_retraining.retraining.detector import Detector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PeriodicDetector(Detector):
    def __init__(self, period: int):
        self.period = period    
        self.start_timestamp = None
        self.current_verdict = False

    def new_data(self, data):
        if len(data) == 0:
            return
        timestamp = np.timedelta64(int(data["Timestamp"].values[0]), 's')
        
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        elif((timestamp - self.start_timestamp) > np.timedelta64(self.period, 'D')):
            logger.info(f"PeriodicDetector verdict: retrain at {int(data['Timestamp'].values[0])}")
            self.current_verdict = True
    
    def reset(self):
        self.current_verdict = False
        self.start_timestamp = None

    def verdict(self):
        return self.current_verdict
