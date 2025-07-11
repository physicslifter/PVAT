"""
Tools for generating synthetic data
"""
import numpy as np

class SyntheticBeamCalibration:
    def __init__(self, 
                 sweep_speed, 
                 slit_size, 
                 time_points, 
                 space_points
                 ):
        """
        sweep_speed: the sweep speed of the generated data
        slit_size: the slit size of the camera
        time_points: how many points to generate
        space_points: how many space points to generate
        """
        self.sweep_speed = sweep_speed
        self.slit_size = slit_size
        self.time_points = time_points
        self.space_points = space_points
        self.data = np.zeros((self.time_points, self.space_points))

    def generate_background(self, amp):
        """
        Generates a background with some noise for the shot
        """
        self.background = np.random.random([self.time_points, self.space_points])
        self.data += self.background

    def get_beam_line(self, start_time, pulse_width, amplitude):
        """
        Generates a line for the synthetic beam to follow
        """
        pass

    def generate_fiducial(self, timing_offset):
        """
        Generates a fiducial for the shot, with a certain offset for the fiducial from the drive beam
        """
        pass

    def generate_data(self):
        """
        Generates synthetic data for the setup
        """
        pass

