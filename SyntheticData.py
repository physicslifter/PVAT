"""
Tools for generating synthetic data
"""
import numpy as np

def gaussian(x, # x data for the peak
             a, # amplitude of the peak
             b, # center of the peak
             c, # std dev
             background, # background amplitude
             ):
  return a * np.exp(-(x - b)**2 / (2 * c**2)) + background

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
        self.data = np.zeros((self.space_points, self.time_points))
        self.generate_time()
        self.generate_space()

    def generate_time(self):
        self.time = np.linspace(0, self.sweep_speed, self.time_points)
        self.time_resolution = self.time/self.time_points

    def generate_space(self):
        self.space = np.linspace(0, self.slit_size, self.space_points)

    def generate_background(self, amp):
        """
        Generates a background with some noise for the shot
        """
        self.background = np.random.random([self.space_points, self.time_points])
        self.data += self.background

    def generate_beam(self, center_time, pulse_width, amplitude, max_loc):
        """
        Generates a line for the synthetic beam to follow
        """
        self.has_beam_line = True
        std_dev = pulse_width/(2*(2*np.log(2))**0.5)
        blank_line = np.zeros(self.time_points)
        beam_lines = []
        skip_val = self.space_points/100
        count = 1
        for i in self.space:
            if i < max_loc:
                if count == 1:
                    line_data = gaussian(x = self.time, 
                                         a = amplitude + np.random.normal()*amplitude/10, 
                                         b = center_time + np.random.normal()*self.time_resolution*self.time_points/50, 
                                         c = std_dev, 
                                         background = 0)
                else:
                    line_data = line_data
            else:
                line_data = blank_line
            beam_lines.append(line_data)
            if count < skip_val:
                count += 1
            if count == skip_val:
                count = 1
        beam_data = np.vstack(beam_lines)
        self.data += beam_data
        
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

