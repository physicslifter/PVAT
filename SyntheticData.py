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

#define model function and pass independant variables x and y as a list
def twoD_Gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

def gaussian_2d(x, y, amplitude, xo, yo, sigma_x, sigma_y):
    """
    Calculates the value of a 2D Gaussian function at given coordinates.

    Args:
        coords (tuple): A tuple (x, y) representing the coordinates where
                        the Gaussian value is to be calculated.
        amplitude (float): The peak value of the Gaussian.
        mu_x (float): The x-coordinate of the center of the Gaussian.
        mu_y (float): The y-coordinate of the center of the Gaussian.
        sigma_x (float): The standard deviation in the x-direction.
        sigma_y (float): The standard deviation in the y-direction.

    Returns:
        numpy.ndarray: The value of the 2D Gaussian at the given coordinates.
    """
    exponent = -(((x - xo)**2) / (2 * sigma_x**2) + ((y - yo)**2) / (2 * sigma_y**2))
    return amplitude * np.exp(exponent)

class SyntheticData:
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

class SyntheticBeamCalibration(SyntheticData):
    def __init__(self, 
                 sweep_speed, 
                 slit_size, 
                 time_points, 
                 space_points
                 ):
       super().__init__(sweep_speed, slit_size, time_points, space_points)

    def generate_background(self, amp):
        """
        Generates a background with some noise for the shot
        """
        self.background = amp*(np.random.random([self.space_points, self.time_points])+0.5)
        self.data += self.background

    def generate_beam(self, center_time, pulse_width, amplitude, max_loc, shift=0):
        """
        Generates a line for the synthetic beam to follow

        shift is the slope of the shift line (time/space)
        """
        self.has_beam_line = True
        std_dev = pulse_width/(2*(2*np.log(2))**0.5)
        blank_line = np.zeros(self.time_points)
        beam_lines = []
        skip_val = self.space_points/100
        count = 1
        self.beam_center = center_time
        for i in self.space:
            if i < max_loc:
                if count == 1:
                    line_data = gaussian(x = self.time, 
                                         a = amplitude + np.random.normal()*amplitude/10, 
                                         b = center_time + +shift*(self.space.max() - i) + np.random.normal()*self.time_resolution*self.time_points/100, 
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
        
    def generate_fiducial(self, timing_offset, space_loc, amp, width, height):
        """
        Generates a fiducial for the shot, with a certain offset for the fiducial from the drive beam
        The fiducial will appear at the specified time/space
        """
        sigma_x = width/(2*(2*np.log(2))**0.5)
        sigma_y = height/(2*(2*np.log(2))**0.5)
        fiducial_lines = []
        for i in self.space:
            fiducial_line = gaussian_2d(self.time,
                                     i,
                                     amplitude = amp,
                                     xo = self.beam_center + timing_offset,
                                     yo = space_loc,
                                     sigma_x = sigma_x,
                                     sigma_y = sigma_y
                                     )
            fiducial_lines.append(fiducial_line)
        #Add the fiducial to the data
        fiducial = np.vstack(fiducial_lines)
        self.data += fiducial

class SyntheticShot:
    """
    synthetic shot data from JLF
    """
    

    def generate_data(self):
        """
        Generates synthetic data for the setup
        """
        pass

