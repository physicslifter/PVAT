import numpy as np
from PIL import Image
import os
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import shift
from tifffile import imsave
import shutil

def gaussian(x, a, b, c, background):
  return a * np.exp(-(x - b)**2 / (2 * c**2)) + background

class GaussianPeak:
    """
    Class for handline a Gaussian fit to the peak
    """
    def __init__(self, y, x=None):
        self.x = x if type(x) != type(None) else np.arange(len(y))
        self.y = y
        self.fit()

    def fit(self):
        fit_params, _ = curve_fit(gaussian, self.x, self.y)
        self.amp, self.mean, self.std_dev, self.background = fit_params
        self.fit_y = gaussian(self.x, self.amp, self.mean, self.std_dev, self.background)

class ImageCorrection:
    """
    Class for storing a correction to an image
    """
    def __init__(self, fname = None):
        self.fname = fname
        if fname != None:
            self.open(fname)
        
    def create(self, space, time_shift):
        self.space = space #space axis for the image
        self.time_shift = time_shift #how much to shift in time for a certain space
        self.spacing = self.space[1] - self.space[0]
        self.maxes = np.array(self.space)
        self.mins = self.maxes + self.spacing

    def open(self, fname):
        """
        Opens an existing image correction
        """
        try:
            data = pd.read_csv(fname)
            self.space = data.space
            self.time_shift = data.time_shift
        except:
            raise Exception("File {fname} couldn't be found or was stored in incorrect format")

    def save(self, fname=None):
        if fname == None:
            if self.fname == None:
                raise Exception("Nowhere to save the file")
            else:
                fname = self.fname
        if fname.split(".")[-1].lower() != "csv":
            raise Exception("Must save as a csv file")
        data = pd.DataFrame({"space": self.space, "time_shift":self.time_shift})
        data.to_csv(fname)

    def get_inverse(self):
        """
        Inverses the correction
        """
        self.time_shift = -self.time_shift

    def plot(self, ax=None, flip:bool = False):
        """
        ax specifies an axis
        flip plots space vs time (default is time vs space)
        """
        if ax != None:
            if flip == False:
                ax.plot(self.space, self.time_shift)
            if flip == True:
                ax.plot(self.time_shift, self.space)
        else:
            plt.plot(self.space, self.time)
            name = self.fname.split("/")[-1].split(".")[0]
            plt.title(f"Correction: {name}")
            plt.xlabel("Dist. from slit bottom")
            plt.ylabel("Time Shift (ns)")

class VISARImage:
    """
    A simpler & more general class for the VISAR Image
    """
    def __init__(self, fname:str=None, data:np.array=None, sweep_speed:int=20, slit_size:int=500):
        self.fname = fname
        self.data = data
        self.sweep_speed = sweep_speed
        self.slit_size = slit_size
        self.time_aligned = False
        self.has_data = False
        self.get_data()
        self.align_space()
        self.align_time() #Default time. You can pass in a new time calibration later

    def get_data(self):
        if self.fname == None and type(self.data) == type(None): #if neither file nor data has been passed in
            raise Exception("VISARImage requires either a filename or data")
        elif self.fname != None and type(self.data) != type(None): #if both a filename and data are passed in
            if self.has_data == False:
                raise Exception("Cannot pass in both data and a tif file. Specify one.")
        elif self.fname != None and type(self.data) == type(None): #If we have a file for the data
            tif = Image.open(self.fname)
            self.data = np.array(tif).T
            self.has_data = True
        elif self.fname == None and type(self.data) != type(None): #if we have data for the shot
            self.data = self.data

    def align_time(self, time=None):
        if time != None:
            self.time = time
        else:
            self.time = np.linspace(0, self.sweep_speed, self.data.shape[1])
        #time resolution assuming 
        self.time_resolution = self.time[1] - self.time[0]
        self.time_aligned = True

    def align_space(self):
        self.space = np.linspace(0, self.slit_size, self.data.shape[0])
        self.space_aligned = True
        self.space_per_pixel = self.space[1] - self.space[0]

    def take_lineout(self, min, max):
        """
        takes horizontal lineout between min and max values
        """
        min = int(min/self.space_per_pixel)
        max = int(max/self.space_per_pixel)
        self.lineout = self.data[min:max].mean(axis = 0)
        return self.lineout
    
    def take_chunk(self, min_val, max_val):
        """
        Takes a horizontal chunk of data, but unlike lineout it doesn't flatten to the average
        """
        at_max_space = True if max_val >= max(self.space) else False
        min_val = int(min_val/self.space_per_pixel)
        max_val = int(max_val/self.space_per_pixel)
        max_val = max_val + 1 if at_max_space == True else max_val
        print(min_val, max_val)
        chunk = self.data[min_val:max_val]
        return chunk
    
    def fit_lineout(self, type="gaussian"):
        """
        Fits a peak to the lineout. 
        Currently only does Gaussian
        """
        self.fit = GaussianPeak(self.lineout, x = self.time)
    
    def plot_fit(self, ax):
        ax.plot(self.time, self.fit.fit_y, c = "magenta", label = "Peak Fit")
        ax.plot([self.time[0], self.time[-1]], [self.fit.background, self.fit.background], color = "green", label = "Background Fit")

    def update_heatmap_threshold(self, vmin, vmax):
        if self.plotted == False:
            raise Exception("Data not yet plotted")
        self.visar_mesh.set_clim(vmin = vmin, vmax = vmax)

    def show_data(self, ax, minmax=(300, 4000)):
        if self.has_data == False:
            self.get_data()
        if self.time_aligned == False:
            self.align_time()
        if self.space_aligned == False:
            self.align_space()
        data = self.data
        vmin, vmax = minmax
        self.vmin = vmin
        self.vmax = vmax
        X, Y = np.meshgrid(self.time, self.space)
        self.visar_mesh = ax.pcolormesh(X, Y, data, norm = colors.LogNorm(vmin=vmin,vmax=vmax, clip=True), cmap='magma')
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Dist from slit bottom (um)")
        self.plotted = True

    def save_tif(self, save_name):
        #saves the current image as a tif file
        if save_name.split(".")[-1] != "tif":
            raise Exception("Must save as a .tif file")
        print(type(self.data))
        imsave(save_name, self.data.astype(np.float32).T)

    def set_time_to_zero(self, time):
        """
        Given a time, this function adjusts the time axis so that 0 is at the specified time on the current axis
        """
        self.time += -time

    def chop_by_time(self, min_time, max_time):
        """
        removes data before min_time and after max_time
        """
        min_index = int(min_time/self.time_resolution)
        max_index = int(max_time/self.time_resolution)
        self.data = self.data[:, min_index:max_index]
        self.time = self.time[min_index:max_index]

    def apply_correction(self, correction:ImageCorrection, negative = False):
        """
        Given an ImageCorrection, this function corrects the image accordingly
        
        if negative == True, apply the opposite correction
        """
        #Divide image into chunks according to the correction
        c = 0
        #Get any data before the correction
        if negative == True:
            correction.get_inverse()
        initial_chunk = self.take_chunk(min_val = max(correction.maxes), max_val = max(self.space))
        #get any data after the correction
        final_chunk = self.take_chunk(min_val = min(self.space), max_val = min(correction.mins))
        chunks = []
        max_pos_shift_ns = max(correction.time_shift)
        min_neg_shift_ns = min(correction.time_shift)
        min_neg_shift = int(min_neg_shift_ns/self.time_resolution) if min_neg_shift_ns < 0 else 0
        max_pos_shift = int(max_pos_shift_ns/self.time_resolution) if max_pos_shift_ns > 0 else 0
        #Correct initial and final chunks to be the proper lengths
        initial_chunk = np.pad(initial_chunk, ((0, 0), (np.abs(min_neg_shift), max_pos_shift)), constant_values = np.NaN)
        final_chunk = np.pad(final_chunk, ((0, 0), (np.abs(min_neg_shift), max_pos_shift)), constant_values = np.NaN)
        #start chunks w/ initial chunk
        chunks.append(final_chunk)
        #get bounding boxes for the chunks so that no data is skipped
        for max_val, min_val, time_shift in zip(correction.maxes[::-1], correction.mins[::-1], correction.time_shift[::-1]):
            #print(min_val, max_val)
            chunk = self.take_chunk(min_val = min_val, max_val = max_val)
            
            #shift the chunk accordingly
            shift_val = int(time_shift/self.time_resolution)
            if shift_val < 0:
                chunk = np.pad(chunk, ((0, 0), (0, np.abs(shift_val))), constant_values = np.NaN)
            elif shift_val > 0:
                chunk = np.pad(chunk, ((0, 0), (shift_val, 0)), constant_values = np.NaN)
            
            #Append end & beginning to make all chunks uniform
            if shift_val < 0:
                if shift_val > min_neg_shift:
                    diff = np.abs(shift_val - min_neg_shift)
                    chunk = np.pad(chunk, ((0, 0), (diff, max_pos_shift)), constant_values = np.NaN)
                else: #if the minimum val
                    chunk = np.pad(chunk, ((0, 0), (0, max_pos_shift)), constant_values = np.NaN)
            elif shift_val > 0:
                if shift_val < max_pos_shift:
                    diff = np.abs(shift_val - max_pos_shift)
                    chunk = np.pad(chunk, ((0, 0), (np.abs(min_neg_shift), diff)), constant_values = np.NaN)
                else: #if the max val
                    chunk = np.pad(chunk, ((0, 0), (np.abs(min_neg_shift), 0)), constant_values = np.NaN)
            else: #If there's no shift
                chunk = np.pad(chunk, ((0, 0), (np.abs(min_neg_shift), max_pos_shift)), constant_values = np.NaN)
            chunks.append(chunk)
            c += 1
        
        #Add final chunk in
        chunks.append(initial_chunk)
        for chunk in chunks:
            pass
            #print(chunk)
        #Put chunks back together to get new data
        self.data = np.vstack(chunks)
        #get new time axis
        self.data = np.nan_to_num(self.data, nan = 0)
        new_min = min(self.time) + min_neg_shift*self.time_resolution
        new_max = max(self.time) + max_pos_shift*self.time_resolution
        self.time = np.linspace(new_min, new_max, self.data.shape[1])
        if negative == True: #flip correction back if we've flipped it
            correction.get_inverse()

class RefImage:
    def __init__(self, fname:str=None, folder:str=None, sweep_speed:int = 20, slit_size:int = 500):
        self.fname = fname
        self.folder = folder
        self.sweep_speed = sweep_speed
        self.slit_size = slit_size
        self.get_data()
        self.beam_chopped = False
        self.has_correction = False

    def get_saved_data(self):
        """
        function for getting data from a saved RefImage
        """
        pass

    def get_data(self):
        """
        folder and data are both always specified
        1. check folder to see if data exists
        2. if overwrite == True, treat the data folder as if it's empty
        """
        self.img = VISARImage(fname = self.fname, sweep_speed = self.sweep_speed, slit_size = self.slit_size)
        if type(self.folder) != type(None):
            print(os.path.exists(self.folder))
            if not os.path.exists(f"{self.folder}"):
                os.mkdir(self.folder)

    def show_raw_visar(self, minmax = None):
        fig = plt.subplots()
        ax = fig.add_subplot(1, 1, 1)
        if minmax == None:
            self.img.show_data(ax, minmax)
        
    def chop_beam(self, ybounds, num_slices):
        """
        function for chopping up the drive beam to see if a drift is present

        xbounds: tuple containing (min, max) of the x values for the beam
        ybounds: tuple containing (min, max) of the y values for the beam
        num_slices: the number of slices to chop the beam into
        """
        self.slices = []
        self.fit_centers = []
        self.fits = []
        ymin = ybounds[0]#int(ybounds[0]/self.img.space_per_pixel)
        ymax = ybounds[1]#int(ybounds[1]/self.img.space_per_pixel)
        self.time_shifts = []
        self.locs = []
        slice_width = (ymax - ymin)/num_slices
        loc = ymax
        for slice in range(num_slices):
            slice_min = loc - slice_width
            slice_max = loc
            slice_data = self.img.take_lineout(slice_min, slice_max)
            self.img.fit_lineout()
            fit = self.img.fit
            self.fits.append(fit.fit_y)
            center = fit.mean
            if slice == 0:
                base_time = center
            self.time_shifts.append(center - base_time)
            self.locs.append(loc)
            self.fit_centers.append(center)
            self.slices.append(slice_data)
            loc = slice_min
        self.beam_chopped = True

    def check_beam_chop(self):
        if self.beam_chopped == False:
            raise Exception("No chopping has been performed")

    def save_chop_as_correction(self):
        self.correction = ImageCorrection()
        if self.beam_chopped == False: #if beam hasn't been chopped, save a zero time shift correction
            self.correction.create(space = self.img.space, time_shift = np.zeros(self.img.space))
        else: #if beam has been chopped, save the most recent correction
            self.correction.create(space = self.locs, time_shift = -np.array(self.time_shifts))
        if type(self.folder) != type(None):
            self.correction.save(fname = f"{self.folder}/correction.csv")
        self.has_correction = True

    def plot_chop(self, minmax, savename:str=None, show:bool=True):
        """
        Shows the chopping, the peak lineouts, and the measured fit locations
        as a function of spatial position
        """
        self.check_beam_chop()
        fig = plt.figure(figsize = (5, 8))
        ax1 = fig.add_subplot(2, 1, 1) #Showing the chopped beam
        ax2 = fig.add_subplot(4, 1, 3, sharex = ax1) #Showing the lineouts for the chops
        ax3 = fig.add_subplot(4, 1, 4) #showing the changing location
        
        self.img.show_data(ax = ax1, minmax = minmax) #put data on the 1st axis
        #chop up the graph accordingly
        for loc in self.locs:
            ax1.axhline(loc)

        #put lineouts on the graph
        for loc, slice, center in zip(self.locs, self.slices, self.fit_centers):
            ax2.plot(self.img.time, slice, label = loc)
            ax2.vlines(center, slice.min(), slice.max())

        #put the changing location on the plot
        if self.has_correction == False:
            self.save_chop_as_correction()
        self.correction.plot(ax3)

        ax1.set_title(f"{self.fname.split('/')[-1]}\nChopped VISAR Data")
        ax2.set_title("Peak lineouts")
        #ax2.legend(ncol = 2)
        ax3.set_title("Correction")
        fig.tight_layout()
        if savename != None:
            plt.savefig(savename)
        if show == True:
            plt.show()

    def take_lineouts(self, fiducial_min, fiducial_max, beam_min, beam_max):
        """
        Given bounds for the beam and fiducial lineouts, this function
        saves the data to the proper folder
        """
        self.fiducial_lineout = self.img.take_lineout(fiducial_min, fiducial_max)
        self.beam_lineout = self.img.take_lineout(beam_min, beam_max)
        
    def save_lineouts(self):
        df = pd.DataFrame({"time":self.img.time, "fiducial":self.fiducial_lineout, "beam":self.beam_lineout})
        df.to_csv(f"{self.folder}/lineouts.csv")
        print("Saved")

    def delete_folder(self):
        """
        Delete the folder for the reference
        """
        shutil.rmtree(self.folder)
        

def get_ref(folder):
    """
    Given a reference folder, this function returns the saved lineouts and correction from the function
    """
    pass

class TimingRef:
    """
    holds a timing reference
    """
    def __init__(self, name):
        self.name = name
