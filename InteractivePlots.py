"""
Interactive plots for performing analyses
"""
from VISAR import *
from matplotlib import pyplot as plt
from matplotlib.widgets import RangeSlider, Slider, Button, TextBox
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from scipy import ndimage
from scipy.fft import rfft, ifft, rfftfreq

plt.style.use(["fast"]) #fast plotting

class BeamAligner:
    """
    Interactive plot for aligning the beam
    """
    def __init__(self, ref_img:RefImage):
        self.ref = ref_img
        self.name = f"{ref_img.fname.split('/')[-1].lower().replace('.tif', '')}"
        self.showing_visar = False
        self.showing_lineouts = False
        self.has_lineout_bounds = False
        self.has_chop = False
        self.correction_save_name = None
        self.timing_save_name = None
        #self.ref.img.data = ndimage.uniform_filter(self.ref.img.data, size = (int(len(self.ref.img.space)/50), int(len(self.ref.img.time)/50)))
        #self.ref.img.data = ndimage.minimum_filter(self.ref.img.data, size = (int(len(self.ref.img.space)/50), int(len(self.ref.img.time)/50)))
        #self.ref.img.data[self.ref.img.data < np.median(self.ref.img.data)/2] = 0.01

    def initialize_plot(self):
        """
        Set up the plot
        """
        self.fig = plt.figure(figsize = (10,8))
        gs = GridSpec(1, 3, figure = self.fig)
        self.img_ax = self.fig.add_subplot(gs[0, :2])
        self.img_ax.set_title(self.name)
        self.lineout_ax = self.fig.add_subplot(gs[0, 2], sharex = self.img_ax)
        self.lineout_ax.set_title("Lineouts")
        #Create space for sliders
        self.fig.subplots_adjust(bottom=0.3)
        #slider axes
        self.lineout_slider_ax = self.fig.add_axes([0.2, 0.16, 0.33, 0.03])
        self.colormap_slider_ax = self.fig.add_axes([0.2, 0.12, 0.33, 0.03])
        self.chopper_slider_ax = self.fig.add_axes([0.2, 0.08, 0.33, 0.03])
        self.fiducual_lineout_slider_ax = self.fig.add_axes([0.2, 0.2, 0.33, 0.03])
        self.time_shift_slider_ax = self.fig.add_axes([0.2, 0.04, 0.33, 0.03])

        #Getting initial slider bounds
        beam_slider_min = self.ref.img.space.min() + (self.ref.img.space.max() - self.ref.img.space.min())*0.1
        beam_slider_max = self.ref.img.space.min() + (self.ref.img.space.max() - self.ref.img.space.min())*0.5
        fiducial_slider_max = int(self.ref.img.space.max() - 70*self.ref.img.space_per_pixel)
        fiducial_slider_min = fiducial_slider_max - 40*self.ref.img.space_per_pixel

        #add sliders to the plot
        self.beam_lineout_slider = RangeSlider(self.lineout_slider_ax, "Beam Lineout", self.ref.img.space.min(), self.ref.img.space.max(), valinit = [beam_slider_min, beam_slider_max])
        self.colormap_slider = RangeSlider(self.colormap_slider_ax, "Heatmap Threshold", self.ref.img.data.min(), self.ref.img.data.max())
        self.chopper_slider = Slider(self.chopper_slider_ax, "Num Chops", 1, 100, valinit = 10)
        self.fiducial_lineout_slider = RangeSlider(self.fiducual_lineout_slider_ax, "Fiducial\nLineout", self.ref.img.space.min(), self.ref.img.space.max(), valinit = [fiducial_slider_min, fiducial_slider_max])
        self.time_shift_slider = Slider(self.time_shift_slider_ax, "Start Time", self.ref.img.time.min(), self.ref.img.time.max(), valinit = 0)

        #Adding buttons for performing functions
        self.get_chop_ax = self.fig.add_axes([0.67, 0.2, 0.1, 0.06])
        self.apply_correction_ax = self.fig.add_axes([0.67, 0.13, 0.1,0.06])
        self.take_lineout_ax = self.fig.add_axes([0.67, 0.06, 0.1,0.06])
        self.save_time_calibration_ax = self.fig.add_axes([0.8, 0.2, 0.1,0.06])
        self.center_time_ax = self.fig.add_axes([0.8, 0.13, 0.1,0.06])

        self.get_chop_button = Button(self.get_chop_ax, "Get Chop")
        self.apply_correction_button = Button(self.apply_correction_ax, "Apply\nCorrection")
        self.take_lineout_button = Button(self.take_lineout_ax, "Take\nLineout")
        self.save_time_calibration_button = Button(self.save_time_calibration_ax, "Save Time\nCalibration")
        self.center_time_button = Button(self.center_time_ax, "Center Time")

    def plot_visar(self):
        self.ref.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.showing_visar = True

    def plot_lineouts(self):
        fiducial_lineout = self.ref.img.take_lineout(min = self.fiducial_lineout_slider.val[0], max = self.fiducial_lineout_slider.val[1])
        beam_lineout = self.ref.img.take_lineout(min = self.beam_lineout_slider.val[0], max = self.beam_lineout_slider.val[1])
        self.fiducial_lineout = self.lineout_ax.plot(self.ref.img.time, fiducial_lineout/fiducial_lineout.max(), label = "Fiducial", c = "green")[0]
        self.beam_lineout = self.lineout_ax.plot(self.ref.img.time, beam_lineout/beam_lineout.max(), label = "Beam", c = "goldenrod")[0]
        half_loc = self.get_half_max(self.ref.img.time, beam_lineout)
        self.lineout_half_label = self.lineout_ax.axhline(half_loc, xmin = self.ref.img.time.min(), xmax = self.ref.img.time.max(), label = "Drive Half Max", c = "red")
        beam_lineout = beam_lineout/beam_lineout.max()
        self.lineout_ax.legend()
        self.current_beam_lineout = beam_lineout
        self.current_fiducial_lineout = fiducial_lineout
        self.showing_lineouts = True

    def plot_lineout_bounds(self):
        self.fiducial_max = self.img_ax.axhline(self.fiducial_lineout_slider.val[1], xmin = self.ref.img.time.min(), xmax = self.ref.img.time.max(), c = "lime", label = "Fiducial Lineout")
        self.fiducial_min = self.img_ax.axhline(self.fiducial_lineout_slider.val[0], xmin = self.ref.img.time.min(), xmax = self.ref.img.time.max(), c = "lime", label = "Beam Lineout")
        self.beam_max = self.img_ax.axhline(self.beam_lineout_slider.val[1], xmin = self.ref.img.time.min(), xmax = self.ref.img.time.max(), c = "yellow")
        self.beam_min = self.img_ax.axhline(self.beam_lineout_slider.val[0], xmin = self.ref.img.time.min(), xmax = self.ref.img.time.max(), c = "yellow")
        self.zero_time_img = self.img_ax.axvline(self.time_shift_slider.val, ymin = self.ref.img.space.min(), ymax = self.ref.img.space.max())
        self.zero_time_lineout = self.lineout_ax.axvline(self.time_shift_slider.val, ymin = 0, ymax = 1)
        self.has_lineout_bounds = True

    def update_beam_slider(self, val):
        #Function for moving the beam lineout slider
        #reset the slider positions
        self.beam_min.set_ydata([val[0], val[0]])
        self.beam_max.set_ydata([val[1], val[1]])

    def update_fiducial_slider(self, val):
        self.fiducial_min.set_ydata([val[0], val[0]])
        self.fiducial_max.set_ydata([val[1], val[1]])

    def update_colormap_slider(self, val):
        self.ref.img.update_heatmap_threshold(vmin = val[0], vmax = val[1])
        self.fig.canvas.draw_idle()

    def update_time_slider(self, val): 
        self.zero_time_img.set_xdata([val, val])
        self.zero_time_lineout.set_xdata([val, val])
        self.fig.canvas.draw_idle()

    def get_half_max(self, time, lineout):
        #gets the half max of a beam profile
        lineout = lineout
        peak = GaussianPeak(x = time, y = np.nan_to_num(lineout, nan = 0))
        print(peak.background/lineout.max(), peak.amp/lineout.max(), peak.mean/lineout.max(), peak.std_dev/lineout.max())
        scaled_background = peak.background/lineout.max()
        loc = scaled_background + 0.5*(1 - scaled_background)
        return loc

    def take_lineouts(self):
        fiducial_lineout = self.ref.img.take_lineout(min = self.fiducial_lineout_slider.val[0], max = self.fiducial_lineout_slider.val[1])
        beam_lineout = self.ref.img.take_lineout(min = self.beam_lineout_slider.val[0], max = self.beam_lineout_slider.val[1])
        self.fiducial_lineout.set_xdata(self.ref.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.beam_lineout.set_xdata(self.ref.img.time)
        self.beam_lineout.set_ydata(beam_lineout/beam_lineout.max())
        beam_lineout = beam_lineout/beam_lineout.max()
        no_zeros = beam_lineout[beam_lineout != 0]
        half_loc = self.get_half_max(self.ref.img.time, beam_lineout)
        half_loc = self.get_half_max(self.ref.img.time, beam_lineout)
        self.lineout_half_label.set_xdata([self.ref.img.time.min(), self.ref.img.time.max()])
        self.lineout_half_label.set_ydata([half_loc, half_loc])
        print("Lineout Taken")
        self.current_beam_lineout = beam_lineout
        print(1)
        self.current_fiducial_lineout = fiducial_lineout
        print(1)
        self.fig.canvas.draw_idle()
        print("DONE")

    def click_take_lineout(self, val):
        self.take_lineouts()

    def click_get_chop(self, val):
        self.ref.chop_beam(ybounds = (self.beam_lineout_slider.val[0], self.beam_lineout_slider.val[1]), num_slices = int(self.chopper_slider.val))
        self.ref.plot_chop(minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))

    def click_zero_time(self, val):
        self.take_lineouts()
        self.ref.img.set_time_to_zero(self.time_shift_slider.val)
        self.ref.img.show_data(self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.time_shift_slider.valmin = self.ref.img.time.min()
        self.time_shift_slider.valmax = self.ref.img.time.max()
        self.time_shift_slider_ax.set_xlim(self.ref.img.time.min(), self.ref.img.time.max())
        self.time_shift_slider.set_val(0)
        self.fiducial_lineout.set_xdata(self.ref.img.time)
        self.beam_lineout.set_xdata(self.ref.img.time)
        self.lineout_half_label.set_xdata([self.ref.img.time.min(), self.ref.img.time.max()])
        self.img_ax.set_xlim(self.ref.img.time.min(), self.ref.img.time.max())
        self.fig.canvas.draw_idle()

    def set_correction_save_name(self, fname):
        self.correction_save_name = fname

    def set_lineout_save_name(self, fname):
        self.timing_save_name = fname

    def click_apply_correction(self, val):
        #Save the correction
        self.ref.save_chop_as_correction() #saves most recent chop as the correction
        
        #Apply the correction to the data
        self.ref.img.apply_correction(self.ref.correction)
        self.ref.img.show_data(self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))

        #update slider values
        self.time_shift_slider.valmin = self.ref.img.time.min()
        self.time_shift_slider.valmax = self.ref.img.time.max()

        self.fig.canvas.draw_idle()

    def get_lineout_save_name(self):
        if type(self.timing_save_name) == type(None):
            self.timing_save_name = f"{self.ref.folder}/lineouts.csv"

    def click_save_time_cal(self, val):
        self.get_lineout_save_name()
        print(self.timing_save_name)
        if self.timing_save_name != None:
            fiducial_lineout = self.fiducial_lineout.get_ydata()
            beam_lineout = self.beam_lineout.get_ydata()
            time = self.fiducial_lineout.get_xdata()
            df = pd.DataFrame({"time": time, "beam": beam_lineout, "fiducial": fiducial_lineout})
            df.to_csv(self.timing_save_name)

            #updates saved CSV, PNG
            #plot_filename = self.timing_save_name.replace('.csv', '.png')
            #self.fig.savefig(plot_filename) #new
        else:
            pass

    def set_sliders(self):
        """
        Setting functions to sliders
        """
        self.fiducial_lineout_slider.on_changed(self.update_fiducial_slider)
        self.beam_lineout_slider.on_changed(self.update_beam_slider)
        self.colormap_slider.on_changed(self.update_colormap_slider)
        self.time_shift_slider.on_changed(self.update_time_slider)

    def set_buttons(self):
        self.take_lineout_button.on_clicked(self.click_take_lineout)
        self.get_chop_button.on_clicked(self.click_get_chop)
        self.apply_correction_button.on_clicked(self.click_apply_correction)
        self.center_time_button.on_clicked(self.click_zero_time)
        self.save_time_calibration_button.on_clicked(self.click_save_time_cal)

    def show_plot(self):
        if self.showing_visar == False:
            self.plot_visar()
        if self.showing_lineouts == False:
            self.plot_lineouts()
        if self.has_lineout_bounds == False:
            self.plot_lineout_bounds()
        self.set_sliders()
        self.set_buttons()
        plt.show()


class ShotRefAligner:
    """
    Class for aligning a shot ref
    """
    def __init__(self, img:VISARImage):
        self.img = img
        self.name = f"{img.fname.split('/')[-1].lower().replace('.tif', '')}"
        self.showing_visar = False
        self.showing_lineout = False
        self.beam_ref = ""
        self.folder = ""
        self.has_shear_line = False
        self.sheared_angle = 0
        self.has_ref_lineout = 0
        self.beam_calibration_applied = False

    def initialize_plot(self):
        self.fig = plt.figure(figsize = (10, 8))
        gs = GridSpec(3, 1, figure = self.fig)

        #create axes
        self.img_ax = self.fig.add_subplot(gs[:2, 0])
        self.img_ax.set_title(f"Time & Space Calibration: {self.name}")
        self.lineout_ax = self.fig.add_subplot(gs[2,0])
        plt.setp(self.img_ax.get_xticklabels(), visible=False)

        #Label timing sections
        self.fig.text(0.75, 0.59, "Shearing", size = "large", weight = "bold")
        self.fig.text(0.75, 0.25, "Timing", size = "large", weight = "bold")
        self.fig.text(0.75, 0.87, "Beam\nReference", size = "large", weight = "bold")

        #create room for buttons
        self.fig.subplots_adjust(right = 0.7, bottom = 0.2)

        #Beam reference text entry
        self.ref_file_ax = self.fig.add_axes([0.72, 0.81, 0.15, 0.04])
        self.ref_file_entry = TextBox(self.ref_file_ax, "Ref\nFolder", initial = self.beam_ref)
        label = self.ref_file_entry.ax.get_children()[0]
        label.set_position([1.2, 0.9])
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')

        #create shearing button axes
        self.add_shear_button_ax = self.fig.add_axes([0.72, 0.5, 0.14, 0.07])
        self.shear_button_ax = self.fig.add_axes([0.72, 0.4, 0.14, 0.07])

        #create shearing buttons
        self.add_shear_button = Button(self.add_shear_button_ax, label = "Add Shear")
        self.shear_button = Button(self.shear_button_ax, label = "Shear")

        #create shear slider
        self.shear_slider_ax = self.fig.add_axes([0.9, 0.3, 0.03, 0.25])
        self.shear_slider = Slider(ax = self.shear_slider_ax, label = "Shear\nAngle", valmin = -3, valmax = 3, valinit = 0, orientation = "vertical")

        #bottom sliders
        self.colormap_slider_ax = self.fig.add_axes([0.15, 0.1, 0.45, 0.03])
        self.time_shift_slider_ax = self.fig.add_axes([0.15, 0.06, 0.45, 0.03])
        self.fiducial_slider_ax = self.fig.add_axes([0.15, 0.02, 0.45, 0.03])
        self.colormap_slider = RangeSlider(self.colormap_slider_ax, "Heatmap\nThreshold", self.img.data.min(), self.img.data.max())
        self.time_shift_slider = Slider(self.time_shift_slider_ax, "Time Shift", -self.img.sweep_speed/2, self.img.sweep_speed/2, valinit = 0)
        self.fiducial_slider = RangeSlider(self.fiducial_slider_ax, "Fiducial Bounds", valmin = self.img.space.min(), valmax = self.img.space.max())

        #timing buttons
        self.save_time_calibration_ax = self.fig.add_axes([0.72, 0.05, 0.14 ,0.07])
        self.center_time_ax = self.fig.add_axes([0.72, 0.15, 0.14,0.07])
        self.save_time_calibration_button = Button(self.save_time_calibration_ax, "Save Time\nCalibration")
        self.center_time_button = Button(self.center_time_ax, "Center Time")

        #Beam Calibration Buttons
        self.beam_calibration_button_ax = self.fig.add_axes([0.72, 0.64, 0.14, 0.07])
        self.beam_calibration_lineout_button_ax = self.fig.add_axes([0.72, 0.73, 0.14, 0.07])
        self.beam_calibration_button = Button(self.beam_calibration_button_ax, "Apply Beam\nCalibration")
        self.beam_calibration_lineout_button = Button(self.beam_calibration_lineout_button_ax, "Get Ref\nLineout")

    def plot_initial_lineouts(self):
        """
        Plots lineouts and fiducial bounds
        """
        self.fiducial_lower = self.img_ax.axhline([self.fiducial_slider.val[0]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime", label = "Fiducial Bounds")
        self.fiducial_upper = self.img_ax.axhline([self.fiducial_slider.val[1]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime")
        fiducial_lineout = self.img.take_lineout(min = self.fiducial_slider.val[0], max = self.fiducial_slider.val[1])
        self.fiducial_lineout = self.lineout_ax.plot(self.img.time, fiducial_lineout/fiducial_lineout.max(), label = "Fiducial")[0]
        self.showing_lineout = True

    def update_fiducial_bounds(self, val):
        vmin, vmax = val
        self.fiducial_lower.set_ydata([vmin, vmin])
        self.fiducial_upper.set_ydata([vmax, vmax])
        fiducial_lineout = self.img.take_lineout(vmin, vmax)
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.lineout_ax.set_ylim(0, 1)

    def set_ref_folder(self, folder):
        self.beam_ref = folder

    def set_folder(self, folder):
        #sets the save folder for the analysis
        self.folder = folder

    def plot_visar(self):
        self.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.showing_visar = True

    def update_colormap_slider(self, val):
        self.img.update_heatmap_threshold(vmin = val[0], vmax = val[1])
        self.fig.canvas.draw_idle()

    def update_time_shift_slider(self, val):
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        self.fiducial_lineout.set_xdata(self.img.time - val)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.fig.canvas.draw_idle()

    def click_get_ref_lineout(self, val):
        if self.has_ref_lineout == False: #do nothing if we already have the lineout
            if self.beam_ref != "": #If a beam ref has been passed in
                try:
                    ref_data = pd.read_csv(f"{self.beam_ref}/lineouts.csv")
                except:
                    raise Exception("Ref data not found")
            self.lineout_ax.plot(ref_data.time, ref_data.beam, label = "Reference Beam")
            self.lineout_ax.plot(ref_data.time, ref_data.fiducial, label = "Reference Fiducial")
            self.lineout_ax.legend()
            self.has_ref_lineout = True
            self.fig.canvas.draw_idle()

    def click_apply_beam_calibration(self, val):
        if self.beam_calibration_applied == False:
            calibration = ImageCorrection(f"{self.beam_ref}/correction.csv")
            self.img.apply_correction(calibration)
            self.img.show_data(self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
            self.fig.canvas.draw_idle()
        self.beam_calibration_applied = True

    def click_add_shear(self, val):
        """
        Add a shear line to the plot for reference
        """
        self.shear_line = self.img_ax.plot(self.img.time, np.tan(np.radians(self.shear_slider.val))*np.arange(len(self.img.time))*self.img.space_per_pixel + (self.img.space.max() - self.img.space.min())/2, color = "yellow", lw = 2, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
        self.has_shear_line = True
        self.fig.canvas.draw_idle()

    def click_shear(self, val):
        if self.has_shear_line == True:
            self.img.shear_data(angle = -self.shear_slider.val)
            self.img.show_data(self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
            self.sheared_angle += -self.shear_slider.val
            print(self.sheared_angle)
        self.fig.canvas.draw_idle()

    def update_shear_slider(self, val):
        if self.has_shear_line == True:
            self.shear_line[0].set_ydata(np.tan(np.radians(self.shear_slider.val))*np.arange(len(self.img.time))*self.img.space_per_pixel+ (self.img.space.max() - self.img.space.min())/2)
        self.fig.canvas.draw_idle()

    def click_center_time(self, val):
        self.img.set_time_to_zero(self.time_shift_slider.val)
        self.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.img_ax.set_xlim(self.img.time.min(), self.img.time.max())
        self.img_ax.set_ylim(self.img.space.min(), self.img.space.max())
        self.fig.canvas.draw_idle()

    def click_save_time_calibration(self, val):
        df = pd.DataFrame({"time":self.img.time})
        df.to_csv(f"{self.folder}/time.csv")
        info = pd.DataFrame({"beam_ref":[self.beam_ref], "shear": [self.sheared_angle]})
        info.to_csv(f"{self.folder}/info.csv")

    def set_sliders(self):
        self.colormap_slider.on_changed(self.update_colormap_slider)
        self.fiducial_slider.on_changed(self.update_fiducial_bounds)
        self.shear_slider.on_changed(self.update_shear_slider)
        self.time_shift_slider.on_changed(self.update_time_shift_slider)

    def set_buttons(self):
        self.add_shear_button.on_clicked(self.click_add_shear)
        self.shear_button.on_clicked(self.click_shear)
        self.beam_calibration_lineout_button.on_clicked(self.click_get_ref_lineout)
        self.beam_calibration_button.on_clicked(self.click_apply_beam_calibration)
        self.center_time_button.on_clicked(self.click_center_time)
        self.save_time_calibration_button.on_clicked(self.click_save_time_calibration)

    def show_plot(self):
        if self.showing_visar == False:
            self.plot_visar()
        if self.showing_lineout == False:
            self.plot_initial_lineouts()
        self.set_sliders()
        self.set_buttons()
        plt.show()

class ShotAligner:
    """
    Class for aligning a shot ref
    """
    def __init__(self, img:VISARImage):
        self.img = img
        self.name = f"{img.fname.split('/')[-1].lower().replace('.tif', '')}"
        self.showing_visar = False
        self.showing_lineout = False
        self.beam_ref = ""
        self.folder = ""
        self.shot_ref = ""
        self.has_shear_line = False
        self.sheared_angle = 0
        self.has_ref_lineout = 0
        self.beam_calibration_applied = False

    def initialize_plot(self):
        self.fig = plt.figure(figsize = (10, 8))
        gs = GridSpec(3, 1, figure = self.fig)

        #create axes
        self.img_ax = self.fig.add_subplot(gs[:2, 0])
        self.img_ax.set_title(f"Time & Space Calibration: {self.name}")
        self.lineout_ax = self.fig.add_subplot(gs[2,0])
        plt.setp(self.img_ax.get_xticklabels(), visible=False)

        #Label timing sections
        self.fig.text(0.85, 0.55, "Shearing", size = "large", weight = "bold")
        self.fig.text(0.85, 0.35, "Timing", size = "large", weight = "bold")
        self.fig.text(0.85, 0.81, "Beam\nReference", size = "large", weight = "bold")

        #create room for buttons
        self.fig.subplots_adjust(right = 0.8, bottom = 0.2)

        #create shearing button axes
        self.shear_button_ax = self.fig.add_axes([0.82, 0.46, 0.14, 0.07])

        #create shearing buttons
        self.shear_button = Button(self.shear_button_ax, label = "Do Shear\nFrom Ref")

        #bottom sliders
        self.colormap_slider_ax = self.fig.add_axes([0.15, 0.1, 0.45, 0.03])
        self.time_shift_slider_ax = self.fig.add_axes([0.15, 0.06, 0.45, 0.03])
        self.fiducial_slider_ax = self.fig.add_axes([0.15, 0.02, 0.45, 0.03])
        self.colormap_slider = RangeSlider(self.colormap_slider_ax, "Heatmap\nThreshold", self.img.data.min(), self.img.data.max())
        self.time_shift_slider = Slider(self.time_shift_slider_ax, "Time Shift", -self.img.sweep_speed/2, self.img.sweep_speed/2, valinit = 0)
        self.fiducial_slider = RangeSlider(self.fiducial_slider_ax, "Fiducial Bounds", valmin = self.img.space.min(), valmax = self.img.space.max())

        #timing buttons
        self.save_time_calibration_ax = self.fig.add_axes([0.82, 0.15, 0.14 ,0.07])
        self.center_time_ax = self.fig.add_axes([0.82, 0.25, 0.14,0.07])
        self.save_time_calibration_button = Button(self.save_time_calibration_ax, "Save Time\nCalibration")
        self.center_time_button = Button(self.center_time_ax, "Center Time")

        #Beam Calibration Buttons
        self.beam_calibration_button_ax = self.fig.add_axes([0.82, 0.73, 0.14, 0.07])
        self.beam_calibration_lineout_button_ax = self.fig.add_axes([0.82, 0.64, 0.14, 0.07])
        self.beam_calibration_button = Button(self.beam_calibration_button_ax, "Apply Beam\nCalibration")
        self.beam_calibration_lineout_button = Button(self.beam_calibration_lineout_button_ax, "Get Ref\nLineout")

    def plot_initial_lineouts(self):
        """
        Plots lineouts and fiducial bounds
        """
        self.fiducial_lower = self.img_ax.axhline([self.fiducial_slider.val[0]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime", label = "Fiducial Bounds")
        self.fiducial_upper = self.img_ax.axhline([self.fiducial_slider.val[1]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime")
        fiducial_lineout = self.img.take_lineout(min = self.fiducial_slider.val[0], max = self.fiducial_slider.val[1])
        self.fiducial_lineout = self.lineout_ax.plot(self.img.time, fiducial_lineout/fiducial_lineout.max(), label = "Fiducial")[0]
        self.showing_lineout = True

    def update_fiducial_bounds(self, val):
        vmin, vmax = val
        self.fiducial_lower.set_ydata([vmin, vmin])
        self.fiducial_upper.set_ydata([vmax, vmax])
        fiducial_lineout = self.img.take_lineout(vmin, vmax)
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.lineout_ax.set_ylim(0, 1)

    def set_beam_ref_folder(self, folder):
        self.beam_ref = folder

    def set_shot_ref_folder(self, folder):
        self.shot_ref = folder

    def set_folder(self, folder):
        #sets the save folder for the analysis
        self.folder = folder

    def plot_visar(self):
        self.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.showing_visar = True

    def update_colormap_slider(self, val):
        self.img.update_heatmap_threshold(vmin = val[0], vmax = val[1])
        self.fig.canvas.draw_idle()

    def update_time_shift_slider(self, val):
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        self.fiducial_lineout.set_xdata(self.img.time - val)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.fig.canvas.draw_idle()

    def click_get_ref_lineout(self, val):
        if self.has_ref_lineout == False: #do nothing if we already have the lineout
            if self.beam_ref != "": #If a beam ref has been passed in
                try:
                    ref_data = pd.read_csv(f"{self.beam_ref}/lineouts.csv")
                except:
                    raise Exception("Ref data not found")
            self.lineout_ax.plot(ref_data.time, ref_data.beam, label = "Reference Beam")
            self.lineout_ax.plot(ref_data.time, ref_data.fiducial, label = "Reference Fiducial")
            self.lineout_ax.legend()
            self.has_ref_lineout = True
            self.fig.canvas.draw_idle()

    def click_apply_beam_calibration(self, val):
        if self.beam_calibration_applied == False:
            calibration = ImageCorrection(f"{self.beam_ref}/correction.csv")
            self.img.apply_correction(calibration)
            self.img.show_data(self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
            self.fig.canvas.draw_idle()
        self.beam_calibration_applied = True

    def click_shear(self, val):
        info = pd.read_csv(f"{self.shot_ref}/info.csv")
        self.img.shear_data(angle = info["shear"].values[0])
        self.img.show_data(self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.fig.canvas.draw_idle()

    def click_center_time(self, val):
        self.img.set_time_to_zero(self.time_shift_slider.val)
        self.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.img_ax.set_xlim(self.img.time.min(), self.img.time.max())
        self.img_ax.set_ylim(self.img.space.min(), self.img.space.max())
        self.fig.canvas.draw_idle()

    def click_save_time_calibration(self, val):
        df = pd.DataFrame({"time":self.img.time})
        df.to_csv(f"{self.folder}/time.csv")
        info = pd.DataFrame({"beam_ref":[self.beam_ref], 
                             "shot_ref": [self.shot_ref], 
                             "fname": [self.img.fname],
                             "sweep_speed": [self.img.sweep_speed],
                             "slit_size": [self.img.slit_size]})
        info.to_csv(f"{self.folder}/info.csv")

    def set_sliders(self):
        self.colormap_slider.on_changed(self.update_colormap_slider)
        self.fiducial_slider.on_changed(self.update_fiducial_bounds)
        self.time_shift_slider.on_changed(self.update_time_shift_slider)

    def set_buttons(self):
        self.shear_button.on_clicked(self.click_shear)
        self.beam_calibration_lineout_button.on_clicked(self.click_get_ref_lineout)
        self.beam_calibration_button.on_clicked(self.click_apply_beam_calibration)
        self.center_time_button.on_clicked(self.click_center_time)
        self.save_time_calibration_button.on_clicked(self.click_save_time_calibration)

    def show_plot(self):
        if self.showing_visar == False:
            self.plot_visar()
        if self.showing_lineout == False:
            self.plot_initial_lineouts()
        self.set_sliders()
        self.set_buttons()
        plt.show()

class AnalysisPlot:
    """
    Class for performing the actual analysis once everything has been calibrated
    """
    def __init__(self, shot_folder):
        self.shot_folder = shot_folder
        self.open_shot()
        self.showing_visar = False
        self.showing_phase_region = False
        self.has_fft_lineout = False

    def open_shot(self):
        if not os.path.exists(self.shot_folder):
            raise Exception("Folder path not valid")
        
        try:
            self.time = pd.read_csv(f"{self.shot_folder}/time.csv")
            self.info = pd.read_csv(f"{self.shot_folder}/info.csv")
            ref_folder = self.info["shot_ref"].values[0]
            beam_folder = self.info["beam_ref"].values[0]
            fname = self.info["fname"].values[0]
            sweep_speed = self.info["sweep_speed"].values[0]
            slit_size = self.info["slit_size"].values[0]
            ref_info = pd.read_csv(f"{ref_folder}/info.csv")
            shear_angle = ref_info["shear"].values[0]
            correction = ImageCorrection(f"{beam_folder}/correction.csv")
            self.img = VISARImage(fname, sweep_speed = sweep_speed, slit_size = slit_size)
            self.img.apply_correction(correction) #apply beam correction
            self.img.shear_data(shear_angle) #apply shear from shot ref
            self.img.align_time(self.time.time)
        except:
            raise Exception("Shot folder could not be read")
        
    def initialize_plot(self):
        self.name = f"{self.img.fname.split('/')[-1].lower().replace('.tif', '')}"
        gs = GridSpec(3, 2, width_ratios = [1, 1], height_ratios = [5, 2, 2])
        self.fig = plt.figure(figsize = (8, 8))

        self.img_ax = self.fig.add_subplot(gs[0])
        self.phase_ax = self.fig.add_subplot(gs[1])
        self.fourier_lineout_ax = self.fig.add_subplot(gs[4])
        self.velocity_lineout_ax = self.fig.add_subplot(gs[5])
        #self.fig.subplots_adjust(bottom = 0.3)

        #Label sections
        self.fig.text(0.12, 0.46, "Phase Region", size = "medium", weight = "bold")
        self.fig.text(0.12, 0.33, "Fourier Controls", size = "medium", weight = "bold")
        self.fig.text(0.77, 0.47, "Filtering", size = "medium", weight = "bold")

        #phase region sliders
        min_time = (self.img.time.max() - self.img.time.min())*0.2 + self.img.time.min()
        max_time = (self.img.time.max() - self.img.time.min())*0.8 + self.img.time.min()
        min_space = (self.img.space.max() - self.img.space.min())*0.2 + self.img.space.min()
        max_space = (self.img.space.max() - self.img.space.min())*0.8 + self.img.space.min()
        self.x_slider_ax = self.fig.add_axes([0.16, 0.42, 0.2, 0.03])
        self.x_slider = RangeSlider(self.x_slider_ax, "x bounds", self.img.time.min(), self.img.time.max(), valinit = [min_time, max_time])
        self.y_slider_ax = self.fig.add_axes([0.16, 0.39, 0.2, 0.03])
        self.y_slider = RangeSlider(self.y_slider_ax, "y bounds", self.img.space.min(), self.img.space.max(), valinit = [min_space, max_space])
        self.ref_slider_ax = self.fig.add_axes([0.16, 0.36, 0.2, 0.03])
        self.ref_slider = RangeSlider(self.ref_slider_ax, "Ref Bounds", min_time, max_time, valinit = [min_time, min_time + self.img.time_resolution*30])

        #fft slider
        self.fft_slider_ax = self.fig.add_axes([0.16, 0.29, 0.2, 0.03])
        self.fft_slider = RangeSlider(self.fft_slider_ax, "Filter", 0, 1)
        
        #fft buttons
        self.fft_button_ax = self.fig.add_axes([0.1, 0.255, 0.09, 0.03])
        self.fft_button = Button(self.fft_button_ax, label = "Get FFT")
        self.filter_button_ax = self.fig.add_axes([0.23, 0.255, 0.09, 0.03])
        self.filter_button = Button(self.filter_button_ax, label = "Filter")

        #phase buttons
        self.get_phase_button_ax = self.fig.add_axes([0.58, 0.4, 0.1, 0.05])
        self.get_phase_button = Button(self.get_phase_button_ax, "Get Phase")
        self.get_phase_button_ax = self.fig.add_axes([0.58, 0.34, 0.1, 0.05])
        self.get_phase_button = Button(self.get_phase_button_ax, "Save", color = "salmon")

        #velocity slider
        self.velo_slider_ax = self.fig.add_axes([0.62, 0.27, 0.25, 0.03])
        self.velo_slider = RangeSlider(self.velo_slider_ax, "Lineout\n Bounds", 0, 1)

        #title axes
        self.img_ax.set_title("Calibrated Image")
        self.phase_ax.set_title("Phase")
        self.fourier_lineout_ax.set_title("Fourier Transform")
        self.velocity_lineout_ax.set_title("Velocity")

    def show_visar(self):
        self.img.show_data(self.img_ax)
        self.showing_visar = True

    def show_phase_region(self):
        #Get bounds from the phase lineouts
        self.min_x = self.img_ax.plot([self.x_slider.val[0], self.x_slider.val[0]], [self.y_slider.val[0], self.y_slider.val[1]], color = "k")
        self.max_x = self.img_ax.plot([self.x_slider.val[1], self.x_slider.val[1]], [self.y_slider.val[0], self.y_slider.val[1]], color = "k")
        self.min_y = self.img_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.y_slider.val[0], self.y_slider.val[0]], color = "k")
        self.max_y = self.img_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.y_slider.val[1], self.y_slider.val[1]], color = "k")
        self.min_ref = self.img_ax.plot([self.x_slider.val[0], self.x_slider.val[0]], [self.y_slider.val[0], self.y_slider.val[1]], color = "lime")
        self.max_ref = self.img_ax.plot([self.ref_slider.val[1], self.ref_slider.val[1]], [self.y_slider.val[0], self.y_slider.val[1]], color = "lime")
        print(f"=====\n=====\n{self.y_slider.val}")
        self.showing_phase_region = True

    def update_x_slider(self, val):
        #update lines
        self.min_x[0].set_xdata([val[0], val[0]])
        self.max_x[0].set_xdata([val[1], val[1]])
        self.min_y[0].set_xdata([val[0], val[1]])
        self.max_y[0].set_xdata([val[0], val[1]])
        #update bounds on ref slider
        self.ref_slider.valmin = val[0]
        self.ref_slider.valmax = val[1]
        self.ref_slider_ax.set_xlim(val[0], val[1])
        if self.ref_slider.val[0] < val[0]:
            self.ref_slider.set_val([val[0], self.ref_slider.val[1]])
        if self.ref_slider.val[1] > val[1]:
            self.ref_slider.set_val([self.ref_slider.val[0], val[1]])
        self.fig.canvas.draw_idle()

    def update_y_slider(self, val):
        self.min_x[0].set_ydata([val[0], val[1]])
        self.max_x[0].set_ydata([val[0], val[1]])
        self.min_y[0].set_ydata([val[0], val[0]])
        self.max_y[0].set_ydata([val[1], val[1]])
        self.min_ref[0].set_ydata([val[0], val[1]])
        self.max_ref[0].set_ydata([val[0], val[1]])
        self.fig.canvas.draw_idle()

    def update_ref_slider(self, val):
        self.min_ref[0].set_xdata([val[0], val[0]])
        self.max_ref[0].set_xdata([val[1], val[1]])
        self.fig.canvas.draw_idle()

    def click_fft_button(self, val):
        ref_lineout = self.img.take_vert_lineout(self.ref_slider.val[0], self.ref_slider.val[1])
        fft_data = rfft(ref_lineout)
        freq = rfftfreq(len(ref_lineout))
        if self.has_fft_lineout == False:
            self.fft = self.fourier_lineout_ax.plot(freq, fft_data)
            #set fft slider bounds
            self.fft_slider.xmin = min(freq)
            self.fft_slider.xmax = max(freq)
            self.fft_slider.set_val(((max(freq) - min(freq))*0.2 + min(freq), (max(freq) - min(freq))*0.8 + min(freq)))
        else:
            self.fft[0].set_data(freq, fft_data)

    def set_sliders(self):
        self.x_slider.on_changed(self.update_x_slider)
        self.y_slider.on_changed(self.update_y_slider)
        self.ref_slider.on_changed(self.update_ref_slider)

    def set_buttons(self):
        self.fft_button.on_clicked(self.click_fft_button)

    def show_plot(self):
        if self.showing_visar == False:
            self.show_visar()
        if self.showing_phase_region == False:
            self.show_phase_region()
        self.set_sliders()
        self.set_buttons()
        plt.tight_layout()
        plt.show()
