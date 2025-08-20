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
from scipy import ndimage, signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import savgol_filter
from scipy.ndimage import map_coordinates
import shutil
import os

plt.style.use(["fast"]) #fast plotting

#Helper functions
def launch_beamref_plot(fname, folder, sweep_speed, slit_size):
    from VISAR import RefImage
    from InteractivePlots import BeamAligner
    ref_img = RefImage(fname=fname, folder=folder, sweep_speed=sweep_speed, slit_size=slit_size)
    aligner = BeamAligner(ref_img)
    aligner.set_lineout_save_name(os.path.join(folder, "lineouts.csv"))
    aligner.set_correction_save_name(os.path.join(folder, "correction.csv"))
    aligner.show_plot()

def get_beamref_params(beamref_folder, real_data_csv):
    info_path = os.path.join(beamref_folder, "info.csv")
    fname, sweep_speed, slit_size = None, None, None

    if os.path.exists(info_path):
        info = pd.read_csv(info_path)
        if "Filepath" in info.columns:
            fname = info.at[0, "Filepath"]
        elif "filepath" in info.columns:
            fname = info.at[0, "filepath"]
        sweep_speed = info.at[0, "sweep_speed"] if "sweep_speed" in info.columns else None
        slit_size = info.at[0, "slit_size"] if "slit_size" in info.columns else None

    if not fname or pd.isna(fname):
        tif_files = [f for f in os.listdir(beamref_folder) if f.lower().endswith(".tif")]
        if tif_files:
            fname = os.path.join(beamref_folder, tif_files[0])
        else:
            raise Exception("No .tif file found in BeamRef folder.")

    df = pd.read_csv(real_data_csv, dtype=str)
    row = df[df['filepath'] == fname]
    if row.empty:
        base = os.path.basename(fname)
        row = df[df['filename'].astype(str).apply(lambda x: os.path.basename(str(x)) == base)]
    if not row.empty:
        sweep_speed = row.iloc[0].get('sweep_speed', sweep_speed)
        slit_size = row.iloc[0].get('slit_size', slit_size)

    if not sweep_speed or pd.isna(sweep_speed):
        sweep_speed = 20
    if not slit_size or pd.isna(slit_size):
        slit_size = 500

    return fname, float(sweep_speed), float(slit_size)
    
def igor_like_phase(img, xmin, xmax, ymin, ymax, angle_deg, fmin, fmax, use_hann=True, vpf=1.0):
    angle = np.deg2rad(angle_deg)
    ny = ymax - ymin + 1
    nx = xmax - xmin + 1

    # Pre-allocate
    unit_phasor = np.zeros((nx, ny), dtype=np.complex128)

    # Column loop = IGOR's ii
    for ii, x0 in enumerate(range(xmin, xmax+1)):
        # Sample along a tilted line: x = x0 + p*tan(angle), y = ymin + p
        p = np.arange(ny)
        y_line = ymin + p
        x_line = x0 + np.tan(angle) * p

        # Bilinear sample the image along the line
        # map_coordinates expects (rows, cols) = (y, x)
        ctemp = map_coordinates(img, [y_line, x_line], order=1, mode='nearest').astype(np.float64)

        # DC removal (per column)
        ctemp = ctemp - ctemp.mean()

        # Optional window (Parzen/Hann)
        if use_hann:
            win = np.hanning(ny)
            ctemp_win = ctemp * win
        else:
            ctemp_win = ctemp

        # FFT (along y), hard band-pass
        spec = fft(ctemp_win)
        # frequency index in IGOR is integer bins; here treat fmin/fmax as bin indices or normalized freq
        # If fmin/fmax are normalized (0..0.5), map to bins:
        freqs = np.fft.fftfreq(ny)          # cycles/sample, symmetric
        band = (np.abs(freqs) >= fmin) & (np.abs(freqs) <= fmax)
        spec_filtered = np.where(band, spec, 0)

        # IFFT -> complex analytic-like signal (IGOR uses IFFT/C)
        analytic = ifft(spec_filtered)

        # Polar → phase for this column; unwrap along y
        phase_col = np.unwrap(np.angle(analytic))

        # Store unit phasor e^{i phi}
        unit_phasor[ii, :] = np.exp(1j * phase_col)

    # Cross-column phase increment and cumulative sum along x
    dphi = np.angle(unit_phasor[1:, :] / unit_phasor[:-1, :])      # Δϕ_x
    phi = np.zeros((nx, ny))
    phi[1:, :] = np.cumsum(dphi, axis=0)

    # Scale to physical units (match IGOR)
    phi *= (vpf / (2*np.pi))
    return phi.T   # return as [y, x] if you prefer

class ManualRefChop:
    """
    Plot for aligning a beam reference manually
    """
    def __init__(self, ref_img):
        self.ref = ref_img


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
        self.apply_correction_ax = self.fig.add_axes([0.67, 0.06, 0.1,0.06])
        self.take_lineout_ax = self.fig.add_axes([0.8, 0.2, 0.1,0.06])
        self.save_time_calibration_ax = self.fig.add_axes([0.8, 0.13, 0.1,0.06])
        self.center_time_ax = self.fig.add_axes([0.8, 0.06, 0.1,0.06])
        self.manual_chop_ax = self.fig.add_axes([0.67, 0.13, 0.1, 0.06])

        self.get_chop_button = Button(self.get_chop_ax, "Auto Chop")
        self.apply_correction_button = Button(self.apply_correction_ax, "Apply\nCorrection")
        self.take_lineout_button = Button(self.take_lineout_ax, "Take\nLineout")
        self.save_time_calibration_button = Button(self.save_time_calibration_ax, "Save Time\nCalibration")
        self.center_time_button = Button(self.center_time_ax, "Center Time")
        self.manual_chop_button = Button(self.manual_chop_ax, "Manual Chop")
        
        #Saving progress
        self.save_progress_ax = self.fig.add_axes([0.85, 0.01, 0.13, 0.05])
        self.save_progress_button = Button(self.save_progress_ax, "Save Progress")
        self.save_progress_button.on_clicked(lambda event: self.save_progress())

    def plot_visar(self):
        self.ref.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.showing_visar = True

    def plot_lineouts(self):
        fiducial_lineout = self.ref.img.take_lineout(min = self.fiducial_lineout_slider.val[0], max = self.fiducial_lineout_slider.val[1])
        beam_lineout = self.ref.img.take_lineout(min = self.beam_lineout_slider.val[0], max = self.beam_lineout_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        beam_lineout = beam_lineout - np.median(beam_lineout)
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
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        beam_lineout = beam_lineout - np.median(beam_lineout)
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
        chop = [int(0.1*len(self.ref.img.time)), int(0.9*len(self.ref.img.time))]
        ymin = min((beam_lineout[chop[0]:chop[1]]/beam_lineout.max()).min(), (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min())
        ymax = max((beam_lineout/beam_lineout[chop[0]:chop[1]].max()).max(), (fiducial_lineout/fiducial_lineout[chop[0]:chop[1]].max()).max())
        self.lineout_ax.set_ylim(ymin, ymax)
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

    def click_manual_chop(self):
        """
        Let's user perform a manual chop by hand
        This is useful if the program has trouble fitting the peaks
        """

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

    def get_state_dict(self):
        return {
            "beam_lineout_slider_min": self.beam_lineout_slider.val[0],
            "beam_lineout_slider_max": self.beam_lineout_slider.val[1],
            "colormap_slider_min": self.colormap_slider.val[0],
            "colormap_slider_max": self.colormap_slider.val[1],
            "chopper_slider": self.chopper_slider.val,
            "fiducial_lineout_slider_min": self.fiducial_lineout_slider.val[0],
            "fiducial_lineout_slider_max": self.fiducial_lineout_slider.val[1],
            "time_shift_slider": self.time_shift_slider.val,
        }

    def set_state_from_dict(self, state):
        try:
            self.beam_lineout_slider.set_val([float(state["beam_lineout_slider_min"]), float(state["beam_lineout_slider_max"])])
            self.colormap_slider.set_val([float(state["colormap_slider_min"]), float(state["colormap_slider_max"])])
            self.chopper_slider.set_val(float(state["chopper_slider"]))
            self.fiducial_lineout_slider.set_val([float(state["fiducial_lineout_slider_min"]), float(state["fiducial_lineout_slider_max"])])
            self.time_shift_slider.set_val(float(state["time_shift_slider"]))
            # Add more as needed
        except Exception as e:
            print(f"Error loading progress: {e}")
    
    def save_progress(self):
        state = self.get_state_dict()
        df = pd.DataFrame([state])
        progress_path = os.path.join(self.ref.folder, "progress.csv")
        df.to_csv(progress_path, index=False)

    def load_progress(self):
        progress_path = os.path.join(self.ref.folder, "progress.csv")
        if os.path.exists(progress_path):
            df = pd.read_csv(progress_path)
            if not df.empty:
                self.set_state_from_dict(df.iloc[0].to_dict())

    def show_plot(self):
        self.initialize_plot()
        if self.showing_visar == False:
            self.plot_visar()
        if self.showing_lineouts == False:
            self.plot_lineouts()
        if self.has_lineout_bounds == False:
            self.plot_lineout_bounds()
        self.set_sliders()
        self.set_buttons()
        self.load_progress()
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
        
        #Edit Beamref Button
        self.edit_beamref_ax = self.fig.add_axes([0.82, 0.82, 0.14, 0.07])
        self.edit_beamref_button = Button(self.edit_beamref_ax, "Edit BeamRef")
        
        #Save Progress
        self.save_progress_ax = self.fig.add_axes([0.85, 0.01, 0.13, 0.05])
        self.save_progress_button = Button(self.save_progress_ax, "Save Progress")
        self.save_progress_button.on_clicked(lambda event: self.save_progress())

    def plot_initial_lineouts(self):
        """
        Plots lineouts and fiducial bounds
        """
        self.fiducial_lower = self.img_ax.axhline([self.fiducial_slider.val[0]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime", label = "Fiducial Bounds")
        self.fiducial_upper = self.img_ax.axhline([self.fiducial_slider.val[1]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime")
        fiducial_lineout = self.img.take_lineout(min = self.fiducial_slider.val[0], max = self.fiducial_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        self.fiducial_lineout = self.lineout_ax.plot(self.img.time, fiducial_lineout/fiducial_lineout.max(), label = "Fiducial")[0]
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)
        self.showing_lineout = True

    def update_fiducial_bounds(self, val):
        vmin, vmax = val
        self.fiducial_lower.set_ydata([vmin, vmin])
        self.fiducial_upper.set_ydata([vmax, vmax])
        fiducial_lineout = self.img.take_lineout(vmin, vmax)
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)

    def set_beam_ref_folder(self, folder):
        self.beam_ref = folder
        
    def set_shot_ref_folder(self, folder):
        self.shot_ref = folder

    def set_folder(self, folder):
        self.folder = folder

    def plot_visar(self):
        self.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.showing_visar = True

    def update_colormap_slider(self, val):
        self.img.update_heatmap_threshold(vmin = val[0], vmax = val[1])
        self.fig.canvas.draw_idle()

    def update_time_shift_slider(self, val):
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)
        self.fiducial_lineout.set_xdata(self.img.time - val)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.fig.canvas.draw_idle()

    def click_get_ref_lineout(self, val):
        if self.has_ref_lineout == False: #do nothing if we already have the lineout
            if self.beam_ref != "": #If a beam ref has been passed in
                ref_data_path = os.path.join(self.beam_ref, "lineouts.csv")
                ref_data_path = os.path.abspath(ref_data_path)
                if not os.path.exists(ref_data_path):
                    print(f"Reference lineouts file not found: {ref_data_path}")
                    return  # Or show a GUI message
                ref_data = pd.read_csv(ref_data_path)
            self.lineout_ax.plot(ref_data.time, ref_data.beam, label = "Reference Beam")
            self.lineout_ax.plot(ref_data.time, ref_data.fiducial, label = "Reference Fiducial")
            self.lineout_ax.legend()
            self.has_ref_lineout = True
            self.fig.canvas.draw_idle()

    def click_apply_beam_calibration(self, val):
        correction_path = os.path.join(self.beam_ref, "correction.csv")
        print(f"DEBUG: self.beam_ref = {self.beam_ref}")
        if not self.beam_ref or not os.path.exists(self.beam_ref):
            print(f"[ERROR] BeamRef folder is not set or does not exist: {self.beam_ref}")
            return
        correction_path = os.path.abspath(correction_path)
        if not os.path.exists(correction_path):
            print(f"Correction file not found: {correction_path}")
            return
        try:
            calibration = ImageCorrection(correction_path)
        except Exception as e:
            print(f"Failed to load correction: {e}")
            return
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
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.img_ax.set_xlim(self.img.time.min(), self.img.time.max())
        self.img_ax.set_ylim(self.img.space.min(), self.img.space.max())
        self.fig.canvas.draw_idle()

    def click_save_time_calibration(self, val):
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        fiducial_lineout = fiducial_lineout/fiducial_lineout.max()
        df = pd.DataFrame({"time":self.img.time, "fiducial": fiducial_lineout})
        df.to_csv(f"{self.folder}/time.csv")
        info = pd.DataFrame({"beam_ref":[self.beam_ref], "shear": [self.sheared_angle]})
        info.to_csv(f"{self.folder}/info.csv", index=False)

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
        self.edit_beamref_button.on_clicked(self.click_edit_beamref)

    def get_state_dict(self):
        return {
            "colormap_slider_min": self.colormap_slider.val[0],
            "colormap_slider_max": self.colormap_slider.val[1],
            "time_shift_slider": self.time_shift_slider.val,
            "fiducial_slider_min": self.fiducial_slider.val[0],
            "fiducial_slider_max": self.fiducial_slider.val[1],
            "shear_slider": self.shear_slider.val,
            "beam_ref": getattr(self, "beam_ref", ""),
        }

    def set_state_from_dict(self, state):
        try:
            self.colormap_slider.set_val([float(state["colormap_slider_min"]), float(state["colormap_slider_max"])])
            self.time_shift_slider.set_val(float(state["time_shift_slider"]))
            self.fiducial_slider.set_val([float(state["fiducial_slider_min"]), float(state["fiducial_slider_max"])])
            self.shear_slider.set_val(float(state["shear_slider"]))
            if "beam_ref" in state:
                self.beam_ref = state["beam_ref"]
        except Exception as e:
            print(f"Error loading progress: {e}")

    def click_edit_beamref(self, event):
        self.open_beamref_interactive_plot()

    def save_progress(self):
        state = self.get_state_dict()
        df = pd.DataFrame([state])
        progress_path = os.path.join(self.folder, "progress.csv")
        df.to_csv(progress_path, index=False)

    def load_progress(self):
        progress_path = os.path.join(self.folder, "progress.csv")
        if os.path.exists(progress_path):
            df = pd.read_csv(progress_path)
            if not df.empty:
                self.set_state_from_dict(df.iloc[0].to_dict())

    def show_plot(self):
        self.initialize_plot()
        if self.showing_visar == False:
            self.plot_visar()
        if self.showing_lineout == False:
            self.plot_initial_lineouts()
        self.set_sliders()
        self.set_buttons()
        plt.show()
    
    def open_beamref_interactive_plot(self):
        info_path = os.path.join(self.folder, "info.csv")
        if not os.path.exists(info_path):
            print("No info file found.")
            return
    
        info_df = pd.read_csv(info_path)
        if 'beam_ref_path' not in info_df.columns:
            print("No 'beam_ref_path' column found in info.xlsx. Please associate a BeamRef with this analysis.")
            return
        current_beamref = info_df.at[0, 'beam_ref_path']
        if not os.path.exists(current_beamref):
            print(f"BeamRef folder {current_beamref} does not exist.")
            return
    
        real_data_csv = "data/real_info.csv" #adjust later...
        try:
            fname, sweep_speed, slit_size = get_beamref_params(current_beamref, real_data_csv)
        except Exception as e:
            print(f"Error getting BeamRef parameters: {e}")
            return
        print("Launching BeamAligner interactive plot process...")

        launch_beamref_plot(fname, current_beamref, sweep_speed, slit_size)


class ShotAligner:
    """
    Class for aligning a shot ref
    """
    def __init__(self, img:VISARImage, go_to_analysis_callback=None):
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
        self.has_phase = False
        self.beam_calibration_applied = False
        self.go_to_analysis_callback = go_to_analysis_callback

    def initialize_plot(self):
        self.fig = plt.figure(figsize = (10, 8))
        gs = GridSpec(3, 1, figure = self.fig)

        #create axes
        self.img_ax = self.fig.add_subplot(gs[:2, 0])
        self.img_ax.set_title(f"Time & Space Calibration: {self.name}")
        self.lineout_ax = self.fig.add_subplot(gs[2,0])
        plt.setp(self.img_ax.get_xticklabels(), visible=False)

        #Label timing sections
        self.fig.text(0.85, 0.65, "Shearing", size = "large", weight = "bold")
        self.fig.text(0.85, 0.52, "Timing", size = "large", weight = "bold")
        self.fig.text(0.85, 0.85, "Beam\nReference", size = "large", weight = "bold")

        #create room for buttons
        self.fig.subplots_adjust(right = 0.8, bottom = 0.2)

        #create shearing button axes
        self.shear_button_ax = self.fig.add_axes([0.82, 0.57, 0.14, 0.07])

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
        self.save_time_calibration_ax = self.fig.add_axes([0.82, 0.01, 0.14 ,0.07])
        self.center_time_ax = self.fig.add_axes([0.82, 0.44, 0.14,0.07])
        self.save_time_calibration_button = Button(self.save_time_calibration_ax, "Save", color = "salmon")
        self.center_time_button = Button(self.center_time_ax, "Center Time")

        #Beam Calibration Buttons
        self.beam_calibration_button_ax = self.fig.add_axes([0.82, 0.77, 0.14, 0.07])
        self.beam_calibration_lineout_button_ax = self.fig.add_axes([0.82, 0.69, 0.14, 0.07])
        self.beam_calibration_button = Button(self.beam_calibration_button_ax, "Apply Beam\nCalibration")
        self.beam_calibration_lineout_button = Button(self.beam_calibration_lineout_button_ax, "Get Ref\nLineout")

        #breakout time slider & button
        self.breakout_time_button_ax = self.fig.add_axes([0.79, 0.09, 0.2, 0.03])
        self.breakout_time_button = Button(self.breakout_time_button_ax, "Mark Breakout Time")
        self.breakout_time_slider_ax = self.fig.add_axes([0.85, 0.15, 0.03, 0.23])
        self.breakout_time_slider = Slider(self.breakout_time_slider_ax, "$t_{breakout}$", valmin = self.img.time.min(), valmax = self.img.time.max(), valinit = self.img.time.min(), orientation = "vertical")
        self.breakout_uncertainty_ax = self.fig.add_axes([0.91, 0.15, 0.03, 0.23])
        self.breakout_uncertainty_slider = Slider(self.breakout_uncertainty_ax, "$\sigma$", valmin = 0, valmax = 2, valinit = 0, orientation = "vertical")

        # Add "Go to Analysis" Button
        self.go_to_analysis_ax = self.fig.add_axes([0.85, 0.07, 0.13, 0.05])
        self.go_to_analysis_button = Button(self.go_to_analysis_ax, "Go to Analysis", color="lightgreen")
        self.go_to_analysis_button.on_clicked(self.click_go_to_analysis)

    def plot_initial_lineouts(self):
        """
        Plots lineouts and fiducial bounds
        """
        self.breakout = self.img_ax.plot([self.breakout_time_slider.val, self.breakout_time_slider.val], [self.img.space.min(), self.img.space.max()], color = "lime")
        self.breakout_uncertainty = self.img_ax.fill_betweenx(self.img.space, self.breakout_time_slider.val - self.breakout_uncertainty_slider.val, self.breakout_time_slider.val + self.breakout_uncertainty_slider.val, color = "lime", alpha = 0.5)
        self.fiducial_lower = self.img_ax.axhline([self.fiducial_slider.val[0]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime", label = "Fiducial Bounds")
        self.fiducial_upper = self.img_ax.axhline([self.fiducial_slider.val[1]], xmin = self.img.time.min(), xmax = self.img.time.max(), color = "lime")
        fiducial_lineout = self.img.take_lineout(min = self.fiducial_slider.val[0], max = self.fiducial_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        self.fiducial_lineout = self.lineout_ax.plot(self.img.time, fiducial_lineout/fiducial_lineout.max(), label = "Fiducial")[0]
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)
        self.showing_lineout = True

    def update_fiducial_bounds(self, val):
        vmin, vmax = val
        self.fiducial_lower.set_ydata([vmin, vmin])
        self.fiducial_upper.set_ydata([vmax, vmax])
        fiducial_lineout = self.img.take_lineout(vmin, vmax)
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)

    def set_beam_ref_folder(self, folder):
        self.beam_ref = folder

    def set_shot_ref_folder(self, folder):
        self.shot_ref = folder

    def set_folder(self, folder):
        self.folder = folder

    def plot_visar(self):
        self.img.show_data(ax = self.img_ax, minmax = (self.colormap_slider.val[0], self.colormap_slider.val[1]))
        self.showing_visar = True

    def update_colormap_slider(self, val):
        self.img.update_heatmap_threshold(vmin = val[0], vmax = val[1])
        self.fig.canvas.draw_idle()

    def update_time_shift_slider(self, val):
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)
        self.fiducial_lineout.set_xdata(self.img.time - val)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.fig.canvas.draw_idle()

    def update_breakout_time(self):
        """
        Updates the breakout time plots
        """
        self.breakout[0].set_xdata(self.breakout_time_slider.val)
        self.breakout_uncertainty.remove()
        self.breakout_uncertainty = self.img_ax.fill_betweenx(self.img.space, self.breakout_time_slider.val - self.breakout_uncertainty_slider.val, self.breakout_time_slider.val + self.breakout_uncertainty_slider.val, color = "lime", alpha = 0.5)

    def update_breakout_sliders(self, val):
        self.update_breakout_time()

    def click_get_ref_lineout(self, val):
        if self.has_ref_lineout == False: #do nothing if we already have the lineout
            if self.beam_ref != "": #If a beam ref has been passed in
                ref_data_path = os.path.join(self.beam_ref, "lineouts.csv")
                ref_data_path = os.path.abspath(ref_data_path)
                if not os.path.exists(ref_data_path):
                    print(f"Reference lineouts file not found: {ref_data_path}")
                    return  # Or show a GUI message
                ref_data = pd.read_csv(ref_data_path)
            self.lineout_ax.plot(ref_data.time, ref_data.beam, label = "Reference Beam")
            self.lineout_ax.plot(ref_data.time, ref_data.fiducial, label = "Reference Fiducial")
            self.lineout_ax.legend()
            self.has_ref_lineout = True
            self.fig.canvas.draw_idle()

    def click_apply_beam_calibration(self, val):
        correction_path = os.path.join(self.beam_ref, "correction.csv")
        print(f"DEBUG: self.beam_ref = {self.beam_ref}")
        if not self.beam_ref or not os.path.exists(self.beam_ref):
            print(f"[ERROR] BeamRef folder is not set or does not exist: {self.beam_ref}")
            return
        correction_path = os.path.abspath(correction_path)
        if not os.path.exists(correction_path):
            print(f"Correction file not found: {correction_path}")
            return
        try:
            calibration = ImageCorrection(correction_path)
        except Exception as e:
            print(f"Failed to load correction: {e}")
            return
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
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        chop = [int(0.1*len(self.img.time)), int(0.9*len(self.img.time))]
        ymin =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).min()
        ymax =  (fiducial_lineout[chop[0]:chop[1]]/fiducial_lineout.max()).max()
        self.lineout_ax.set_ylim(ymin, ymax)
        self.fiducial_lineout.set_xdata(self.img.time)
        self.fiducial_lineout.set_ydata(fiducial_lineout/fiducial_lineout.max())
        self.breakout_time_slider.valmin = self.img.time.min() - self.time_shift_slider.val
        self.breakout_time_slider.valmax = self.breakout_time_slider.valmin + (self.img.time.max() - self.img.time.min())*0.9
        self.breakout_time_slider.set_val(self.breakout_time_slider.valmin)
        self.breakout_time_slider_ax.set_ylim(self.breakout_time_slider.valmin, self.breakout_time_slider.valmax)
        self.breakout_time_slider.val = self.breakout_time_slider.val - self.time_shift_slider.val
        self.update_breakout_time()
        self.img_ax.set_xlim(self.img.time.min(), self.img.time.max())
        self.img_ax.set_ylim(self.img.space.min(), self.img.space.max())
        self.fig.canvas.draw_idle()

    def click_save_time_calibration(self, val):
        fiducial_lineout = self.img.take_lineout(self.fiducial_slider.val[0], self.fiducial_slider.val[1])
        fiducial_lineout = fiducial_lineout - np.median(fiducial_lineout)
        fiducial_lineout = fiducial_lineout/fiducial_lineout.max()
        df = pd.DataFrame({"time":self.img.time, "fiducial": fiducial_lineout})
        df.to_csv(f"{self.folder}/time.csv")
        info = pd.DataFrame({"beam_ref":[self.beam_ref], 
                             "shot_ref": [self.shot_ref], 
                             "fname": [self.img.fname],
                             "sweep_speed": [self.img.sweep_speed],
                             "slit_size": [self.img.slit_size]})
        info.to_csv(f"{self.folder}/info.csv", index=False)

    def click_go_to_analysis(self, event):
        folder = self.folder
        callback = self.go_to_analysis_callback
        assert self.fig is not None, "Figure not initialized!"
        plt.close(self.fig)
        if callback and folder:
            import threading
            threading.Timer(0.1, lambda: callback(folder)).start()

    def set_sliders(self):
        self.colormap_slider.on_changed(self.update_colormap_slider)
        self.fiducial_slider.on_changed(self.update_fiducial_bounds)
        self.time_shift_slider.on_changed(self.update_time_shift_slider)
        self.breakout_time_slider.on_changed(self.update_breakout_sliders)
        self.breakout_uncertainty_slider.on_changed(self.update_breakout_sliders)

    def set_buttons(self):
        self.shear_button.on_clicked(self.click_shear)
        self.beam_calibration_lineout_button.on_clicked(self.click_get_ref_lineout)
        self.beam_calibration_button.on_clicked(self.click_apply_beam_calibration)
        self.center_time_button.on_clicked(self.click_center_time)
        self.save_time_calibration_button.on_clicked(self.click_save_time_calibration)
        #self.edit_beamref_button.on_clicked(self.click_edit_beamref)
        # self.go_to_analysis_button.on_clicked(self.click_go_to_analysis)

    def get_state_dict(self):
        return {
            "colormap_slider_min": self.colormap_slider.val[0],
            "colormap_slider_max": self.colormap_slider.val[1],
            "time_shift_slider": self.time_shift_slider.val,
            "fiducial_slider_min": self.fiducial_slider.val[0],
            "fiducial_slider_max": self.fiducial_slider.val[1],
            "beam_ref": getattr(self, "beam_ref", ""),
        }

    def set_state_from_dict(self, state):
        try:
            self.colormap_slider.set_val([float(state["colormap_slider_min"]), float(state["colormap_slider_max"])])
            self.time_shift_slider.set_val(float(state["time_shift_slider"]))
            self.fiducial_slider.set_val([float(state["fiducial_slider_min"]), float(state["fiducial_slider_max"])])
            if "beam_ref" in state:
                self.beam_ref = state["beam_ref"]
        except Exception as e:
            print(f"Error loading progress: {e}")

    def click_edit_beamref(self, event):
        self.open_beamref_interactive_plot()

    def save_progress(self):
        state = self.get_state_dict()
        df = pd.DataFrame([state])
        progress_path = os.path.join(self.folder, "progress.csv")
        df.to_csv(progress_path, index=False)

    def load_progress(self):
        progress_path = os.path.join(self.folder, "progress.csv")
        if os.path.exists(progress_path):
            df = pd.read_csv(progress_path)
            if not df.empty:
                self.set_state_from_dict(df.iloc[0].to_dict())
                
    def show_plot(self):
        self.initialize_plot()
        if self.showing_visar == False:
            self.plot_visar()
        if self.showing_lineout == False:
            self.plot_initial_lineouts()
        self.set_sliders()
        self.set_buttons()
        self.load_progress()
        plt.show()

    def open_beamref_interactive_plot(self):
        import os
        import pandas as pd
    
        info_path = os.path.join(self.folder, "info.csv")
        if not os.path.exists(info_path):
            print("No info file found.")
            return
    
        info_df = pd.read_csv(info_path)
        if 'beam_ref_path' not in info_df.columns:
            print("No 'beam_ref_path' column found in info.xlsx. Please associate a BeamRef with this analysis.")
            return
        current_beamref = info_df.at[0, 'beam_ref_path']
        if not os.path.exists(current_beamref):
            print(f"BeamRef folder {current_beamref} does not exist.")
            return
    
        real_data_csv = "data/real_info.csv" #adjust later...
        try:
            fname, sweep_speed, slit_size = get_beamref_params(current_beamref, real_data_csv)
        except Exception as e:
            print(f"Error getting BeamRef parameters: {e}")
            return
        print("Launching BeamAligner interactive plot process...")

        launch_beamref_plot(fname, current_beamref, sweep_speed, slit_size)
        
    def click_go_to_analysis(self, event):
        folder = self.folder
        callback = self.go_to_analysis_callback
        assert self.fig is not None, "Figure not initialized!"
        plt.close(self.fig)
        if callback and folder:
            import threading
            threading.Timer(0.1, lambda: callback(folder)).start()

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
        self.transformed = False
        self.has_phase = False

    def open_shot(self):
        if not os.path.exists(self.shot_folder):
            raise Exception("Folder path not valid")
        
        try:
            self.time = pd.read_csv(f"{self.shot_folder}/time.csv")
            self.info = pd.read_csv(f"{self.shot_folder}/info.csv")
        except:
            raise Exception(f"Shot folder {self.shot_folder} could not be read")
        ref_folder = self.info["shot_ref"].values[0]
        if not ref_folder or str(ref_folder).lower() in ['nan', '', 'none']:
            raise Exception("ShotRef folder not set for this analysis. Please ensure the reference is selected and available.")
        beam_folder = self.info["beam_ref"].values[0]
        fname = self.info["fname"].values[0]
        sweep_speed = self.info["sweep_speed"].values[0]
        slit_size = self.info["slit_size"].values[0]
        ref_info = pd.read_csv(f"{ref_folder}/info.csv")
        shear_angle = ref_info["shear"].values[0]
        correction = ImageCorrection(f"{beam_folder}/correction.csv")
        self.img = VISARImage(fname, sweep_speed = sweep_speed, slit_size = slit_size)
        print("\n\n\n====\n\n\n")
        self.img.apply_correction(correction) #apply beam correction
        self.img.shear_data(shear_angle) #apply shear from shot ref
        self.img.align_time(self.time.time)
        
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
        self.get_phase_button_ax = self.fig.add_axes([0.55, 0.43, 0.1, 0.05])
        self.get_phase_button = Button(self.get_phase_button_ax, "Get Phase")
        self.zero_phase_button_ax = self.fig.add_axes([0.55, 0.37, 0.1, 0.05])
        self.zero_phase_button = Button(self.zero_phase_button_ax, "Zero Phase")
        self.save_phase_button_ax = self.fig.add_axes([0.55, 0.31, 0.1, 0.05])
        self.save_phase_button = Button(self.save_phase_button_ax, "Save", color = "salmon")

        #velocity slider
        self.velo_slider_ax = self.fig.add_axes([0.62, 0.27, 0.25, 0.03])
        self.velo_slider = RangeSlider(self.velo_slider_ax, "Lineout\n Bounds", 0, 1)

        #Filtering info
        self.gaussian_filter_ax = self.fig.add_axes([0.75, 0.42, 0.08, 0.03])
        self.gaussian_filter_button = Button(self.gaussian_filter_ax, "Gauss")  
        self.median_filter_ax = self.fig.add_axes([0.75, 0.38, 0.08, 0.03])
        self.median_filter_button = Button(self.median_filter_ax, "Median")  
        self.median_x_entry_ax = self.fig.add_axes([0.86, 0.38, 0.04, 0.03])
        self.median_y_entry_ax = self.fig.add_axes([0.93, 0.38, 0.04, 0.03])
        self.median_x_entry = TextBox(self.median_x_entry_ax, "x ", initial = "1")
        self.median_y_entry = TextBox(self.median_y_entry_ax, "y ", initial = "1")

        #vpf entry box
        self.vpf_entry_ax = self.fig.add_axes([0.62, 0.23, 0.1, 0.03])
        self.vpf_box = TextBox(self.vpf_entry_ax, "", "1", color = "salmon")
        self.vpf = float(self.vpf_box.text)
        self.vpf_button_ax = self.fig.add_axes([0.57, 0.23, 0.04, 0.03])
        self.vpf_button = Button(self.vpf_button_ax, "VPF", color = "salmon")

        #title axes
        self.img_ax.set_title("Calibrated Image")
        self.phase_ax.set_title("Phase")
        self.fourier_lineout_ax.set_title("Fourier Transform")
        self.velocity_lineout_ax.set_title("Velocity")

    def fft_plot_zoom_update(self, ax_instance):
        valmin, valmax = ax_instance.get_xlim()
        #update slider bounds
        self.fft_slider.valmin = valmin
        self.fft_slider.valmax = valmax
        #update slider ax bounds
        self.fft_slider_ax.set_xlim(valmin, valmax)
        slider_val = list(self.fft_slider.val)
        new_slider_val = slider_val
        if valmin < self.fft_slider.val[0]:
            new_slider_val[0] = valmin
        if valmax > self.fft_slider.val[1]:
            new_slider_val[1] = valmax
        self.fft_slider.set_val(new_slider_val)

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

    def update_fourier_slider(self, val):
        if self.has_fft_lineout == True:
            self.fourier_min[0].set_xdata([self.fft_slider.val[0], self.fft_slider.val[0]])
            self.fourier_min[0].set_ydata([self.ref_fft.min(), self.ref_fft.max()])
            self.fourier_max[0].set_xdata([self.fft_slider.val[1], self.fft_slider.val[1]])
            self.fourier_max[0].set_ydata([self.ref_fft.min(), self.ref_fft.max()])

    def update_velo_slider(self, val):
        if self.has_phase == True:
            self.min_phase_lineout[0].set_ydata([self.velo_slider.val[0], self.velo_slider.val[0]])
            self.max_phase_lineout[0].set_ydata([self.velo_slider.val[1], self.velo_slider.val[1]])
            min_loc = int((self.velo_slider.val[0] - self.y_slider.val[0])/self.img.space_per_pixel)
            max_loc = int((self.velo_slider.val[1] - self.y_slider.val[0])/self.img.space_per_pixel)
            self.velocity = self.phase[min_loc:max_loc, :].mean(axis = 0)
            self.velocity_lineout_ax.set_ylim(self.velocity.min(), self.velocity.max())
            self.velo[0].set_ydata(self.velocity)
            self.velocity_lineout_ax.set_ylim(self.velocity.min(), self.velocity.max())

    def click_fft_button(self, val):
        #get the fft from the reference region
        ref_lineout = self.img.take_vert_lineout(self.ref_slider.val[0], self.ref_slider.val[1], self.y_slider.val[0], self.y_slider.val[1])
        self.ref_lineout = ref_lineout
        print(ref_lineout.shape)
        fft_data = np.abs(fft(ref_lineout))
        self.initial_phase = np.angle(fft_data)
        self.ref_fft = fft_data
        freq = fftfreq(len(ref_lineout))
        self.freq = freq
        self.fft_slider.xmin = min(freq)
        self.fft_slider.xmax = max(freq)
        self.fft_slider.set_val(((max(freq) - min(freq))*0.2 + min(freq), (max(freq) - min(freq))*0.8 + min(freq)))
        if self.has_fft_lineout == False:
            self.fft = self.fourier_lineout_ax.plot(freq[:len(ref_lineout)//2], fft_data[0:len(ref_lineout)//2])
            self.fourier_min = self.fourier_lineout_ax.plot([self.fft_slider.val[0], self.fft_slider.val[0]], [fft_data.min(), fft_data.max()], color = "red")
            self.fourier_max = self.fourier_lineout_ax.plot([self.fft_slider.val[1], self.fft_slider.val[1]], [fft_data.min(), fft_data.max()], color = "red")
        else:
            self.fft[0].set_data(freq[:len(ref_lineout)//2], fft_data[0:len(ref_lineout)//2])
            self.fourier_min[0].set_xdata([self.fft_slider.val[0], self.fft_slider.val[0]])
            self.fourier_min[0].set_ydata([fft_data.min(), fft_data.max()])
            self.fourier_max[0].set_xdata([self.fft_slider.val[0], self.fft_slider.val[0]])
            self.fourier_max[0].set_ydata([fft_data.min(), fft_data.max()])
        self.fig.canvas.draw_idle()
        self.has_fft_lineout = True

    def click_filter_button(self, val):
        """
        Get the fourier transform for the VISAR data
        """
        #fourier_filter = np.multiply((self.freq < self.fft_slider.val[1]), (self.freq > self.fft_slider.val[0]))
        min_space = int((self.y_slider.val[0] - self.img.space.min())/self.img.space_per_pixel)
        max_space = int((self.y_slider.val[1] - self.img.space.min())/self.img.space_per_pixel)
        min_time = int((self.x_slider.val[0] - self.img.time.min())/self.img.time_resolution)
        max_time = int((self.x_slider.val[1] - self.img.time.min())/self.img.time_resolution)
        data_chunk = self.img.data[min_space:max_space, min_time:max_time]
        fft_ = fft(data_chunk, axis = 0)
        phase = np.angle(fft_)
        freq = np.abs(np.vstack([fftfreq(data_chunk.shape[1]) for i in range(data_chunk.shape[0])]))
        #print(freq.shape, fft_.shape)
        filter = np.logical_and(freq > self.fft_slider.val[0], freq < self.fft_slider.val[1])
        filtered_fft = fft_*filter
        freq = fftfreq(data_chunk.shape[0])
        filter_mask = np.logical_and(freq > self.fft_slider.val[0], freq < self.fft_slider.val[1])
        filtered_fft = fft_ * filter_mask[:,np.newaxis]
        #self.phase = np.angle(filtered_fft)

        self.phase = self.img.get_phase_igor(
            x_bounds=(self.x_slider.val[0],self.x_slider.val[1]),
            y_bounds=(self.y_slider.val[0], self.y_slider.val[1]),
            fband=(self.fft_slider.val[0], self.fft_slider.val[1]),
            angle_deg=0.0,          # or your shear/tilt if not already corrected upstream
            use_hann=True,
            vpf=1)

        #==
        #Before fix
        #self.phase = np.unwrap(np.angle(filtered_fft), axis=0)
        #self.phase = self.phase/(2*np.pi)
        #==

        #print(data_chunk.shape, self.phase.shape)
        #self.phase = self.phase - self.initial_phase[:, np.newaxis]
        self.original_phase = self.phase
        fourier_filtered_all = ifft(filtered_fft, n = data_chunk.shape[0], axis = 0).real
        fourier_filtered = fourier_filtered_all.real
        #get phase from the filtered chunk
        fft2 = fft(fourier_filtered, axis = 0)
        print(self.phase.shape)
        #self.phase = np.unwrap(np.angle(fft2), axis=0)/(2*np.pi)
        self.original_phase = self.phase
        print(self.phase.shape, fourier_filtered.shape)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax.set_title("Fourier Filtered")
        ax2.set_title("Reference Lineout")
        X, Y = np.meshgrid(np.linspace(self.x_slider.val[0], self.x_slider.val[1], fourier_filtered.shape[1] + 1), np.linspace(self.y_slider.val[0], self.y_slider.val[1], fourier_filtered.shape[0] + 1))
        #print(np.shape(X))
        #print(np.shape(Y))
        #print(np.shape(fourier_filtered))
        #print(fourier_filtered.min(), fourier_filtered.max())
        heat_min = 0.01 if fourier_filtered.min() <= 0 else fourier_filtered.min()
        ax.pcolormesh(X, Y, fourier_filtered, cmap='magma')
        ax2.plot(np.linspace(self.y_slider.val[0], self.y_slider.val[1], len(self.ref_lineout)), self.ref_lineout)
        self.transformed = True
        plt.show()

    def initialize_velo_plot(self):
        """
        When we get the phase, we should also get the velocity lineout plot
        """
        phase_time = np.linspace(self.x_slider.val[0], self.x_slider.val[1], self.phase.shape[1])
        min_loc, max_loc = int(self.phase.shape[0]* 0.2), int(self.phase.shape[0]*0.8)
        min_loc_space = min_loc*self.img.space_per_pixel + self.y_slider.val[0]
        max_loc_space = max_loc*self.img.space_per_pixel + self.y_slider.val[0]
        self.velo_slider.valmin = self.y_slider.val[0]
        self.velo_slider.valmax = self.y_slider.val[1]
        self.velo_slider.set_val([min_loc_space, max_loc_space])
        self.velo_slider_ax.set_xlim([min_loc_space, max_loc_space])
        self.min_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[0], self.velo_slider.val[0]], c = "red")
        self.max_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[1], self.velo_slider.val[1]], c = "red")
        self.velocity = self.phase[min_loc:max_loc, :].mean(axis = 0)
        self.velocity = savgol_filter(self.velocity, window_length = int(len(self.img.time)/100), polyorder = 2)
        self.velo = self.velocity_lineout_ax.plot(phase_time, self.velocity)
        self.fig.canvas.draw_idle()

    def click_get_phase(self, val):
        """
        Gets phase from transformed data
        """
        self.phase_ax.clear()
        self.phase_ax.set_title("Phase")
        self.velocity_lineout_ax.clear()
        self.velocity_lineout_ax.set_title("Velocity")
        if self.transformed == True:
            X, Y = np.meshgrid(np.linspace(self.x_slider.val[0], self.x_slider.val[1], self.phase.shape[1] + 1), np.linspace(self.y_slider.val[0], self.y_slider.val[1], self.phase.shape[0] + 1))
            self.phase_ax.pcolormesh(X, Y, self.original_phase, cmap = "viridis")
            self.initialize_velo_plot()
            self.has_phase = True
            self.vpf_applied = False
        self.fig.canvas.draw_idle()

    def click_gaussian_filter(self, val):
        if self.transformed == True:
            self.phase = ndimage.gaussian_filter(self.phase, sigma = 2)
            self.phase_ax.clear()
            self.phase_ax.set_title("Phase")
            X, Y = np.meshgrid(np.linspace(self.x_slider.val[0], self.x_slider.val[1], self.phase.shape[1] + 1), np.linspace(self.y_slider.val[0], self.y_slider.val[1], self.phase.shape[0] + 1))
            vmin = int((self.velo_slider.val[0] - self.y_slider.val[0])/self.img.space_per_pixel)
            vmax = int((self.velo_slider.val[1] - self.y_slider.val[0])/self.img.space_per_pixel)
            self.velocity = self.phase[vmin:vmax, :].mean(axis = 0)
            self.phase_ax.pcolormesh(X, Y, self.phase, cmap = "viridis")
            self.velo[0].set_ydata(self.velocity)
            self.min_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[0], self.velo_slider.val[0]], color = "red")
            self.max_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[1], self.velo_slider.val[1]], color = "red")
            print("Gaussian filtered")
            self.vpf_applied = False
            self.fig.canvas.draw_idle()

    def click_median_filter(self, val):
        if self.transformed == True:
            self.phase = ndimage.median_filter(self.phase, size = (int(self.median_x_entry.text), int(self.median_y_entry.text)))
            self.phase_ax.clear()
            self.phase_ax.set_title("Phase")
            X, Y = np.meshgrid(np.linspace(self.x_slider.val[0], self.x_slider.val[1], self.phase.shape[1] + 1), np.linspace(self.y_slider.val[0], self.y_slider.val[1], self.phase.shape[0] + 1))
            vmin = int((self.velo_slider.val[0] - self.y_slider.val[0])/self.img.space_per_pixel)
            vmax = int((self.velo_slider.val[1] - self.y_slider.val[0])/self.img.space_per_pixel)
            self.velocity = self.phase[vmin:vmax, :].mean(axis = 0)
            self.phase_ax.pcolormesh(X, Y, self.phase, cmap = "viridis")
            self.velo[0].set_ydata(self.velocity)
            self.min_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[0], self.velo_slider.val[0]], color = "red")
            self.max_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[1], self.velo_slider.val[1]], color = "red")
            print("Median filtered")
            self.vpf_applied = False
            self.fig.canvas.draw_idle()

    def click_zero_phase(self, val):
        if self.transformed == True:
            minval = int((self.ref_slider.val[0] - self.x_slider.val[0])/self.img.time_resolution)
            maxval = int((self.ref_slider.val[1] - self.x_slider.val[0])/self.img.time_resolution)
            self.zero_phase = self.velocity[minval:maxval].mean()
            self.phase -= self.zero_phase
            self.phase_ax.clear()
            self.phase_ax.set_title("Phase")
            X, Y = np.meshgrid(np.linspace(self.x_slider.val[0], self.x_slider.val[1], self.phase.shape[1] + 1), np.linspace(self.y_slider.val[0], self.y_slider.val[1], self.phase.shape[0] + 1))
            vmin = int((self.velo_slider.val[0] - self.y_slider.val[0])/self.img.space_per_pixel)
            vmax = int((self.velo_slider.val[1] - self.y_slider.val[0])/self.img.space_per_pixel)
            self.velocity = self.phase[vmin:vmax, :].mean(axis = 0)
            self.phase_ax.pcolormesh(X, Y, self.phase, cmap = "viridis")
            self.velo[0].set_ydata(self.velocity)
            self.min_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[0], self.velo_slider.val[0]], color = "red")
            self.max_phase_lineout = self.phase_ax.plot([self.x_slider.val[0], self.x_slider.val[1]], [self.velo_slider.val[1], self.velo_slider.val[1]], color = "red")
            print("Phase zeroed")
            self.velocity_lineout_ax.set_ylim(self.velocity.min(), self.velocity.max())
            self.fig.canvas.draw_idle()

    def click_vpf(self, val):
        """
        Apply vpf correction upon click
        """
        if self.vpf_applied == False:
            self.velocity = self.velocity * float(self.vpf_box.text)
            #self.velocity = (self.phase.mean(axis=0) / (2 * np.pi)) * self.vpf
        self.velo[0].set_ydata(self.velocity)
        self.velocity_lineout_ax.set_ylim(self.velocity.min(), self.velocity.max())
        self.fig.canvas.draw_idle()
        self.vpf_applied = True

    def set_sliders(self):
        self.x_slider.on_changed(self.update_x_slider)
        self.y_slider.on_changed(self.update_y_slider)
        self.ref_slider.on_changed(self.update_ref_slider)
        self.fft_slider.on_changed(self.update_fourier_slider)
        self.velo_slider.on_changed(self.update_velo_slider)

    def set_buttons(self):
        self.fft_button.on_clicked(self.click_fft_button)
        self.filter_button.on_clicked(self.click_filter_button)
        self.get_phase_button.on_clicked(self.click_get_phase)
        self.gaussian_filter_button.on_clicked(self.click_gaussian_filter)
        self.median_filter_button.on_clicked(self.click_median_filter)
        self.zero_phase_button.on_clicked(self.click_zero_phase)
        self.vpf_button.on_clicked(self.click_vpf)

    def show_plot(self):
        self.initialize_plot()
        if self.showing_visar == False:
            self.show_visar()
        if self.showing_phase_region == False:
            self.show_phase_region()
        self.set_sliders()
        self.set_buttons()
        self.fourier_lineout_ax.callbacks.connect("xlim_changed", self.fft_plot_zoom_update)
        plt.tight_layout()
        plt.show()
