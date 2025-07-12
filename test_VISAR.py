"""
Testing VISAR.py
"""
from VISAR import *
from InteractivePlots import *
from SyntheticData import *
from matplotlib import pyplot as plt
import pandas as pd
import os

#Parameters and saving data
ref_file = "data/VISAR1/0409_1635_20ns_1ns_westbeam_Visar1.tif"
shot_name = "0409_1635_20ns_1ns_westbeam_Visar1"
ref_name = "0409_1635_20ns_1ns_westbeam_Visar1"
sweep_speed = 20
slit_size = 500
info_csv_path = "./python_analysis/info.csv"

def create_info_csv(shot_name, ref_name, sweep_speed, slit_size, info_csv_path=info_csv_path):
    new_rows = [
        {"shot": shot_name, "sweep_time": sweep_speed, "slit_size": slit_size},
        {"shot": ref_name, "sweep_time": sweep_speed, "slit_size": slit_size}
    ]
    os.makedirs(os.path.dirname(info_csv_path), exist_ok=True)
    if os.path.exists(info_csv_path):
        df = pd.read_csv(info_csv_path)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df = pd.DataFrame(new_rows)
    df.to_csv(info_csv_path, index=False)
    print(f"Saved shot/ref info to {info_csv_path}")

#Testing...
get_data = 0
show_base_tif = 0
test_lineout = 0
test_ref = 0
test_ref_split = 0
test_image_correction = 0 #plots a corrected image
demo_img_correction = 0 #demonstrates improvement from the correction
test_interactive_ref_plot = 0 #tests the interactive reference timing plot
test_initialize_image_w_data = 0 #tests image initialization with data
test_synthetic_beam_lineout = 0
test_shot_aligner_plot = 1 #tests the interactive plot for shot alignment

#Tests
if any([test_ref, test_ref_split, test_image_correction, demo_img_correction, test_interactive_ref_plot, test_initialize_image_w_data]):
    ref = RefImage(ref_file, sweep_speed, slit_size)
    create_info_csv(shot_name, ref_name, sweep_speed, slit_size)

#Tests
if any([test_ref, test_ref_split, test_image_correction, demo_img_correction, test_interactive_ref_plot, test_initialize_image_w_data]):
    ref = RefImage(ref_file, sweep_speed, slit_size)
    create_info_csv(shot_name, ref_name, sweep_speed, slit_size)

if test_ref == True:
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(ref_file)
    ref.img.show_data(ax = ax1, minmax = (100, 2000))
    plt.show()

if test_ref_split == True:
    ref.chop_beam(ybounds = (0, 450), num_slices = 20)
    ref.plot_chop(minmax = (100, 2000))

if test_image_correction == True:
    """
    Pass a correction to an image to demonstrate it works
    """
    #read in a ref file and chop beam to get the correction
    ref.chop_beam(ybounds = (0, 450), num_slices = 50)
    ref.save_chop_as_correction() #get the correction but don't save it to a file
    
    #set up the plot
    fig = plt.figure(figsize = (8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Raw Image")
    ax2 = fig.add_subplot(1, 2, 2, sharex = ax1)
    ax2.set_title("Corrected Image")

    #plot the raw image on the plot
    ref.img.show_data(ax = ax1, minmax = (100, 2000))

    #use the correction from the chopped beam to adjust the plot
    ref.img.apply_correction(ref.correction)
    ref.img.show_data(ax = ax2, minmax = (100, 2000))
    ax1.set_xlim(2.1, 5)
    plt.show()

if demo_img_correction == True:
    #read in a ref file and chop beam to get the correction
    ref.chop_beam(ybounds = (0, 450), num_slices = 50)
    ref.save_chop_as_correction() #get the correction but don't save it to a file
    ref.plot_chop(minmax = (100, 2000)) #chop w/out correction
    ref.save_chop_as_correction()
    ref.img.apply_correction(ref.correction)

    #Show the improvement from the correction
    ref.chop_beam(ybounds = (0, 450), num_slices = 50)
    plt.clf()
    ref.plot_chop(minmax = (100, 2000))

if test_interactive_ref_plot == True:
    aligner = BeamAligner(ref)
    aligner.initialize_plot()
    aligner.set_lineout_save_name("ref_lineout.csv")
    aligner.set_correction_save_name("ref_correction.csv")
    aligner.show_plot()

if test_initialize_image_w_data == True:
    data = ref.img.data
    #initialize a new file with the reference data
    new_img = VISARImage(fname = None, data = data, sweep_speed = 20, slit_size = 500)
    #show the new image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    new_img.show_data(ax)
    plt.show()

if test_synthetic_beam_lineout == True:
    simulated = SyntheticBeamCalibration(sweep_speed = 20, slit_size = 500, time_points = 1000, space_points = 500)
    simulated.generate_background(500)
    simulated.generate_beam(3.5, 1, 2500, max_loc = 430, shift = 2/500)
    simulated.generate_fiducial(timing_offset = 3.2, space_loc = 465, amp = 2000, width = 4, height = 10)
    synthetic_img = VISARImage(fname = None, data = simulated.data, sweep_speed = simulated.sweep_speed, slit_size = simulated.slit_size)
    fig = plt.subplots()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Simulated Beam Reference")
    synthetic_img.show_data(ax, minmax = (simulated.data.min(), simulated.data.max()))
    #If the tif doesn't currently exist, save it
    if os.path.exists("SyntheticData/20nsBeamReference.tif") == False:
        synthetic_img.save_tif("SyntheticData/20nsBeamReference.tif")
    plt.show()


if test_shot_aligner_plot == True:
    shot_file = "../JLF_2025/VISAR1/0408_1452_Shot54_Visar1_ref.tif"
    img = VISARImage(shot_file)
    aligner = ShotAligner(img)
    aligner.initialize_plot()
    aligner.show_plot()