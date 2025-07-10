"""
Testing VISAR.py
"""
from VISAR import *
from InteractivePlots import*
from matplotlib import pyplot as plt

get_data = 0
show_base_tif = 0
test_lineout = 0
test_ref = 0
test_ref_split = 0
test_image_correction = 0 #plots a corrected image
demo_img_correction = 0 #demonstrates improvement from the correction
test_interactive_ref_plot = 1 #tests the interactive reference timing plot

if test_ref == True:
    # 20 ns ref file
    ref_file = "../../VISAR1/0409_1635_20ns_1ns_westbeam_Visar1.tif"
    ref = RefImage(fname = ref_file, sweep_speed = 20, slit_size = 500)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(ref_file)
    ref.img.show_data(ax = ax1, minmax = (100, 2000))
    plt.show()

if test_ref_split == True:
    ref_file = "../../VISAR1/0409_1635_20ns_1ns_westbeam_Visar1.tif"
    ref = RefImage(fname = ref_file, sweep_speed = 20, slit_size = 500)
    ref.chop_beam(ybounds = (0, 450), num_slices = 20)
    ref.plot_chop(minmax = (100, 2000))

if test_image_correction == True:
    """
    Pass a correction to an image to demonstrate it works
    """
    #read in a ref file and chop beam to get the correction
    ref_file = "../../VISAR1/0409_1635_20ns_1ns_westbeam_Visar1.tif"
    ref = RefImage(fname = ref_file, sweep_speed = 20, slit_size = 500)
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
    ref_file = "../../VISAR1/0409_1635_20ns_1ns_westbeam_Visar1.tif"
    ref = RefImage(fname = ref_file, sweep_speed = 20, slit_size = 500)
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
    ref_file = "../../VISAR1/0409_1635_20ns_1ns_westbeam_Visar1.tif"
    ref = RefImage(fname = ref_file, sweep_speed = 20, slit_size = 500)
    aligner = BeamAligner(ref)
    aligner.initialize_plot()
    aligner.show_plot()
