"""
Script for generating synthetic data
"""
from VISAR import *
from SyntheticData import *

show = 0

if not os.path.exists("SyntheticData"):
    os.mkdir("SyntheticData")

#Generate a Beam Reference
synthetic_beam = SyntheticBeamCalibration(sweep_speed = 20, slit_size = 500, time_points = 1000, space_points = 500)
synthetic_beam.generate_background(500)
synthetic_beam.generate_beam(3.5, 1, 2500, max_loc = 430, shift = 2/500)
synthetic_beam.generate_fiducial(time_loc = 5, space_loc = 465, amp = 2000, width = 4, height = 10)
synthetic_beam_img = VISARImage(fname = None, data = synthetic_beam.data, sweep_speed = synthetic_beam.sweep_speed, slit_size = synthetic_beam.slit_size)
synthetic_beam_img.save_tif("SyntheticData/20nsBeamReference.tif")

#get correction from the beam reference
ref = RefImage(fname = "SyntheticData/20nsBeamReference.tif", sweep_speed = 20, slit_size = 500)
ref.chop_beam(ybounds = (0, 430), num_slices = 25)
ref.save_chop_as_correction()

#Generate a Shot Reference
synthetic = SyntheticShot(sweep_speed = 20, slit_size = 500, time_points = 1000, space_points = 500)
velocity_profile = np.zeros(len(synthetic.time))
synthetic.generate_background(1000)
synthetic.generate_fringes(num_fringes = 10,
                            intensity = 2000,
                           velocity = velocity_profile,
                           vpf = 3.4,
                           fringe_max = 440,
                           fringe_min = 0)
synthetic.generate_fiducial(time_loc = 5, space_loc = 465, amp = 2000, width = 4, height = 10)
synthetic_shot_ref_img = VISARImage(fname = None, data = synthetic.data, sweep_speed = synthetic.sweep_speed, slit_size = synthetic.slit_size)
synthetic_shot_ref_img.apply_correction(ref.correction, negative = True)
synthetic_shot_ref_img.shear_data(1.3)
synthetic_shot_ref_img.chop_by_time(1.65, 20)
synthetic_shot_ref_img.chop_by_space(22, 500)
synthetic_shot_ref_img.chop_by_time(1.65, 20)
synthetic_shot_ref_img.save_tif("SyntheticData/20nsShotReference.tif")

#Generate Shot data
synthetic = SyntheticShot(sweep_speed = 20, slit_size = 500, time_points = 1000, space_points = 500)
velocity_profile = 5*np.tanh((synthetic.time - 10)*10) + 5
synthetic.generate_background(1000)
synthetic.generate_fringes(num_fringes = 10,
                            intensity = 2000,
                           velocity = velocity_profile,
                           vpf = 3.4,
                           fringe_max = 440,
                           fringe_min = 0)
synthetic.generate_fiducial(time_loc = 5, space_loc = 465, amp = 2000, width = 4, height = 10)
synthetic_shot_img = VISARImage(fname = None, data = synthetic.data, sweep_speed = synthetic.sweep_speed, slit_size = synthetic.slit_size)
synthetic_shot_img.apply_correction(ref.correction, negative = True)
synthetic_shot_img.shear_data(1.3)
#chop data to remove 0s added in from the correction
synthetic_shot_img.chop_by_time(1.65, 20)
synthetic_shot_img.chop_by_space(22, 500)
synthetic_shot_img.save_tif("SyntheticData/20nsShot.tif")


#show the synthetic images
fig = plt.figure(figsize = (12, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
for ax, title, img in zip([ax1, ax2, ax3], ["Beam", "Shot Reference", "Shot"], [synthetic_beam_img, synthetic_shot_ref_img, synthetic_shot_img]):
    ax.set_title(title)
    minval = 0.01 if img.data.min() <= 0 else img.data.min()
    img.show_data(ax, minmax = (500, img.data.max()))

fig.suptitle("Synthetic Data")
plt.tight_layout()

if show == True:
    plt.show()