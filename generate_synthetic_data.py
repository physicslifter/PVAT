"""
Script for generating synthetic data

That also saves a CSV file for synthetic data
"""

from VISAR import VISARImage, RefImage
from SyntheticData import SyntheticBeamCalibration, SyntheticShot
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan


show = 0

if not os.path.exists("SyntheticData"):
    os.mkdir("SyntheticData")

# Shared parameters
shared_params = {
    "sweep_speed": 20,
    "slit_size": 500,
    "etalon": 20,
    "time_points": 1000,
    "space_points": 500,
    "time_loc": 5,
    "space_loc": 465,
    "amp": 2000,
    "width": 4,
    "height": 10,
}

# Type-specific
dataset_types = [
    {
        "name": "SyntheticBeam",
        "tif_file": "SyntheticData/20nsBeamReference.tif",
        "type": "beamref",
        "ref": nan,
    },
    {
        "name": "SyntheticShotRef",
        "tif_file": "SyntheticData/20nsShotReference.tif",
        "type": "shotref",
        "ref": nan,
    },
    {
        "name": "SyntheticShot",
        "tif_file": "SyntheticData/20nsShot.tif",
        "type": "shot",
        "ref": "SyntheticShotRef"
    },
]

rows = []
for entry in dataset_types:
    row = {**entry, **shared_params}
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("SyntheticData/synthetic_info.csv", index=False)

#Generating Beam Reference, Shot Reference, and Shot Data
images = []

for _, row in df.iterrows():
    params = row.to_dict()
    if params["type"] == "beamref":
        synthetic_beam = SyntheticBeamCalibration(
            params["sweep_speed"], params["slit_size"], params["time_points"], params["space_points"]
        )
        synthetic_beam.generate_background(500)
        synthetic_beam.generate_beam(3.5, 1, 2500, max_loc=430, shift=2/500)
        synthetic_beam.generate_fiducial(params["time_loc"], params["space_loc"], params["amp"], params["width"], params["height"])
        img = VISARImage(fname=None, data=synthetic_beam.data, sweep_speed=params["sweep_speed"], slit_size=params["slit_size"])
        img.save_tif(params["tif_file"])
        images.append((img, "Beam"))
        ref = RefImage(fname=params["tif_file"], sweep_speed=params["sweep_speed"], slit_size=params["slit_size"])
        ref.chop_beam(ybounds=(0, 430), num_slices=25)
        ref.save_chop_as_correction()
    elif params["type"] == "shotref":
        synthetic = SyntheticShot(
            params["sweep_speed"], params["slit_size"], params["time_points"], params["space_points"]
        )
        velocity_profile = np.zeros(params["time_points"])
        synthetic.generate_background(1000)
        synthetic.generate_fringes(
            num_fringes=10,
            intensity=2000,
            velocity=velocity_profile,
            vpf=3.4,
            fringe_max=440,
            fringe_min=0,
        )
        synthetic.generate_fiducial(params["time_loc"], params["space_loc"], params["amp"], params["width"], params["height"])
        img = VISARImage(fname=None, data=synthetic.data, sweep_speed=params["sweep_speed"], slit_size=params["slit_size"])
        img.apply_correction(ref.correction, negative=True)
        img.shear_data(1.3)
        img.chop_by_time(1.65, 20)
        img.chop_by_space(22, 500)
        img.chop_by_time(1.65, 20)
        img.save_tif(params["tif_file"])
        images.append((img, "Shot Reference"))
    elif params["type"] == "shot":
        synthetic = SyntheticShot(
            params["sweep_speed"], params["slit_size"], params["time_points"], params["space_points"]
        )
        velocity_profile = 5 * np.tanh((synthetic.time - 10) * 10) + 5
        synthetic.generate_background(1000)
        synthetic.generate_fringes(
            num_fringes=10,
            intensity=2000,
            velocity=velocity_profile,
            vpf=3.4,
            fringe_max=440,
            fringe_min=0,
        )
        synthetic.generate_fiducial(params["time_loc"], params["space_loc"], params["amp"], params["width"], params["height"])
        img = VISARImage(fname=None, data=synthetic.data, sweep_speed=params["sweep_speed"], slit_size=params["slit_size"])
        img.apply_correction(ref.correction, negative=True)
        img.shear_data(1.3)
        img.chop_by_time(1.65, 20)
        img.chop_by_space(22, 500)
        img.save_tif(params["tif_file"])
        images.append((img, "Shot"))


#show the synthetic images
fig = plt.figure(figsize = (12, 5))
for i, (img, title) in enumerate(images):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(title)
    minval = 0.01 if img.data.min() <= 0 else img.data.min()
    img.show_data(ax, minmax=(500, img.data.max()))
fig.suptitle("Synthetic Data")
plt.tight_layout()

if show == True:
    plt.show()