"""
Script for setting up tutorials and analysis workflow
after downloading the repository
"""
import os
import pandas as pd

#Generate the Synthetic Data
import generate_synthetic_data

#create Analysis Folder
if not os.path.exists("Analysis"):
    os.mkdir("Analysis")

#create info excel file for Synthetic analysis data
df = pd.DataFrame({"Name": [], #Name of the data (can be anything)
                   "Type": [], #type of the data (beam_ref, shot_ref, or shot)
                   "Fname": [], #file for the shot
                   "sweep_speed": [], #sweep speed for the shot
                   "slit_size": [], #slit size of the camera
                   "etalon": [] #the width of the etalon
                   })
# Add synthetic beam reference
df.loc[0] = [
    "SyntheticBeam",
    "beam_ref",
    "../SyntheticData/20nsBeamReference.tif",
    20,
    500,
    20
]

#synthetic shot reference
df.loc[1] = [
    "SyntheticShotRef",
    "shot_ref",
    "../SyntheticData/20nsShotReference.tif",
    20,
    500,
    20
]

#synthetic shot reference
df.loc[2] = [
    "SyntheticShot",
    "shot",
    "../SyntheticData/20nsShot.tif",
    20,
    500,
    20
]

#save excel to Analysis folder
df.to_excel("Analysis/info.xlsx")