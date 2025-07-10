"""
Tools for managing an analysis
"""
import os
from VISAR import VISARImage, TimingRef
import pandas as pd

class AnalysisManager:
    """
    Managing an analysis
    """
    def __init__(self, base_directory):
        self.shot_data = {}
        self.base_directory = base_directory
        self.data_info = pd.DataFrame({"shot":[],"sweep_time":[], "slit_size":[]})
        if not os.path.exists(base_directory):
            raise Exception("Cannot find Base Directory")

    def open_analysis(self, name):
        pass

    def create_new_analysis(self, name):
        """
        Generates a new analysis with the specified name
        """
        self.name = name
        if name in os.listdir(self.base_directory):
            raise Exception(f"Analysis {name} already exists")
        os.mkdir(f"{self.base_directory}/{self.name}") #make directory for the analysis
        os.mkdir(f"{self.base_directory}/{self.name}/Shots")
        os.mkdir(f"{self.base_directory}/{self.name}/TimingRefs")
        self.data_info.to_csv(f"{self.base_directory}/{self.name}/info.csv")

    def add_shot_to_analysis(self, 
                             shot_file, 
                             ref_file,
                             sweep_speed,
                             slit_size):
        """
        Given tif files for the shot and reference, add them
        to the shot data dictionary

        shot_file: the tif file for the shot
        ref_file: the tif file for the reference
        sweep_speed: the sweep speed of the camera for the shot
        slit_size: slit width of the camera for the shot
        timing_ref: name of timing reference
        """
        #Read in data for the shot
        shot_label = f"{shot_file.split('/')[-1].lower().replace('.tif', '')}"
        ref_label = shot_label + "_ref"
        try:
            self.shot_data[shot_label] = VISARImage(fname = shot_file, 
                                                    sweep_speed = sweep_speed, 
                                                    slit_size = slit_size)
            self.shot_data[ref_label] = VISARImage(fname = ref_file, 
                                                    sweep_speed = sweep_speed, 
                                                    slit_size = slit_size)
        except:
            raise Exception("Files not found")
        
        #add info for the shot
        self.data_info.loc[len(self.data_info)] = [shot_label, sweep_speed, slit_size]
        self.data_info.to_csv(f"{self.base_directory}/{self.name}/info.csv")

        #create a folder for the shot
        if shot_label not in os.listdir(f"{self.base_directory}/{self.name}"):
            os.mkdir(f"{self.base_directory}/{self.name}/{shot_label}")