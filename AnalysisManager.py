"""
Tools for managing an analysis
"""
import os
from VISAR import VISARImage, RefImage
import pandas as pd
from InteractivePlots import BeamAligner
import shutil

class SingleShotAnalysis:
    """
    Class for managing a single analysis
    """
    def __init__(self, shot_fname, ref_fname, sweep_speed, slit_size):
        self.space_calibrated = False
        self.time_calibrated = False
        self.etalons_defined = False
        self.phase_unwrap_region_defined = False
        self.fft_complete = False

class BeamAlignmentAnalysis:
    """
    Analysis for aligning a Beam reference
    """
    def __init__(self, fname, sweep_speed, slit_size):
        """
        fname: filename of the tif for the shot
        sweep_speed & slit_size are camera params
        """
        self.ref = RefImage(fname = fname, sweep_speed = sweep_speed, slit_size = slit_size)

    def check_folder(self, folder):
        """
        Checks the folder for the reference and sees if there exists analysis data
        """
        timing_lineout_found = False if "peak_timing.csv" not in os.listdir() else True
        image_correction_found = False if "correction.csv" not in os.listdir() else True
        return timing_lineout_found, image_correction_found

    def perform_alignment(self, folder):
        """
        Folder is the name of the folder to put the alignment in
        """
        self.aligner = BeamAligner(self.ref)
        self.aligner.set_lineout_save_name()

class AnalysisManager:
    """
    Managing an analysis
    """
    def __init__(self):
        pass

    def open_analysis(self, name):
        pass

    def create_new_analysis(self, base_directory):
        """
        Generates a new analysis with the specified name
        """
        self.shot_data = {}
        self.base_directory = base_directory
        self.data_info = pd.DataFrame({"shot":[],"sweep_time":[], "slit_size":[], "fname":[], "ref_file":[]})
        if not os.path.exists(base_directory):
            raise Exception("Cannot find Base Directory")
        if os.path.exists(base_directory):
            raise Exception(f"Analysis {base_directory} already exists")
        os.mkdir(f"{self.base_directory}") #make directory for the analysis
        os.mkdir(f"{self.base_directory}/Shots")
        os.mkdir(f"{self.base_directory}/TimingRefs")
        self.data_info.to_csv(f"{self.base_directory}/info.csv")

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

    def remove_analysis(self):
        #deletes an analysis
        shutil.rmtree(self.base_directory)