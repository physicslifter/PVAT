"""
Tools for managing an analysis
"""
import os
from VISAR import VISARImage, RefImage, TimingRef
import pandas as pd
from InteractivePlots import BeamAligner
import shutil
import re

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
    def __init__(self, base_directory):
        self.shot_data = {}
        self.base_directory = base_directory
        self.name = None  # Will be set when creating/opening an analysis
        self.data_info = pd.DataFrame({"shot": [], "sweep_time": [], "slit_size": [], "fname": [], "ref_file": []})

    @property
    def analysis_path(self):
        if self.name is None:
            raise Exception("No analysis selected.")
        return os.path.join(self.base_directory, self.name)

    def open_analysis(self, name):
        self.name = name
        info_csv = os.path.join(self.analysis_path, "info.csv")
        if os.path.exists(info_csv):
            self.data_info = pd.read_csv(info_csv)
        else:
            self.data_info = pd.DataFrame({"shot": [], "sweep_time": [], "slit_size": [], "fname": [], "ref_file": []})

    def create_new_analysis(self, name):
        """
        Generates a new analysis with the specified name
        """
        self.name = name
        if os.path.exists(self.analysis_path):
            raise Exception(f"Analysis {name} already exists at {self.analysis_path}")
        os.makedirs(os.path.join(self.analysis_path, "Shots"))
        os.makedirs(os.path.join(self.analysis_path, "TimingRefs"))
        self.data_info.to_csv(os.path.join(self.analysis_path, "info.csv"), index=False)

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
        shot_label = os.path.splitext(os.path.basename(shot_file))[0].lower()
        ref_label = os.path.splitext(os.path.basename(ref_file))[0].lower()
        try:
            self.shot_data[shot_label] = VISARImage(fname=shot_file, sweep_speed=sweep_speed, slit_size=slit_size)
            self.shot_data[ref_label] = VISARImage(fname=ref_file, sweep_speed=sweep_speed, slit_size=slit_size)
        except Exception as e:
            raise Exception(f"Files not found or error loading images: {e}")

        # Add info for the shot
        self.data_info.loc[len(self.data_info)] = [shot_label, sweep_speed, slit_size, shot_file, ref_file]
        self.data_info.to_csv(os.path.join(self.analysis_path, "info.csv"), index=False)

        # Create a folder for the shot if needed
        shots_folder = os.path.join(self.analysis_path, "Shots")
        os.makedirs(shots_folder, exist_ok=True)
        shot_subfolder = os.path.join(shots_folder, shot_label)
        if shot_label not in os.listdir(shots_folder):
            os.mkdir(shot_subfolder)

    def organize_analysis_files(self, shot_label, ref_label):
        """
        Moves ref and shot output files into the correct subfolders.
        Expects these files in the current directory:
          - ref_lineout.csv
          - ref_lineout.png
          - ref_correction.csv
          - shot_lineout.csv
          - shot_lineout.png
        """
        shot_folder = os.path.join(self.analysis_path, "Shots", shot_label)
        ref_folder = os.path.join(self.analysis_path, "TimingRefs", ref_label)
        os.makedirs(shot_folder, exist_ok=True)
        os.makedirs(ref_folder, exist_ok=True)

        #Move shot files
        shot_files = ["shot_lineout.csv", "shot_lineout.png"]
        for fname in shot_files:
            src = fname
            dst = os.path.join(shot_folder, fname)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"Moved {src} -> {dst}")
            else:
                print(f"File not found: {src}")

        #Move ref files
        ref_files = ["ref_lineout.csv", "ref_lineout.png", "ref_correction.csv"]
        for fname in ref_files:
            src = fname
            dst = os.path.join(ref_folder, fname)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"Moved {src} -> {dst}")
            else:
                print(f"File not found: {src}")

        print("File organization complete.")

    def update_info_csv_from_folders(self):
        """
        Scans the Shots and TimingRefs folders and updates info.csv
        based on the actual files present.
        Assumes that for each file, the filename encodes sweep_time and slit_size,
        e.g., shot1_20ns_500um.tif
        """
        shots_folder = os.path.join(self.analysis_path, "Shots")
        timingrefs_folder = os.path.join(self.analysis_path, "TimingRefs")
        rows = []

        def extract_metadata(filename):
            # Example filename: shot1_20ns_500um.tif
            match = re.match(r"(.+?)_(\d+)ns_(\d+)um\.tif", filename)
            if match:
                shot = match.group(1)
                sweep_time = int(match.group(2))
                slit_size = int(match.group(3))
                return shot, sweep_time, slit_size
            else:
                return filename, None, None

        #Scan Shots folder
        for fname in os.listdir(shots_folder):
            if fname.endswith(".tif"):
                shot, sweep_time, slit_size = extract_metadata(fname)
                rows.append({"shot": shot, "sweep_time": sweep_time, "slit_size": slit_size, "fname": fname, "ref_file": None})

        #Scan TimingRefs folder
        for fname in os.listdir(timingrefs_folder):
            if fname.endswith(".tif"):
                shot, sweep_time, slit_size = extract_metadata(fname)
                rows.append({"shot": shot, "sweep_time": sweep_time, "slit_size": slit_size, "fname": None, "ref_file": fname})

        info_df = pd.DataFrame(rows)
        info_csv_path = os.path.join(self.analysis_path, "info.csv")
        info_df.to_csv(info_csv_path, index=False)
        print(f"Updated info.csv with {len(rows)} entries.")

    def remove_analysis(self):
        #deletes an analysis
        shutil.rmtree(self.base_directory)

class AM2:
    """
    Simple analysis manager to read that works from excel file in Analysis folder
    """
    def __init__(self, base_folder):
        self.base_folder = base_folder
        if not os.path.exists(self.base_folder):
            raise Exception("Base folder not found")
        self.get_excel()
        self.has_folder = False

    def get_excel(self):
        """
        Pulls info from the base folder
        """
        self.info = pd.read_excel(f"{self.base_folder}/info.xlsx")
        self.beam_refs = self.info.where(self.info["Type"] == "beam_ref")["Name"].dropna().values
        self.shot_refs = self.info.where(self.info["Type"] == "shot_ref")["Name"].dropna().values
        self.shots = self.info.where(self.info["Type"] == "shot")["Name"].dropna().values

    def get_filename(self, name):
        "Given an img name, get the file for the img"
        return self.info.where(self.info["Name"] == name).dropna()["Fname"].values[0]

    def create_new_analysis(self, name):
        self.analysis_name = name
        os.mkdir(f"{self.base_folder}/{name}")
        os.mkdir(f"{self.base_folder}/{name}/Shots")
        os.mkdir(f"{self.base_folder}/{name}/ShotRefs")
        os.mkdir(f"{self.base_folder}/{name}/BeamRefs")
        self.has_folder = True

    def open_analysis(self, name):
        self.analysis_name = name
        if name not in os.listdir(self.base_folder):
            raise Exception("Analysis does not exist")

    def analyze_beam_ref(self, name, analysis_name=None):
        if name not in self.beam_refs:
            raise Exception(f"{name} not found in beam references")
        if self.has_folder == False:
            raise Exception("Folder must be created before ref can be analyzed")
        if analysis_name != None and analysis_name in os.listdir(f"{self.base_folder}/{self.analysis_name}/BeamRefs"):
            raise Exception("This name has already been saved")
        info = self.info.where(self.info["Name"] == name).dropna()
        prev_analyses = [i for i in os.listdir(f"{self.base_folder}/{self.analysis_name}/BeamRefs") if name in i]
        analysis_name = f"{name}_{len(prev_analyses)}" if analysis_name == None else analysis_name
        beam_analysis_directory = f"{self.base_folder}/{self.analysis_name}/BeamRefs/{name}/{analysis_name}"
        if not os.path.exists(f"{self.base_folder}/{self.analysis_name}/BeamRefs/{name}"):
            os.mkdir(f"{self.base_folder}/{self.analysis_name}/BeamRefs/{name}")
        os.mkdir(beam_analysis_directory) #make directory for the shot analysis
        ref = RefImage(fname = info["Fname"].values[0], folder = beam_analysis_directory, sweep_speed = info["sweep_speed"].values[0], slit_size = info["slit_size"].values[0])
        aligner = BeamAligner(ref_img = ref)
        aligner.initialize_plot()
        aligner.show_plot()

    def get_shot_ref_analyses(self, name):
        """
        Given an img name, this returns a list
        of the analyses for that shot
        """
        return [i for i in os.listdir(f"{self.base_folder}/{self.analysis_name}/BeamRefs/{name}")]
    
    def get_shot_analyses(self, name):
        return [i for i in os.listdir(f"{self.base_folder}/{self.analysis_name}/Shots/{name}")]
    
    def get_shot_ref_analyses(self, name):
        return [i for i in os.listdir(f"{self.base_folder}/{self.analysis_name}/Shot_Refs/{name}")]        

        