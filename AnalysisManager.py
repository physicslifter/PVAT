"""
Tools for managing an analysis

(Currently SingleShotAnalysis & BeamAlignmentAnalysis Unused...)

"""

import os
from VISAR import *
import pandas as pd
from InteractivePlots import *
import shutil
import re
from datetime import datetime

INFO_COLS = [
    "Name", "Shot no.", "Visar", "Type", "Filename", "Filepath", "DataSource", "sweep_speed", 
    "slit_size", "etalon", "beam_ref_path", "Analysis_Path", "Date_Analyzed", "Notes"
]

def ensure_info_xlsx(path):
    """Ensure an info.xlsx exists with the correct columns."""
    if not os.path.exists(path):
        df = pd.DataFrame(columns=INFO_COLS)
        df.to_excel(path, index=False)

def append_info_xlsx(path, row):
    """Append a row to the info.xlsx at the given path."""
    ensure_info_xlsx(path)
    df = pd.read_excel(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(path, index=False)

def get_next_version(base_folder, base_name):
    """Return the next available version number for a given analysis."""
    if not os.path.exists(base_folder):
        return 1
    existing = [
        d for d in os.listdir(base_folder)
        if d.startswith(base_name + "_") and os.path.isdir(os.path.join(base_folder, d))
    ]
    versions = []
    for d in existing:
        try:
            v = int(d.split("_")[-1])
            versions.append(v)
        except Exception:
            continue
    return max(versions, default=0) + 1

def normalize_path(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))

def safe_str(val, default="None"):
    if pd.isnull(val) or val is None:
        return default
    return str(val)

def strip_version_suffix(name):
    return re.sub(r'_\d+$', '', name)

def safe_float(val, default=None):
    try:
        if pd.isnull(val) or val is None or val == '' or str(val).lower() == 'nan':
            return default
        return float(val)
    except Exception:
        return default

def extract_shot_visar(name):
    """Extract ShotNo and Visar from Name like 'Shot38_Visar2'."""
    m = re.match(r'Shot(\d+)_Visar(\d+)', str(name))
    if m:
        return m.group(1), m.group(2)
    return None, None

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
    def __init__(self, base_directory="Analysis"):
        self.base_directory = base_directory
        os.makedirs(self.base_directory, exist_ok=True)
        self.global_info_path = os.path.join(self.base_directory, "info.xlsx")
        ensure_info_xlsx(self.global_info_path)
        self.analysis_name = None
        self.analysis_path = None
        self.analysis_info_path = None

    def create_or_open_analysis(self, analysis_name):
        """Create or open named analysis folder."""
        self.analysis_name = analysis_name
        self.analysis_path = os.path.join(self.base_directory, analysis_name)
        os.makedirs(self.analysis_path, exist_ok=True)
        self.analysis_info_path = os.path.join(self.analysis_path, "info.xlsx")
        ensure_info_xlsx(self.analysis_info_path)
        for sub in ["Shot", "ShotRef", "BeamRef", "Other"]:
            os.makedirs(os.path.join(self.analysis_path, sub), exist_ok=True)

    def extract_info_from_csv(self, tif_file, csv_path):
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [col.lower() for col in df.columns]
        norm_input = normalize_path(tif_file)
        base = os.path.basename(tif_file)
        base_noext = os.path.splitext(base)[0]
    
        df['norm_filename'] = df.get('filename', pd.Series([""]*len(df))).fillna("").apply(normalize_path)
        df['norm_filepath'] = df.get('filepath', pd.Series([""]*len(df))).fillna("").apply(normalize_path)
    
        matches = df[df['norm_filepath'] == norm_input]
        if matches.empty:
            matches = df[df['norm_filename'] == norm_input]
        if matches.empty:
            matches = df[df['filename'].astype(str).apply(lambda x: os.path.basename(x) == base)]
        if matches.empty:
            matches = df[df['filename'].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0] == base_noext)]
        if matches.empty:
            raise ValueError(f"No entry found in {csv_path} for file {tif_file}")
    
        row = matches.iloc[0]
        
        shot_no = safe_str(row.get("shot no.", ""))
        visar = safe_str(row.get("visar", ""))
        file_type = safe_str(row.get("type", ""))
        name = safe_str(row.get("name", ""))
        
        if not name or name.lower() in ["none", "nan"]:
            if shot_no and shot_no.lower() not in ["none", "nan", ""]:
                if visar and visar.lower() not in ["none", "nan", ""]:
                    name = f"Shot{int(float(shot_no))}_Visar{int(float(visar))}"
                else:
                    name = f"Shot{int(float(shot_no))}"
            else:
                name = base_noext
                
        info_row = {
            "Name": name,
            "Shot no.": shot_no,
            "Visar": visar,
            "Type": file_type,
            "Filename": base_noext,
            "Filepath": safe_str(row.get("filepath", tif_file)),
            "sweep_speed": safe_float(row.get("sweep_time", ""), default=20),
            "slit_size": safe_float(row.get("slit_size", ""), default=500),
            "etalon": safe_str(row.get("etalon", "")),
        }
        
        info_row["DataSource"] = "Synthetic Data" if "synthetic" in csv_path.lower() else "Real Data"
        return info_row

    def save_analysis_instance(self, data_type, base_name, info_row, notes=""):
        """
        Create a new versioned analysis instance and update all info files.
        Returns the instance folder path.
        """
        parent_folder = os.path.join(self.analysis_path, safe_str(data_type))

        os.makedirs(parent_folder, exist_ok=True)
        base_folder = os.path.join(parent_folder, safe_str(base_name))

        os.makedirs(base_folder, exist_ok=True)
        version = get_next_version(base_folder, safe_str(base_name))
        instance_folder = os.path.join(base_folder, f"{safe_str(base_name)}_{version}")
        os.makedirs(instance_folder, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {k: str(v) for k, v in info_row.items()}

        if str(data_type).lower() in ["shot", "shotref"]:
            if "beam_ref_path" not in row or not row["beam_ref_path"]:
                raise ValueError("beam_ref must be set for Shot/ShotRef analyses.")
        elif str(data_type).lower() == "beamref":
            row["beam_ref_path"] = ""  #maybe add the delete column here...
        else:
            row["beam_ref_path"] = ""

        row.update({
            "Analysis_Path": os.path.abspath(instance_folder),
            "Date_Analyzed": now,
            "Notes": notes
        })
        instance_info_path = os.path.join(instance_folder, "info.xlsx")
        pd.DataFrame([row], columns=INFO_COLS).to_excel(instance_info_path, index=False)
        append_info_xlsx(self.analysis_info_path, row)
        append_info_xlsx(self.global_info_path, row)
        return instance_folder

    def list_versions(self, data_type, base_name):
        """List versions for analysis base name."""
        parent_folder = os.path.join(self.analysis_path, data_type)
        base_folder = os.path.join(parent_folder, base_name)
        if not os.path.exists(base_folder):
            return []
        return sorted([
            d for d in os.listdir(base_folder)
            if d.startswith(base_name + "_") and os.path.isdir(os.path.join(base_folder, d))
        ])

    def get_instance_info(self, data_type, base_name, version):
        """Retrieve info for a specific analysis."""
        instance_folder = os.path.join(
            self.analysis_path, data_type, base_name, f"{base_name}_{version}"
        )
        info_path = os.path.join(instance_folder, "info.xlsx")
        if os.path.exists(info_path):
            return pd.read_excel(info_path).iloc[0].to_dict()
        else:
            return None

    def get_latest_version(self, data_type, base_name):
        versions = self.list_versions(data_type, base_name)
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.split('_')[-1]))
        return latest

    def duplicate_version(self, data_type, base_name):
        versions = self.list_versions(data_type, base_name)
        if not versions:
            raise Exception("No existing version to duplicate.")
        latest_version = max(versions, key=lambda v: int(v.split('_')[-1]))
        src_folder = os.path.join(self.analysis_path, data_type, base_name, latest_version)
        next_version_num = int(latest_version.split('_')[-1]) + 1
        dest_folder = os.path.join(self.analysis_path, data_type, base_name, f"{base_name}_{next_version_num}")
        shutil.copytree(src_folder, dest_folder)
        return dest_folder

    def remove_analysis(self, analysis_name):
        """Deletes analysis and data."""
        analysis_path = os.path.join(self.base_directory, analysis_name)
        if os.path.exists(analysis_path):
            shutil.rmtree(analysis_path)

class BeamAM:
    def __init__(self, base_analysis_folder:str):
        if os.path.exists(base_analysis_folder) == False:
            raise Exception("Base analysis folder not found")
        self.base_analysis_folder = base_analysis_folder

    def open_analysis(self, folder_path:str):
        """Opens an analysis form the specified folder

        Args:
            folder_path (str): path for an existing beam analysis
        """
        try:
            print(f"{self.base_analysis_folder}/{folder_path}")
            folder_contents = os.listdir(f"{self.base_analysis_folder}/{folder_path}")
        except:
            raise Exception(f"{folder_path} is invalid")
        files_if_completed = ["correction.csv", "lineouts.csv", "info.csv"]#these files are present in the beam analysis folder if the analysis has been performed
        has_files_test = [i in folder_contents for i in files_if_completed]
        for file, result in zip(files_if_completed, has_files_test):
            if result == False:
                raise Exception(f"{file} not found in {self.base_analysis_folder}/{folder_path}")
        self.correction = ImageCorrection(f"{self.base_analysis_folder}/{folder_path}/correction.csv")
        self.lineout = pd.read_csv(f"{self.base_analysis_folder}/{folder_path}/lineouts.csv")
        self.info = pd.read_csv(f"{self.base_analysis_folder}/{folder_path}/info.csv")
        self.sweep_speed = self.info.sweep_speed
        self.slit_size = self.info.slit_size
        self.fpath = self.info.fpath

    def create_analysis(self, folder, sweep_speed, slit_size, img_path):
        self.sweep_speed = sweep_speed
        self.slit_size = slit_size
        self.fpath = img_path
        #check if the folder exists
        if os.path.exists(f"{self.base_analysis_folder}/{folder}"):
            raise Exception("Folder already exists, try a different name")
        #check if img file exists
        if os.path.exists(img_path) == False:
            raise Exception("img file name not valid")
        #if folder doesn't already exist, set up the file structure
        os.mkdir(f"{self.base_analysis_folder}/{folder}")
        ref_img = RefImage(fname = img_path, 
                           folder = f"{self.base_analysis_folder}/{folder}",
                           sweep_speed = sweep_speed,
                           slit_size = slit_size
                           )
        beam_aligner = BeamAligner(ref_img)
        beam_aligner.show_plot()

class ShotAM:
    def __init__(self, V1_beam_folder:str, V2_beam_folder:str, base_analysis_folder:str, data_folder:str):
        """Initialize

        Args:
            beam_folder (str): _description_
            base_analysis_folder (str): _description_
            data_folder (str): folder where VISAR data is stored
        """
        self.V1_beam_folder = V1_beam_folder
        self.V2_beam_folder = V2_beam_folder
        self.base_analysis_folder = base_analysis_folder
        self.data_folder = data_folder
        self.V2_ref_aligned = False
        self.V2_aligned = False
        self.V1_ref_aligned = False
        self.V1_aligned = False
        self.test_base_folders()
        self.get_info()

    def test_base_folders(self):
        """
        Tests validity of data directory and beam folder
        """
        #test beam alignment folder
        try:
            #if base analysis is not valid, we'll catch it in the BeamAM
            self.V1_beam_analysis = BeamAM(base_analysis_folder = self.base_analysis_folder)
            self.V2_beam_analysis = BeamAM(base_analysis_folder = self.base_analysis_folder)
            self.V1_beam_analysis.open_analysis(self.V1_beam_folder)
            self.V2_beam_analysis.open_analysis(self.V2_beam_folder)
        except:
            raise Exception(f"Folders are invalid")
        
    def get_info(self):
        """
        Pull information from the real_info.csv
        """
        self.info = pd.read_csv("real_info.csv")

    def create_new_analysis(self, shot_ID):
        """Creates a new analysis for the shot

        Args:
            beam_ref (str, optional): folder path for the beam reference to use Defaults to None.
        """
        #test if information for this shot exists
        try:
            self.shot_data = self.info[self.info["Shot no."] == shot_ID]
        except:
            raise Exception("Info not found for this shot")
        self.get_shot_data(shot_ID)
        self.shot_specified = True
        self.shot_ID = shot_ID
        #if info exists, set up the folder structure
        self.setup_file_structure()
        
    def setup_file_structure(self):
        if self.shot_specified == False:
            raise Exception("shot not yet specified")
        #check if a folder exists for the shot
        if not os.path.exists(f"{self.base_analysis_folder}/{self.shot_ID}"):
            os.mkdir(f"{self.base_analysis_folder}/{self.shot_ID}")

        #get index of the current analysis
        index = len(os.listdir(f"{self.base_analysis_folder}/{self.shot_ID}"))

        #set up folders
        self.folder = f"{self.base_analysis_folder}/{self.shot_ID}/{index}"
        os.mkdir(self.folder)
        for dir in ["VISAR1", "VISAR2"]:
            os.mkdir(f"{self.folder}/{dir}")
            for dir2 in ["ShotRef", "Shot"]:
                os.mkdir(f"{self.folder}/{dir}/{dir2}")

    def get_shot_data(self, Shot_ID):
        shot_data = self.shot_data
        self.laser_power = shot_data["Laser Power (W/cm^2)"].values[0]
        self.sweep_speed = shot_data["sweep_time"].values[0]
        self.slit_size = None
        V1_data = shot_data[shot_data.VISAR == 1]
        V2_data = shot_data[shot_data.VISAR == 2]
        self.V1_etalon = V1_data.etalon.values[0]
        self.V2_etalon = V2_data.etalon.values[0]
        self.V1_fname = V1_data[V1_data.Type == "Shot"].filepath.values[0]
        self.V1_ref_fname = V1_data[V1_data.Type == "ShotRef"].filepath.values[0]
        self.V2_fname = V2_data[V2_data.Type == "Shot"].filepath.values[0]
        self.V2_ref_fname = V2_data[V2_data.Type == "ShotRef"].filepath.values[0]

    def open_saved_analysis(self):
        pass

    def align_V2_ref(self):
        shot_ref = VISARImage(
            f"{self.data_folder}/VISAR2/{self.V2_ref_fname}",
            sweep_speed = self.sweep_speed,
            slit_size = self.slit_size
        )
        shot_ref_aligner = ShotRefAligner(shot_ref)
        shot_ref_aligner.set_beam_ref_folder(f"{self.base_analysis_folder}/{self.V2_beam_folder}")
        shot_ref_aligner.set_folder(f"{self.folder}/VISAR2/ShotRef")
        shot_ref_aligner.show_plot()
        self.V2_ref_aligned = True
            
    def align_V2(self):
        if self.V2_ref_aligned == False:
            self.align_V2_ref()
        shot = VISARImage(
            fname = f"{self.data_folder}/VISAR2/{self.V2_fname}",
            sweep_speed = self.sweep_speed,
            slit_size = self.slit_size
        )
        shot_aligner = ShotAligner(shot)
        shot_aligner.set_beam_ref_folder(f"{self.base_analysis_folder}/{self.V2_beam_folder}")
        shot_aligner.set_shot_ref_folder(f"{self.folder}/VISAR2/ShotRef")
        shot_aligner.set_folder(f"{self.folder}/VISAR2/Shot")
        shot_aligner.show_plot()
        self.V1_aligned = True

    def align_V1_ref(self):
        shot_ref = VISARImage(
            f"{self.data_folder}/VISAR1/{self.V1_ref_fname}",
            sweep_speed = self.sweep_speed,
            slit_size = self.slit_size
        )
        shot_ref_aligner = ShotRefAligner(shot_ref)
        shot_ref_aligner.set_beam_ref_folder(f"{self.base_analysis_folder}/{self.V1_beam_folder}")
        shot_ref_aligner.set_folder(f"{self.folder}/VISAR1/ShotRef")
        shot_ref_aligner.show_plot()
        self.V1_ref_aligned = True
            
    def align_V1(self):
        if self.V1_ref_aligned == False:
            self.align_V1_ref()
        
        shot = VISARImage(
            fname = f"{self.data_folder}/VISAR1/{self.V1_fname}",
            sweep_speed = self.sweep_speed,
            slit_size = self.slit_size
        )
        shot_aligner = ShotAligner(shot)
        shot_aligner.set_beam_ref_folder(f"{self.base_analysis_folder}/{self.V1_beam_folder}")
        shot_aligner.set_shot_ref_folder(f"{self.folder}/VISAR1/ShotRef")
        shot_aligner.set_folder(f"{self.folder}/VISAR1/Shot")
        shot_aligner.show_plot()
        self.V1_aligned = True

class AM:
    """General class for managing an analysis
    """
    def __init__(self, base_folder):
        self.base_folder = base_folder