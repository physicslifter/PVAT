"""
Tools for managing an analysis

(Currently SingleShotAnalysis & BeamAlignmentAnalysis Unused...)

"""

import os
from VISAR import VISARImage, RefImage, TimingRef
import pandas as pd
from InteractivePlots import BeamAligner
import shutil
import re
from datetime import datetime

INFO_COLS = [
    "Name", "Shot no.", "Visar", "Type", "Filename", "Filepath", "DataSource", "sweep_speed", "slit_size", "etalon",
    "Analysis_Path", "Date_Analyzed", "Notes"
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
            "Filename": base_noext, #safe_str(row.get("Filename", tif_file)),
            "Filepath": safe_str(row.get("filepath", tif_file)),
            "sweep_speed": safe_float(row.get("sweep_time", ""), default=20),
            "slit_size": safe_float(row.get("slit_size", ""), default=500),
            "etalon": safe_str(row.get("etalon", "")),
        }
        
        info_row["DataSource"] = "Synthetic Data" if "synthetic" in csv_path.lower() else "Real Data"
        return info_row

    def save_analysis_instance(
        self,
        data_type,
        base_name,
        info_row,
        notes=""
    ):
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
        row = {k: safe_str(v) for k, v in info_row.items()}  
        shot_no, visar = extract_shot_visar(row.get("Name", ""))
        row.update({
            # "Shot no.": shot_no,
            # "Visar": visar,
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


# Example:
# if __name__ == "__main__":
#     am = AnalysisManager()
#     am.create_or_open_analysis("Analysis1")
#     tif_file = "0404_1525_Shot38_Visar2_ref.tif"
#     csv_path = "data/real_info.csv"
#     info_row = am.extract_info_from_csv(tif_file, csv_path)
#     folder = am.save_analysis_instance(
#         data_type=info_row["Type"] if info_row["Type"] else "Shot",
#         base_name=info_row["Name"] if info_row["Name"] else "Unknown",
#         info_row=info_row
#     )
#     print(f"Saved new analysis instance at: {folder}")
#     print("Current versions:", am.list_versions(info_row["Type"], info_row["Name"]))