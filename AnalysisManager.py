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
import warnings
import math
from matplotlib import gridspec
from matplotlib import ticker as mticker

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

def write_shot_report(folder:str):
    """
    Given a shot folder, write the report on that folder
    """
    #=====
    #Check whether folder is valid & has all data
    for subfolder in ["VISAR1", "VISAR2"]:
        for visar_folder in ["ShotRef", "Shot"]:
            if not os.path.exists(f"{folder}/{subfolder}/{visar_folder}"):
                raise Exception(f"folder invalid. {folder}/{subfolder}/{visar_folder} not found")
            #ensure info csv exists for all

            #ensure timing/lineouts exist for all
    #=====

    #=====
    #Set up the plot
    fig = plt.figure(figsize = (16, 9))
    gs0 = gridspec.GridSpec(4, 3, figure = fig)

    gsV1 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec = gs0[:2, 1:])
    gsV2 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec = gs0[2:, 1:])

    V1_axes = {}
    V2_axes = {}
    plot_names = ["beam", "corrected_beam", "shot", "corrected_shot", "timing", "phase_region", "fourier_bounds", "fourier_filtered",  "phase", "normalized_+_filtered"]

    V1_outer_gs = gridspec.GridSpec(1, 1)
    V1_outer_gs.update(bottom=.50,left=0.33333, right=1,top=1)
    V1_outer = fig.add_subplot(V1_outer_gs[:])
    #V1_outer.set_ylabel("VISAR 1", size = 25, fontweight = "bold")
    V1_outer.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    V1_outer.set_facecolor("lightsalmon")
    V2_outer_gs = gridspec.GridSpec(1, 1)
    V2_outer_gs.update(bottom=0,left=0.33333, right=1,top=0.5)
    V2_outer = fig.add_subplot(V2_outer_gs[:])
    V2_outer.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    V2_outer.set_facecolor("lightcyan")

    velo_outer_gs = gridspec.GridSpec(1, 1)
    velo_outer_gs.update(bottom=0,left=0, right=0.33333,top=0.75)
    velo_outer = fig.add_subplot(velo_outer_gs[:])
    velo_outer.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
    velo_outer.set_facecolor("thistle")

    for gs, axes, color in zip([gsV1, gsV2], [V1_axes, V2_axes], ["lightsalmon", "lightcyan"]):
        row = 0
        col = 0
        for name in plot_names:
            axis = fig.add_subplot(gs[row, col])
            axis.tick_params(axis = "x", direction = "in", pad = -15, labelsize = 5, labelrotation = 0)
            axis.tick_params(axis = "y", direction = "in", pad = -18, labelsize = 5, labelrotation = 0)
            axis.set_title(name.replace("_", " "))
            axes[name] = axis
            print(row, col)
            row = row if col != 4 else 1
            col = col + 1 if col != 4 else 0
            
        
    velo_ax = fig.add_subplot(gs0[1:4, 0])
    velo_ax.set_title("Velocities")
    velo_ax.tick_params(axis = "x", direction = "in", pad = -10)
    velo_ax.tick_params(axis = "y", direction = "in", pad = -18)
    velo_ax.set_ylabel("Free surface Velocity (km/s)")


    #Set the VISAR labels
    #fig.text(0.1, 0.1 "VISAR 1", size = "large")
    #fig.subplots_adjust(left = 0.2)
    #=====

    #=====
    #Plotting

    #Plot beams & corrections for each shot
    #for each VISAR:
        #1. Go to shot folder & look @ info to get beam ref folder, slit size & sweep speed
        #2. Go to beam folder and get fname for the beam from info.csv
        #3. Create VISAR img from fname & plot on "beam" axis
        #4. Correct VISAR img & plot on "corrected_beam" axis
        #5. Get lineouts from beam folder & use this to set the time axis bounds

    for visar, axes in zip(["VISAR1", "VISAR2"], [V1_axes, V2_axes]):
        #1
        shot_info = pd.read_csv(f"{folder}/{visar}/Shot/info.csv")
        beam_ref_folder = shot_info["beam_ref"].item()

        #2
        beam_info = pd.read_csv(f"{beam_ref_folder}/info.csv")
        beam_fname = beam_info["fpath"].item()
        sweep_speed = beam_info["sweep_speed"].item()

        #3
        beam_img = VISARImage(beam_fname, sweep_speed = sweep_speed) #make beam img
        beam_lineouts = pd.read_csv(f"{beam_ref_folder}/lineouts.csv") # get beamlineout
        beam_time = beam_lineouts["time"]
        beam_img.align_space()
        beam_img.align_time()
        beam_img.show_data(axes["beam"], minmax = (180, 270), ylabel = False)

        correction = ImageCorrection(f"{beam_ref_folder}/correction.csv")
        if visar == "VISAR3":
            beam_img.apply_correction(correction)
        beam_img.show_data(axes["corrected_beam"], minmax = (180, 270), ylabel = False)

        #VISAR
        shot_img = VISARImage(shot_info["fname"].item(), sweep_speed = sweep_speed)
        shot_lineouts = pd.read_csv(f"{folder}/{visar}/Shot/time.csv")
        if visar == "VISAR1":
            v1_time = shot_lineouts.time
        else:
            v2_time = shot_lineouts.time
        shot_img.show_data(axes["shot"], ylabel = False)
        shot_ref_info = pd.read_csv(f"{folder}/{visar}/ShotRef/info.csv")
        if visar == "VISAR1":
            shot_img.apply_correction(correction)
        shot_img.shear_data(shot_ref_info["shear"].values[0])
        shot_img.align_time(shot_lineouts.time)
        shot_img.show_data(axes["corrected_shot"], ylabel = False)
        shot_lineouts = pd.read_csv(f"{folder}/{visar}/Shot/time.csv")

        #plot the timing fiducials
        ax = axes["timing"]
        ax.plot(beam_lineouts.time, beam_lineouts.beam, label = "beam")
        ax.plot(beam_lineouts.time, beam_lineouts.fiducial, label=  "beam fiducial")
        ax.plot(shot_lineouts.time, shot_lineouts.fiducial, label = "shot fiducial")
        ax.legend(fontsize = "x-small")

        #plot lineout region
        analysis_info = pd.read_csv(f"{folder}/{visar}/Shot/analysis_params.csv")
        shot_img.show_data(axes["phase_region"], ylabel = False)
        ax = axes["phase_region"]
        ax.plot([analysis_info["x_bounds"][0], analysis_info["x_bounds"][0]], [analysis_info["y_bounds"][0], analysis_info["y_bounds"][1]], color = "k")
        ax.plot([analysis_info["x_bounds"][1], analysis_info["x_bounds"][1]], [analysis_info["y_bounds"][0], analysis_info["y_bounds"][1]], color = "k")
        ax.plot([analysis_info["x_bounds"][0], analysis_info["x_bounds"][1]], [analysis_info["y_bounds"][0], analysis_info["y_bounds"][0]], color = "k")
        ax.plot([analysis_info["x_bounds"][0], analysis_info["x_bounds"][1]], [analysis_info["y_bounds"][1], analysis_info["y_bounds"][1]], color = "k")
        ax.plot([analysis_info["ref_bounds"][0], analysis_info["ref_bounds"][0]], [analysis_info["y_bounds"][0], analysis_info["y_bounds"][1]], color = "lime")
        ax.plot([analysis_info["ref_bounds"][1], analysis_info["ref_bounds"][1]], [analysis_info["y_bounds"][0], analysis_info["y_bounds"][1]], color = "lime")


        #plot Fourier filter bounds, Fourier filtered data
        ref_lineout = shot_img.take_vert_lineout(analysis_info["ref_bounds"][0], analysis_info["ref_bounds"][1], analysis_info["y_bounds"][0], analysis_info["y_bounds"][1])
        fft_data = np.abs(fft(ref_lineout))
        freq = fftfreq(len(ref_lineout))
        ax = axes["fourier_bounds"]
        ax.plot(freq[:len(ref_lineout)//2], fft_data[0:len(ref_lineout)//2])
        ax.plot([analysis_info["fft_filter"][0], analysis_info["fft_filter"][0]], [fft_data.min(), fft_data.max()], color = "red")
        ax.plot([analysis_info["fft_filter"][1], analysis_info["fft_filter"][1]], [fft_data.min(), fft_data.max()], color = "red")
        ax.set_xlim(analysis_info["fft_filter"][0] - (analysis_info["fft_filter"][1] - analysis_info["fft_filter"][0])*0.1, analysis_info["fft_filter"][1] + (analysis_info["fft_filter"][1] - analysis_info["fft_filter"][0])*0.1)
        filtered_fft = fft_data[np.multiply(freq > analysis_info["fft_filter"][0], freq < analysis_info["fft_filter"][1])]
        ax.set_ylim(filtered_fft.min(), filtered_fft.max())

        #plot filtered data
        min_space = int((analysis_info["y_bounds"][0]- shot_img.space.min())/shot_img.space_per_pixel)
        max_space = int((analysis_info["y_bounds"][1] - shot_img.space.min())/shot_img.space_per_pixel)
        min_time = int((analysis_info["x_bounds"][0] - shot_img.time.min())/shot_img.time_resolution)
        max_time = int((analysis_info["x_bounds"][1] - shot_img.time.min())/shot_img.time_resolution)
        data_chunk = shot_img.data[min_space:max_space, min_time:max_time]
        fft_ = fft(data_chunk, axis = 0)
        #freq = np.abs(np.vstack([fftfreq(data_chunk.shape[1]) for i in range(data_chunk.shape[0])]))
        freq = fftfreq(data_chunk.shape[0])
        filter_mask = np.logical_and(freq > analysis_info["fft_filter"][0], freq < analysis_info["fft_filter"][1])
        filtered_fft = fft_ * filter_mask[:, np.newaxis]
        fourier_filtered_all = ifft(filtered_fft, n = data_chunk.shape[0], axis = 0).real
        fourier_filtered = fourier_filtered_all.real
        X, Y = np.meshgrid(np.linspace(analysis_info["x_bounds"][0], analysis_info["x_bounds"][1], fourier_filtered.shape[1] + 1), np.linspace(analysis_info["y_bounds"][0], analysis_info["y_bounds"][1], fourier_filtered.shape[0] + 1))
        heat_min = 0.01 if fourier_filtered.min() <= 0 else fourier_filtered.min()
        ax = axes["fourier_filtered"]
        ax.pcolormesh(X, Y, fourier_filtered, cmap='magma')

        #plot clean & unclean phase data
        #before cleaning
        phase = shot_img.get_phase_igor(
            x_bounds = (analysis_info["x_bounds"][0], analysis_info["x_bounds"][1]),
            y_bounds = (analysis_info["y_bounds"][0], analysis_info["y_bounds"][1]),
            fband = (analysis_info["fft_filter"][0], analysis_info["fft_filter"][1]),
            angle_deg = 0,
            use_hann = True,
            vpf = analysis_info["vpf"][0]
        )
        X, Y = np.meshgrid(np.linspace(analysis_info["x_bounds"][0], analysis_info["x_bounds"][1], fourier_filtered.shape[1] + 1), np.linspace(analysis_info["y_bounds"][0], analysis_info["y_bounds"][1], fourier_filtered.shape[0] + 1))
        ax = axes["phase"]
        phase_plot = ax.pcolormesh(X, Y, phase, cmap = "viridis")
        fig.colorbar(phase_plot, ax = ax)

        ax = axes["normalized_+_filtered"]
        #apply phase transformations
        if analysis_info["normalized"][0] == True:
            ref_folder = shot_info["shot_ref"][0]
            ref_info = pd.read_csv(f"{ref_folder}/info.csv")
            ref_img = VISARImage(ref_info["fname"][0], sweep_speed = sweep_speed)
            if visar == "VISAR1":
                ref_img.apply_correction(correction)
            ref_img.shear_data(shot_ref_info["shear"].values[0])
            ref_lineouts = pd.read_csv(f"{ref_folder}/time.csv")
            ref_img.align_time(ref_lineouts.time)
            ref_phase = ref_img.get_phase_igor(
                x_bounds = (analysis_info["x_bounds"][0], analysis_info["x_bounds"][1]),
                y_bounds = (analysis_info["y_bounds"][0], analysis_info["y_bounds"][1]),
                fband = (analysis_info["fft_filter"][0], analysis_info["fft_filter"][1]),
                angle_deg = 0,
                use_hann = True,
                vpf = analysis_info["vpf"][0]
            )
            print("=====\n", ref_phase.shape, phase.shape)
            print(len(shot_lineouts.time), len(ref_lineouts.time))
            #fix dimension mismatch if it exists
            if ref_phase.shape[1] != phase.shape[1]:
                t0 = analysis_info["x_bounds"][0]
                t1 = analysis_info["x_bounds"][1]
                # resample reference onto the shot’s x grid (keeps physical alignment)
                tx_shot = np.linspace(t0, t1, phase.shape[1])
                tx_ref  = np.linspace(t0, t1, ref_phase.shape[1])
                ref_phase = np.vstack([np.interp(tx_shot, tx_ref, row) for row in ref_phase])
            phase = phase - ref_phase
        
        if analysis_info["gauss_applied"][0] == True:
            phase = ndimage.gaussian_filter(phase, sigma = 2)

        if analysis_info["median_applied"][0] == True:
            phase = ndimage.median_filter(phase, size = (int(analysis_info["median_x"][0]), int(analysis_info["median_y"][0])))

        phase_plot = ax.pcolormesh(X, Y, phase, cmap = "viridis")
        fig.colorbar(phase_plot, ax = ax)
        #ax.legend()


    #plot the velocities against each other
    time = np.linspace(analysis_info["x_bounds"][0], analysis_info["x_bounds"][1], phase.shape[1])
    v1 = pd.read_csv(f"{folder}/VISAR1/Shot/velocity.csv")
    v2 = pd.read_csv(f"{folder}/VISAR2/Shot/velocity.csv")
    velo_ax.plot(v1.time, v1.velo, label = "VISAR 1", color = "red")
    velo_ax.plot(v2.time, v2.velo, label = "VISAR 2", color = "blue")
    velo_ax.legend()

    #Save the report to the folder

    #=====

    #fix plot ticks
    for axes in V1_axes, V2_axes:
        for key in axes.keys():
            axes[key].xaxis.set_major_locator(mticker.MaxNLocator(prune="both"))
            axes[key].yaxis.set_major_locator(mticker.MaxNLocator(prune="both"))
    velo_ax.xaxis.set_major_locator(mticker.MaxNLocator(prune="both"))
    velo_ax.yaxis.set_major_locator(mticker.MaxNLocator(prune="both"))

    plt.tight_layout()
    """
    plt.subplots_adjust(
        left=0.05,   # space on the left
        right=0.95,  # space on the right
        top=0.95,    # space at the top
        bottom=0.05, # space at the bottom
        wspace=0.1,  # horizontal space between subplots
        hspace=0.1   # vertical space between subplots
        )
    """   
    plt.subplots_adjust(right = 0.96)
    fig.text(0.97, 0.7, "VISAR 1", rotation = "vertical", size = 25, fontweight = "bold")
    fig.text(0.97, 0.2, "VISAR 2", rotation = "vertical", size = 25, fontweight = "bold")
    plt.show()

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
    def __init__(self, V1_beam_folder:str = None, V2_beam_folder:str = None, base_analysis_folder:str = None, data_folder:str = None):
        #test the inputs
        inputs = [V1_beam_folder, V2_beam_folder, base_analysis_folder, data_folder]
        input_test = [False if type(i) == type(None) else True for i in inputs]
        #if all are none, proceed w/o doing anything
        if sum(input_test) == 0:
            pass
        #if all have string inputs, set the folders
        elif math.prod(input_test) == 1:
            self.set_folders(V1_beam_folder, V2_beam_folder, base_analysis_folder, data_folder)
        #if some are string, but some are none, raise a warning
            warnings.warn("Not all inputs passed in. Use set_folders() method to set up the directory")
        self.V2_ref_aligned = False
        self.V2_aligned = False
        self.V1_ref_aligned = False
        self.V1_aligned = False
        self.V1_analyzed = False
        self.V2_analyzed = False
        self.get_info()

    def set_folders(self, V1_beam_folder:str, V2_beam_folder:str, base_analysis_folder:str, data_folder:str):
        self.V1_beam_folder = V1_beam_folder
        self.V2_beam_folder = V2_beam_folder
        self.base_analysis_folder = base_analysis_folder
        self.data_folder = data_folder
        self.test_base_folders()

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
            print(self.base_analysis_folder, self.V1_beam_folder, self.V2_beam_folder)
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

        #add info.csv for the analysis
        self.info = {
            "shot_ID": self.shot_ID,
        }
        print(self.info)
        info_df = pd.DataFrame(self.info, index = [0])
        info_df.to_csv("info.csv")

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
        self.V2_aligned = True

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

    def analyze(self, VISAR_num):
        if VISAR_num not in [1, 2]:
            raise Exception(f"VISAR must be 1 or 2. {VISAR_num} is invalid")
        if VISAR_num == 1:
            if self.V1_aligned == False:
                self.align_V1()
        else:
            if self.V2_aligned == False:
                self.align_V2()
        apply_beam_correction = False if VISAR_num == 2 else True #turn off the beam correction for VISAR 2
        analysis_plot = AnalysisPlot(f"{self.folder}/VISAR{VISAR_num}/Shot", apply_beam_correction)
        analysis_plot.show_plot()
        if VISAR_num == 2:
            self.V2_analyzed = True
        else:
            self.V1_analyzed = True

    def get_analysis_state(self, folder):
        """Generate an analysis state for the directory

        Args:
            dir (_type_): directory for the analysis

        Raises:
            Exception: folder does not exist
            Exception: folder exists, but proper file structure does not
            Exception: folder and VISAR folders exist, but proper file structure does not
        """
        analysis_state = {
            "VISAR1":{
                "ref_aligned": False,
                "shot_aligned": False,
                "has_velo": False
            },
            "VISAR1":{
                "ref_aligned": False,
                "shot_aligned": False,
                "has_velo": False
            }
        }
        #test for directory existence
        if os.path.exists(folder) == False: #throw exception if folder is invalid
            raise Exception(f"Folder {folder} is an invalid directory (could not be found)")
        if os.path.exists(f"{folder}/info.csv") == False:
            raise Exception(f"info.csv not found in {folder}")
        #look for VISAR1 & VISAR2 folders
        for V in ["VISAR1", "VISAR2"]:
            if os.path.exists(f"{folder}/{V}"):
            #look for Shot and ShotRef folders inside directory
                for subfolder in ["Shot", "ShotRef"]:
                    if os.path.exists(f"{folder}/{V}/{subfolder}"):
                        #get state of the analysis
                        if os.path.exists(f"{folder}/{V}/{subfolder}/time.csv") and os.path.exists(f"{folder}/{V}/{subfolder}/time.csv"):
                            key = "ref_aligned" if subfolder == "ShotRef" else "shot_aligned"
                            analysis_state[V][key] = True
                        if subfolder == "Shot":
                            if os.path.exists(f"{folder}/{V}/{subfolder}/analysis_parameters.csv") and os.path.exists(f"{folder}/{V}/{subfolder}/velocity.csv"):
                                analysis_state[V]["has_velo"] = True
                    else:
                        raise Exception(f"subfolder {folder}/{subfolder} does not exist")
            else:
                raise Exception(f"{V} not found in {folder}")

        #return True/False for pass/fail and the state of the current directory if valid
        return analysis_state

    def open_analysis(self, folder:str):
        """
        opening a saved analysis
        """
        state = self.get_analysis_state(folder)
        self.folder = folder
        self.V1_ref_aligned = state["VISAR1"]["ref_aligned"]
        self.V1_aligned = state["VISAR1"]["shot_aligned"]
        self.V1_analyzed = state["VISAR1"]["has_velo"]
        self.V2_ref_aligned = state["VISAR2"]["ref_aligned"]
        self.V2_aligned = state["VISAR2"]["shot_aligned"]
        self.V2_analyzed = state["VISAR2"]["has_velo"]

        

class AM:
    """General class for managing an analysis
    """
    def __init__(self, base_folder):
        self.base_folder = base_folder