"""
Tests for AnalysisManager
"""
from AnalysisManager import AnalysisManager
import datetime

parent_directory = "./python_analysis"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

manager = AnalysisManager(parent_directory)
manager.create_new_analysis(timestamp)  # creates ./python_analysis/{timestamp}
manager.name = timestamp

shot_file = "../VISAR1/0409_1753_Shot65_Visar1.tif"
ref_file = "../VISAR1/0409_1746_Shot65_Visar1_ref.tif"
shot_label = shot_file.split('/')[-1].lower().replace('.tif', '')
ref_label = ref_file.split('/')[-1].lower().replace('.tif', '')

manager.add_shot_to_analysis(
    shot_file = shot_file,
    ref_file = ref_file,
    sweep_speed = 20,
    slit_size = 500
)
manager.organize_analysis_files(shot_label, ref_label)
