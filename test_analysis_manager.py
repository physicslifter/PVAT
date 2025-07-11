"""
Tests for AnalysisManager
"""
from AnalysisManager import AnalysisManager
from pdb import set_trace as st

analysis_directory = "../"

test_folder_setup = True

if test_folder_setup == True:
    manager = AnalysisManager(analysis_directory)
    manager.create_new_analysis("20250709")
    manager.add_shot_to_analysis(
        shot_file = "../../VISAR1/0409_1753_Shot65_Visar1.tif",
        ref_file = "../../VISAR1/0409_1746_Shot65_Visar1_ref.tif",
        sweep_speed = 20,
        slit_size = 500
        )
    