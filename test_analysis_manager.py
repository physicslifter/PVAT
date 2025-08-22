"""
Tests for AnalysisManager
"""
from AnalysisManager import *
import datetime
import shutil

sheindel_test = 0
test_BeamAM = 0
test_shotAM = 1

if test_BeamAM == True:
    #set up BeamAM
    manager = BeamAM("Analysis/AnalysisTests")
    #create new analysis & align
    manager.create_analysis(folder = "testBeamAM",
                            sweep_speed = 20,
                            slit_size = None,
                            img_path = "/Users/wicks3/Desktop/Work/JLF_2025/VISAR2/0409_1635_20ns_1ns_westbeam_Visar2_5.tif"
                           )
    #test if the proper files have been saved
    for file in ["correction.csv", "lineouts.csv", "info.csv"]:
        if file not in os.listdir(f"{manager.base_analysis_folder}/testBeamAM"):
            raise Exception(f"{file} not saved. Test failed!")
    print("Test passed!")

if test_shotAM == True:
    manager = ShotAM(V1_beam_folder = "testBeamAM",
                     V2_beam_folder = "testBeamAM",
                     base_analysis_folder = "Analysis/AnalysisTests",
                     data_folder = "../JLF_2025")
    manager.create_new_analysis(shot_ID = 73)
    manager.align_V2_ref()

if sheindel_test == True:
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
