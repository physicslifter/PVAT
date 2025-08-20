"""
CLI for performing analyses
"""
from AnalysisManager import AnalysisManager, AM2
import os
import sys

def get_response(msg, valid_options, failure_msg = None):
    answer = None
    while answer not in valid_options:
        print(msg)
        answer = input()
        if answer.upper() == "Q":
            break
        answer = int(answer)
        if answer not in valid_options and failure_msg != None:
            print(failure_msg)
    return answer

class CLI:
    def __init__(self):
        self.manager = AnalysisManager("Analysis")
        self.print_welcome_message()
        while(1):
            self.analysis_prompt()
            #run analysis prompt until user quite
    
    def print_welcome_message(self):
        #print welcome msg
        print()
        print()
        print()
        print("=================================")
        print("=================================")
        print("   ______     ___  _____ ")
        print("  |  _ \\ \\   / / \\|_   _|")
        print("  | |_) \\ \\ / / _ \\ | |  ")
        print("  |  __/ \\ V / ___ \\| |  ")
        print("  |_|     \\_/_/   \\_\\_|  ")
        print("                         ")
        print("Python VISAR Analysis Tool")
        print()
        print()
        print("What would you like to do?")
        print("     (1) Start new analysis")
        print("     (2) Open saved analysis")
        self.welcome_response = int(input())
        self.handle_welcome_response()

    def handle_welcome_response(self):
        if self.welcome_response == 1:
            #start new analysis
            self.manager = AM2("Analysis")
            analysis_name = "1"
            while analysis_name == "1":
                analysis_name = input("Enter Name for Analysis: (Press 1 to view all analysis names)\n")
                if analysis_name == "1":
                    print([i for i in os.listdir(self.manager.base_folder) if "." not in i])
                if analysis_name in os.listdir(self.manager.base_folder):
                    print("Analysis already exists, enter another name\n")
                    analysis_name = "1"
            self.manager.create_new_analysis(analysis_name)
            print(f"\n\n\n==\nANALYSIS: {analysis_name}")

        elif self.welcome_response == 2:
            #open a saved analysis
            raise Exception("Response 2 not handled")
        
        else:
            raise Exception("Must be 1 or 2")
        
    def analysis_prompt(self):
        """
        Prompt to run once analysis has been opened
        """
        msg = "What would you like to do?\n(1) Analyze a Beam Reference\n(2) Analyze a Shot\n(Q) Quit"
        failure_msg = "Must be (1), or (2)"
        self.analysis_response = get_response(msg, [1, 2], failure_msg)
        if self.analysis_response == "Q":
            sys.exit(0)
        self.handle_analysis_response()

    def handle_analysis_response(self):
        msg = "What type of analysis?\n (1) Beam reference\n (2) Shot reference\n (3) Shot\n (Q) Quit"
        failure_msg = "Must be 1, 2 or 3"
        if self.analysis_response == 1: #If we want to analyze a beam reference
            msg = "Select Ref file:\n"
            valid_beam_refs = []
            for i, ref_name in enumerate(self.manager.beam_refs):
                msg += f" ({i+1}) {ref_name}"
                valid_beam_refs.append(i + 1)
            ref_num = get_response(msg, valid_beam_refs, "Invalid response")
            ref_name = self.manager.beam_refs[ref_num - 1]
            self.manager.analyze_beam_ref(name = ref_name)
        elif self.analysis_response == 2: #If we want to analyze a shot
            #get reference for the shot
            #get beam reference
            #analyze shot reference
            #analyze shot
            pass
    


#run the command line
CLI()