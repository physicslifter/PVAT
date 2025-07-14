"""
CLI for performing analyses
"""
from AnalysisManager import AnalysisManager

class CLI:
    def __init__(self):
        self.print_welcome_message()
    
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

    def handle_welcome_response(self):
        if self.welcome_response == 1:
            #start new analysis
            base_directory = input("Enter fpath for base directory: \n")
            self.manager = AnalysisManager(base_directory)
        elif self.welcome_response == 2:
            #open a saved analysis
            pass


#run the command line
CLI()