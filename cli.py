"""
CLI for performing analyses
"""

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
        self.welcome_response = input()

#run the command line
CLI()