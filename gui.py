"""
GUI for performing analysis

At least, a draft. Still need to update folder formatting in AnalysisManager for beamref & shot, adjust options to include synthetic (& interactive plot for synthetic)

Likely also, if interactive plot not available, to generate one or offer to show other available data
"""

from AnalysisManager import AM2
from InteractivePlots import BeamAligner, ShotAligner
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import os

class AnalysisGUI:
    def __init__(self):
        self.manager = AM2("Analysis")
        self.analysis_name = None
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.widgets = []
        self.show_welcome_screen()
        plt.show()

    def clear_widgets(self):
        for widget in self.widgets:
            widget.ax.remove()
        self.widgets = []
        self.ax.clear()
        self.ax.axis('off')
        plt.draw()

    def show_welcome_screen(self):
        self.clear_widgets()
        btn_new = Button(plt.axes([0.2, 0.1, 0.25, 0.1]), 'Start New Analysis')
        btn_open = Button(plt.axes([0.55, 0.1, 0.25, 0.1]), 'Open Existing Analysis')
        btn_new.on_clicked(lambda event: self.prompt_analysis_name(new=True))
        btn_open.on_clicked(lambda event: self.list_existing_analyses())
        self.widgets.extend([btn_new, btn_open])
        
        banner = (
        "=================================\n"
        "=================================\n"
        "   ______     ___  _____ \n"
        "  |  _ \\ \\   / / \\|_   _|\n"
        "  | |_) \\ \\ / / _ \\ | |  \n"
        "  |  __/ \\ V / ___ \\| |  \n"
        "  |_|     \\_/_/   \\_\\_|  \n"
        "                         \n"
        "Python VISAR Analysis Tool\n"
        )
        self.ax.clear()
        self.ax.axis('off')
        self.ax.text(0.5, 1, banner, fontsize=14, fontfamily='monospace', 
                     ha='center', va='top')
        
        self.ax.text(0.5, 0.25, "What would you like to do?", fontsize=16,
                     fontweight='bold', ha='center', va='top')
        plt.draw()

    def prompt_analysis_name(self, new):
        self.clear_widgets()
        text_box = TextBox(plt.axes([0.2, 0.5, 0.6, 0.1]), 'Analysis Name')
        def submit(text):
            if new:
                try:
                    self.manager.create_new_analysis(text)
                    self.analysis_name = text
                    self.show_analysis_options()
                except Exception as e:
                    self.ax.set_title(f"Error: {e}")
                    plt.draw()
            else:
                try:
                    self.manager.open_analysis(text)
                    self.analysis_name = text
                    self.show_analysis_options()
                except Exception as e:
                    self.ax.set_title(f"Error: {e}")
                    plt.draw()
        text_box.on_submit(submit)
        self.widgets.append(text_box)
        self.ax.set_title("Enter Analysis Name")
        plt.draw()

    def list_existing_analyses(self):
        self.clear_widgets()
        analyses = [d for d in os.listdir(self.manager.base_folder) if os.path.isdir(os.path.join(self.manager.base_folder, d))]
        if not analyses:
            self.ax.set_title("No existing analyses found.")
            btn_back = Button(plt.axes([0.4, 0.05, 0.2, 0.1]), 'Back')
            btn_back.on_clicked(lambda event: self.show_welcome_screen())
            self.widgets.append(btn_back)
            plt.draw()
            return
        self.ax.set_title("Existing Analyses:\n" + ", ".join(analyses))
        text_box = TextBox(plt.axes([0.2, 0.5, 0.6, 0.1]), 'Analysis to Open')
        text_box.on_submit(lambda text: self.prompt_analysis_name(new=False))
        self.widgets.append(text_box)
        plt.draw()

    def show_analysis_options(self):
        self.clear_widgets()
        btn_beam = Button(plt.axes([0.15, 0.05, 0.2, 0.1]), 'Analyze BeamRef')
        btn_shot = Button(plt.axes([0.4, 0.05, 0.2, 0.1]), 'Analyze Shot')
        btn_saveas = Button(plt.axes([0.65, 0.05, 0.2, 0.1]), 'Save As New')
        btn_beam.on_clicked(lambda event: self.select_beam_ref())
        btn_shot.on_clicked(lambda event: self.select_shot())
        btn_saveas.on_clicked(lambda event: self.save_as_new())
        self.widgets.extend([btn_beam, btn_shot, btn_saveas])
        self.ax.set_title(f"Analysis: {self.analysis_name}\nChoose what to analyze or save as new.")
        plt.draw()

    def select_beam_ref(self):
        self.clear_widgets()
        beam_refs = self.manager.beam_refs
        if len(beam_refs) == 0:
            self.ax.set_title("No beam references found.")
            btn_back = Button(plt.axes([0.4, 0.05, 0.2, 0.1]), 'Back')
            btn_back.on_clicked(lambda event: self.show_analysis_options())
            self.widgets.append(btn_back)
            plt.draw()
            return
        self.ax.set_title("Beam References:\n" + ", ".join(beam_refs))
        text_box = TextBox(plt.axes([0.2, 0.5, 0.6, 0.1]), 'BeamRef Name')
        def submit(name):
            if name in beam_refs:
                self.launch_beam_aligner(name)
            else:
                self.ax.set_title("Invalid name. Try again.")
                plt.draw()
        text_box.on_submit(submit)
        self.widgets.append(text_box)
        plt.draw()

    def select_shot(self):
        self.clear_widgets()
        shots = self.manager.shots
        if len(shots) == 0:
            self.ax.set_title("No shots found.")
            btn_back = Button(plt.axes([0.4, 0.05, 0.2, 0.1]), 'Back')
            btn_back.on_clicked(lambda event: self.show_analysis_options())
            self.widgets.append(btn_back)
            plt.draw()
            return
        self.ax.set_title("Shots:\n" + ", ".join(shots))
        text_box = TextBox(plt.axes([0.2, 0.5, 0.6, 0.1]), 'Shot Name')
        def submit(name):
            if name in shots:
                self.launch_shot_aligner(name)
            else:
                self.ax.set_title("Invalid name. Try again.")
                plt.draw()
        text_box.on_submit(submit)
        self.widgets.append(text_box)
        plt.draw()

    def launch_beam_aligner(self, name):
        # Save progress before launching plot if needed
        plt.close(self.fig)  # Close the GUI window
        try:
            self.manager.analyze_beam_ref(name, self.analysis_name)
        except Exception as e:
            print(f"Error launching BeamAligner: {e}")
        # After closing the aligner, re-open GUI
        self.__init__()

    def launch_shot_aligner(self, name):
        plt.close(self.fig)
        try:
            fname = self.manager.get_filename(name)
            from VISAR import VISARImage
            img = VISARImage(fname)
            aligner = ShotAligner(img)
            aligner.initialize_plot()
            aligner.show_plot()
        except Exception as e:
            print(f"Error launching ShotAligner: {e}")
        self.__init__()

    def save_as_new(self):
        self.clear_widgets()
        text_box = TextBox(plt.axes([0.2, 0.5, 0.6, 0.1]), 'New Analysis Name')
        def submit(new_name):
            try:
                self.manager.create_new_analysis(new_name)
                self.analysis_name = new_name
                self.ax.set_title(f"Saved as new analysis: {new_name}")
                plt.draw()
                self.show_analysis_options()
            except Exception as e:
                self.ax.set_title(f"Error: {e}")
                plt.draw()
        text_box.on_submit(submit)
        self.widgets.append(text_box)
        plt.draw()

if __name__ == "__main__":
    AnalysisGUI()