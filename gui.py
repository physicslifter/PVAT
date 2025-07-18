"""
GUI for performing analysis

Double check right analyses done for shots/shotrefs/beamrefs/others

Need to decide on naming/organization of files, see def _init_...

Likely, if interactive plot not available, to generate one or offer to show other available data
"""

from VISAR import VISARImage, RefImage
from AnalysisManager import AnalysisManager
from InteractivePlots import BeamAligner, ShotAligner, ShotRefAligner
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
import os
import pandas as pd
import numpy as np

def normalize_path(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))

def find_real_data_files(csv_path, shot_input, visar_input, type_input):
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [col.strip() for col in df.columns]
    for col in ['Shot no.', 'VISAR', 'Type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    type_input = type_input.lower()
    if shot_input.lower() == "other" or shot_input == "":
        mask = (
            df['Shot no.'].isnull() |
            (df['Shot no.'].astype(str).str.strip() == "") |
            (df['Shot no.'].astype(str).str.lower() == "nan")
        )
        filtered = df[mask & (df['Type'].str.lower() == type_input)]
        if visar_input:
            filtered = filtered[filtered['VISAR'] == visar_input]
    else:
        filtered = df[
            (df['Shot no.'] == shot_input) &
            (df['VISAR'] == visar_input) &
            (df['Type'].str.lower() == type_input)
        ]
    return filtered

class AnalysisGUI:
    def __init__(self):
        self.base_analysis_dir = "Analysis"
        self.data_sources = {"Real Data": "Data", "Synthetic Data": "SyntheticData"}
        self.data_source = None
        self.analysis_name = None
        self.analysis_folder = None
        self.real_data_csv = "data/real_info.csv" #figure out naming convention to match...
        self.synthetic_data_csv = "SyntheticData/synthetic_info.csv"
        self.analysis_info_xlsx = os.path.join(self.base_analysis_dir, "info.xlsx")
        self.analysis_manager = AnalysisManager(self.base_analysis_dir)
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.widgets = []
        self.show_main_prompt()
        plt.show()
        
    def clear_widgets(self):
        for widget in self.widgets:
            if hasattr(widget, "disconnect_events"):
                widget.disconnect_events()
            widget.ax.remove()
        self.widgets = []
        self.ax.clear()
        self.ax.axis('off')
        plt.draw()
        self.file_buttons = []

    def show_main_prompt(self):
        self.clear_widgets()
        btn_new = Button(plt.axes([0.2, 0.1, 0.25, 0.1]), 'Start New Analysis')
        btn_open = Button(plt.axes([0.55, 0.1, 0.25, 0.1]), 'Open Saved Analysis')
        btn_new.on_clicked(lambda event: self.prompt_data_source(new=True))
        btn_open.on_clicked(lambda event: self.prompt_open_saved())
        self.widgets.extend([btn_new, btn_open])
        self.ax.clear()
        self.ax.axis('off')
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
        self.ax.text(0.5, 1, banner, fontsize=14, fontfamily='monospace',
                     ha='center', va='top')
        self.ax.text(0.5, 0.25, "What would you like to do?", fontsize=16,
                     fontweight='bold', ha='center', va='top')
        plt.draw()

    def prompt_data_source(self, new):
        self.clear_widgets()
        btn_real = Button(plt.axes([0.2, 0.5, 0.25, 0.1]), 'Real Data')
        btn_synth = Button(plt.axes([0.55, 0.5, 0.25, 0.1]), 'Synthetic Data')
        btn_real.on_clicked(lambda event: self.prompt_analysis_name(new, "Real Data"))
        btn_synth.on_clicked(lambda event: self.prompt_analysis_name(new, "Synthetic Data"))
        self.widgets.extend([btn_real, btn_synth])
        self.ax.set_title("Select Data Source")
        plt.draw()
    
    def prompt_open_saved(self):
        self.clear_widgets()
    
        folders = [f for f in os.listdir(self.base_analysis_dir)
                   if os.path.isdir(os.path.join(self.base_analysis_dir, f))]
        folders.sort()
        self.folder_scroll_window = 7
        self.folder_scroll_index = getattr(self, 'folder_scroll_index', 0)
        self.selected_folder_index = getattr(self, 'selected_folder_index', None)
    
        if not hasattr(self, 'folder_button_axes'):
            self.folder_button_axes = []
            self.folder_buttons = []
            y0 = 0.6
            button_height = 0.07
            for i in range(self.folder_scroll_window):
                ax_btn = plt.axes([0.085, y0 - i*button_height, 0.3, button_height*0.9])
                btn = Button(ax_btn, '', color='0.85')
                self.folder_button_axes.append(ax_btn)
                self.folder_buttons.append(btn)
                self.widgets.append(btn)
    
        def update_folder_buttons():
            start = self.folder_scroll_index
            end = min(start + self.folder_scroll_window, len(folders))
            for i in range(self.folder_scroll_window):
                btn = self.folder_buttons[i]
                if start + i < len(folders):
                    folder = folders[start + i]
                    idx = start + i
                    btn.label.set_text(folder)
                    btn.color = 'lightblue' if idx == self.selected_folder_index else '0.85'
                    btn.ax.set_visible(True)
                    # Remove previous callback
                    btn.on_clicked(lambda event: None)
                    # Set new callback
                    def make_on_click(idx=idx):
                        def on_click(event):
                            self.selected_folder_index = idx
                            update_folder_buttons()
                        return on_click
                    btn.on_clicked(make_on_click(idx))
                else:
                    btn.label.set_text("")
                    btn.ax.set_visible(False)
            plt.draw()
    
        def scroll_up(event):
            if self.folder_scroll_index > 0:
                self.folder_scroll_index -= 1
                update_folder_buttons()
        def scroll_down(event):
            if self.folder_scroll_index + self.folder_scroll_window < len(folders):
                self.folder_scroll_index += 1
                update_folder_buttons()
    
        ax_up = plt.axes([0.085, 0.7, 0.3, 0.04])
        btn_up = Button(ax_up, 'Up')
        btn_up.on_clicked(scroll_up)
        self.widgets.append(btn_up)
    
        ax_down = plt.axes([0.085, 0.1, 0.3, 0.04])
        btn_down = Button(ax_down, 'Down')
        btn_down.on_clicked(scroll_down)
        self.widgets.append(btn_down)
    
        self.ax.set_title("Search Your Analysis", fontsize=15, fontweight='bold')
        self.ax.text(0.15, 0.9, "Select Analysis Folder", fontsize=12, ha='center', va='center')

    
        update_folder_buttons()
    
        ax_ds = plt.axes([0.55, 0.75, 0.35, 0.1])
        ds_radio = RadioButtons(ax_ds, ['Real Data', 'Synthetic Data'])
        shot_box = TextBox(plt.axes([0.8, 0.45, 0.1, 0.08]), 'Shot (#, Other, or None)')
        visar_box = TextBox(plt.axes([0.8, 0.35, 0.1, 0.08]), 'Visar (1 or 2)')
        ax_type = plt.axes([0.55, 0.55, 0.35, 0.15])
        type_radio = RadioButtons(ax_type, ['Shot', 'ShotRef', 'BeamRef', 'Other'])
        btn_search = Button(plt.axes([0.55, 0.2, 0.35, 0.08]), 'Search')
    
        self.widgets.extend([ds_radio, shot_box, visar_box, type_radio, btn_search])
    
        self.search_input = {'data_source': 'Real Data', 'shot': '', 'visar': '', 'type': 'BeamRef'}
        ds_radio.on_clicked(lambda label: self.search_input.update({'data_source': label}))
        shot_box.on_submit(lambda text: self.search_input.update({'shot': text.strip()}))
        visar_box.on_submit(lambda text: self.search_input.update({'visar': text.strip()}))
        type_radio.on_clicked(lambda label: self.search_input.update({'type': label}))
    
        btn_search.on_clicked(lambda event: self.list_search_results())
    
        plt.draw()
    
    def list_folder_files(self, folder):
        self.clear_widgets()
        folder_path = os.path.join(self.base_analysis_dir, folder)
        types = ['Shot', 'ShotRef', 'BeamRef']
        y0 = 0.85
        button_height = 0.07
        for t in types:
            type_path = os.path.join(folder_path, t)
            if os.path.exists(type_path):
                for sub in os.listdir(type_path):
                    ax_btn = plt.axes([0.07, y0, 0.3, button_height*0.9])
                    btn = Button(ax_btn, f"{t}/{sub}")
                    btn.on_clicked(lambda event, p=os.path.join(type_path, sub): self.launch_analysis_from_folder(p))
                    self.widgets.append(btn)
                    y0 -= button_height
        plt.draw()
    
    def list_search_results(self):
        if os.path.exists(self.analysis_info_xlsx):
            df = pd.read_excel(self.analysis_info_xlsx, dtype=str)
        else:
            self.ax.set_title("No info.xlsx found in Analysis folder.")
            plt.draw()
            return
        df = pd.read_excel(self.analysis_info_xlsx, dtype=str)
        mask = pd.Series([True] * len(df))
        if hasattr(self, 'selected_folder_index') and self.selected_folder_index is not None:
            folders = [f for f in os.listdir(self.base_analysis_dir)
                       if os.path.isdir(os.path.join(self.base_analysis_dir, f))]
            folders.sort()
            selected_folder = folders[self.selected_folder_index]
            if 'AnalysisFolder' in df.columns:
                mask &= (df['AnalysisFolder'] == selected_folder)
            elif 'Folder' in df.columns:
                mask &= (df['Folder'] == selected_folder)
        if self.search_input['data_source']:
            mask &= (df['DataSource'].str.lower() == self.search_input['data_source'].lower())
        if self.search_input['shot']:
            mask &= (df['ShotNo'].astype(str) == self.search_input['shot'])
        if self.search_input['visar']:
            mask &= (df['Visar'].astype(str) == self.search_input['visar'])
        if self.search_input['type']:
            mask &= (df['Type'].str.lower() == self.search_input['type'].lower())
        filtered = df[mask]
        y0 = 0.5
        button_height = 0.07
        for i, row in filtered.iterrows():
            label = row.get('Name', row.get('Fname', 'Unknown'))
            ax_btn = plt.axes([0.5, y0, 0.35, button_height*0.9])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda event, r=row: self.launch_analysis_from_search(r))
            self.widgets.append(btn)
            y0 -= button_height
        plt.draw()

    def prompt_analysis_name(self, new, data_source):
        self.clear_widgets()
        self.data_source = data_source
        text_box = TextBox(plt.axes([0.2, 0.5, 0.6, 0.1]), 'Analysis Name')
        def submit(text):
            analysis_dir = os.path.join(self.base_analysis_dir, text)
            if new:
                if os.path.exists(analysis_dir):
                    self.ax.set_title(f"Error: Analysis '{text}' already exists.")
                    plt.draw()
                else:
                    os.makedirs(analysis_dir)
                    for sub in ["Shot", "BeamRef", "ShotRef"]:
                        os.makedirs(os.path.join(analysis_dir, sub))
                    self.analysis_name = text
                    self.analysis_folder = analysis_dir
                    self.show_analysis_prompts()
            else:
                if not os.path.exists(analysis_dir):
                    self.ax.set_title(f"Error: Analysis '{text}' not found.")
                    plt.draw()
                else:
                    self.analysis_name = text
                    self.analysis_folder = analysis_dir
                    self.show_analysis_prompts()
        text_box.on_submit(submit)
        self.widgets.append(text_box)
        self.ax.set_title("Enter Analysis Name")
        plt.draw()

    def show_analysis_prompts(self):
        self.clear_widgets()
        if self.data_source == "Synthetic Data":
            btn_beam = Button(plt.axes([0.15, 0.05, 0.2, 0.1]), 'BeamRef')
            btn_shot = Button(plt.axes([0.4, 0.05, 0.2, 0.1]), 'Shot')
            btn_shotref = Button(plt.axes([0.65, 0.05, 0.2, 0.1]), 'ShotRef')
            btn_beam.on_clicked(lambda event: self.launch_synthetic_analysis("BeamRef"))
            btn_shot.on_clicked(lambda event: self.launch_synthetic_analysis("Shot"))
            btn_shotref.on_clicked(lambda event: self.launch_synthetic_analysis("ShotRef"))
            self.widgets.extend([btn_beam, btn_shot, btn_shotref])
            self.ax.set_title(f"Analysis: {self.analysis_name}\nSelect Synthetic Data Type")
            plt.draw()
        else:
            self.prompt_real_data_inputs()

    def prompt_real_data_inputs(self):
        self.clear_widgets()
        shot_box = TextBox(plt.axes([0.4, 0.7, 0.25, 0.08]), 'Shot (Number or "Other")')
        visar_box = TextBox(plt.axes([0.4, 0.55, 0.25, 0.08]), 'VISAR (1 or 2)')
        type_ax = plt.axes([0.4, 0.35, 0.25, 0.15])
        type_radio = RadioButtons(type_ax, ['Shot', 'ShotRef', 'BeamRef', 'Other'])
        btn_submit = Button(plt.axes([0.4, 0.2, 0.25, 0.08]), 'Search')

        self.real_input = {'shot': '', 'visar': '', 'type': 'Shot'}

        shot_box.on_submit(lambda text: self.real_input.update({'shot': text.strip()}))
        visar_box.on_submit(lambda text: self.real_input.update({'visar': text.strip()}))
        type_radio.on_clicked(lambda label: self.real_input.update({'type': label}))
        btn_submit.on_clicked(lambda event: self.search_real_data_files())

        self.widgets.extend([shot_box, visar_box, type_radio, btn_submit])
        self.ax.set_title("Enter Real Data Inputs")
        plt.draw()
    
    def search_real_data_files(self):
        shot = self.real_input.get('shot', '').strip()
        visar = self.real_input.get('visar', '').strip()
        dtype = self.real_input.get('type', '').strip()
        try:
            filtered = find_real_data_files(self.real_data_csv, shot, visar, dtype)
        except Exception as e:
            self.ax.set_title(f"Error loading real_info.csv: {e}")
            plt.draw()
            return
    
        files = filtered['filepath'].dropna().tolist() if 'filepath' in filtered.columns else []
        types = filtered['Type'].tolist() if 'Type' in filtered.columns else []
        display_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        
        self.clear_widgets()
        self.file_buttons = []
        if not files:
            self.ax.text(0.5, 0.7, "No files found for this selection.", fontsize=12, ha='center', va='center')
        else:
            self.ax.text(0.5, 0.95, "Select a file to analyze:", fontsize=12, ha='center', va='center')
            n_files = len(files)
            n_cols = 2
            n_rows = int(np.ceil(n_files / n_cols))
            button_height = 0.4 / n_rows
            button_width = 0.4
    
            self.selected_file = None
            self.selected_type = None
            self.selected_index = None
            self.file_buttons = []
    
            def make_on_click(f, t, ax_btn, idx):
                def on_click(event):
                    self.selected_file = f
                    self.selected_type = t
                    self.selected_index = idx
                    for j, (btn, ax, f_name, f_type) in enumerate(self.file_buttons):
                        ax.clear()
                        color = 'lightblue' if j == idx else '0.85'
                        new_btn = Button(ax, f_name, color=color)
                        new_btn.on_clicked(make_on_click(f_name, f_type, ax, j))
                        self.file_buttons[j] = (new_btn, ax, f_name, f_type)
                    self.fig.canvas.draw_idle()
                return on_click
            
            for i, (f, t) in enumerate(zip(files, types)):
                col = i % n_cols
                row = i // n_cols
                left = 0.05 + col * (button_width + 0.05)
                bottom = 0.7 - (row + 1) * button_height
                ax_btn = plt.axes([left, bottom, button_width, button_height*0.9])
                btn = Button(ax_btn, f, color='0.85')
                btn.on_clicked(make_on_click(f, t, ax_btn, i))
                self.widgets.append(btn)
                self.file_buttons.append((btn, ax_btn, f, t))
    
            btn_launch = Button(plt.axes([0.35, 0.05, 0.3, 0.05]), 'Launch Analysis')
            def launch(event):
                if self.selected_file:
                    self.launch_analysis(self.selected_file, self.selected_type)
                else:
                    self.ax.set_title("Please select a file first.")
                    plt.draw()
            btn_launch.on_clicked(launch)
            self.widgets.append(btn_launch)
    
        btn_back = Button(plt.axes([0.7, 0.1, 0.2, 0.08]), 'Back')
        btn_back.on_clicked(lambda event: self.prompt_real_data_inputs())
        self.widgets.append(btn_back)
        plt.draw()

    def launch_analysis(self, filename, filetype):
        self.clear_widgets()
        if self.data_source == "Real Data":
            csv_path = self.real_data_csv
        else:
            csv_path = self.synthetic_data_csv
    
        if not self.analysis_name:
            self.ax.set_title("No analysis name selected.")
            plt.draw()
            return
        self.analysis_manager.create_or_open_analysis(self.analysis_name)
    
        try:
            info_row = self.analysis_manager.extract_info_from_csv(filename, csv_path)
        except Exception as e:
            self.ax.set_title(f"Could not extract info: {e}")
            plt.draw()
            return
    
        try:
            instance_folder = self.analysis_manager.save_analysis_instance(
                data_type=info_row["Type"] if info_row["Type"] else filetype,
                base_name=info_row["Name"] if info_row["Name"] else os.path.splitext(os.path.basename(filename))[0],
                info_row=info_row,
                notes=""
            )
        except Exception as e:
            self.ax.set_title(f"Failed to save analysis instance: {e}")
            plt.draw()
            return
    
        self.ax.text(0.5, 0.5, f"Analysis at:\n{instance_folder}", fontsize=12, ha='center', va='center')
        plt.draw()
        
        self.launch_interactive_plot(instance_folder, info_row["Type"] if info_row["Type"] else filetype)

        
    def launch_interactive_plot(self, instance_folder, analysis_type):
        """
        Launch the correct interactive plot for the analysis_type.
        """
        
        info_path = os.path.join(instance_folder, "info.xlsx")
        if not os.path.exists(info_path):
            self.ax.set_title("info.xlsx not found in analysis instance.")
            plt.draw()
            return
    
        info = pd.read_excel(info_path).iloc[0]
    
        fname = info.get("Filepath", "")
        sweep_speed = info.get("sweep_speed", "")
        slit_size = info.get("slit_size", "")
        
        print(f"Trying to open file: {fname} (type: {type(fname)})")
        if not fname or not isinstance(fname, str) or not os.path.exists(fname):
            raise FileNotFoundError(f"Filepath '{fname}' does not exist or is invalid.")
    
        try:
            sweep_speed = float(sweep_speed)
        except Exception:
            sweep_speed = None
        try:
            slit_size = float(slit_size)
        except Exception:
            slit_size = None
    
        if analysis_type.lower() == "beamref":
            img = RefImage(fname=fname, folder=instance_folder, sweep_speed=sweep_speed, slit_size=slit_size)
            aligner = BeamAligner(img)
            aligner.set_lineout_save_name(os.path.join(instance_folder, "lineouts.csv"))
            aligner.set_correction_save_name(os.path.join(instance_folder, "correction.csv"))
            aligner.show_plot()
        elif analysis_type.lower() == "shot":
            img = VISARImage(fname=fname, sweep_speed=sweep_speed, slit_size=slit_size)
            aligner = ShotAligner(img)
            aligner.set_folder(instance_folder)
            aligner.show_plot()
        elif analysis_type.lower() == "shotref":
            img = VISARImage(fname=fname, sweep_speed=sweep_speed, slit_size=slit_size)
            aligner = ShotRefAligner(img)
            aligner.set_folder(instance_folder)
            aligner.show_plot()
        else:
            self.ax.set_title(f"Unknown analysis type: {analysis_type}")
            plt.draw()
    
    def launch_synthetic_analysis(self, analysis_type):
        try:
            df = pd.read_csv(self.synthetic_data_csv)
        except Exception as e:
            self.ax.set_title(f"Error loading synthetic_info.csv: {e}")
            plt.draw()
            return
    
        filtered = df[df['type'] == analysis_type]
        files = filtered['tif_file'].tolist() if 'tif_file' in filtered.columns else []
        types = filtered['type'].tolist() if 'type' in filtered.columns else []
    
        self.clear_widgets()
        self.file_buttons = []
        self.selected_file = None
        self.selected_type = None
    
        if not files:
            self.ax.set_title("No files found for this synthetic type.")
            plt.draw()
            return
    
        self.ax.text(0.5, 0.95, "Select a file to analyze:", fontsize=12, ha='center', va='center')
        n_files = len(files)
        n_cols = 2
        n_rows = int(np.ceil(n_files / n_cols))
        button_height = 0.4 / n_rows
        button_width = 0.4
    
        def make_on_click(f, t, ax_btn, idx):
            def on_click(event):
                self.selected_file = f
                self.selected_type = t
                for j, (btn, ax, f_name, f_type) in enumerate(self.file_buttons):
                    ax.clear()
                    color = 'lightblue' if j == idx else '0.85'
                    new_btn = Button(ax, f_name, color=color)
                    new_btn.on_clicked(make_on_click(f_name, f_type, ax, j))
                    self.file_buttons[j] = (new_btn, ax, f_name, f_type)
                self.fig.canvas.draw_idle()
            return on_click
    
        for i, (f, t) in enumerate(zip(files, types)):
            col = i % n_cols
            row = i // n_cols
            left = 0.05 + col * (button_width + 0.05)
            bottom = 0.7 - (row + 1) * button_height
            ax_btn = plt.axes([left, bottom, button_width, button_height*0.9])
            btn = Button(ax_btn, f, color='0.85')
            btn.on_clicked(make_on_click(f, t, ax_btn, i))
            self.widgets.append(btn)
            self.file_buttons.append((btn, ax_btn, f, t))
    
        btn_launch = Button(plt.axes([0.35, 0.05, 0.3, 0.05]), 'Launch Analysis')
        def launch(event):
            if self.selected_file:
                self.launch_analysis(self.selected_file, self.selected_type)
            else:
                self.ax.set_title("Please select a file first.")
                plt.draw()
        btn_launch.on_clicked(launch)
        self.widgets.append(btn_launch)
    
        btn_back = Button(plt.axes([0.7, 0.1, 0.2, 0.08]), 'Back')
        btn_back.on_clicked(lambda event: self.show_analysis_prompts())
        self.widgets.append(btn_back)
    
        plt.draw()

if __name__ == "__main__":
    AnalysisGUI()

