"""
Script for setting up tutorials and analysis workflow
after downloading the repository

Generating CSV files in Data and SyntheticData to be used for analysis

"""
import os
import pandas as pd
from numpy import nan
from datetime import datetime
import re

#Generate the Synthetic Data
import generate_synthetic_data

#create Analysis Folder
if not os.path.exists("Analysis"):
    os.mkdir("Analysis")

#%% JLF shot CSV reorganized

#Paths
data_dir = "data"
shot_log_path = os.path.join(data_dir, "JLF 2025 shot log.xlsx")
real_info_path = os.path.join(data_dir, "real_info.csv")

df = pd.read_excel(shot_log_path, dtype=str, header=[0,1])

# Editing some columns
date_col = ('Date', 'Unnamed: 1_level_1')
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[date_col] = df[date_col].apply(lambda dt: f"{dt.month}/{dt.day}/{dt.year}" if pd.notnull(dt) else "")

target_col = [col for col in df.columns if col[0] == 'Target'][0]
df = df[df[target_col].notnull() & (df[target_col].astype(str).str.strip() != "")]

main_cols = [
    ('Shot no.', 'Unnamed: 0_level_1'), ('Date', 'Unnamed: 1_level_1'),
    ('Beam', 'Unnamed: 2_level_1'), ('Energy (meter)', 'Unnamed: 3_level_1'),
    ('Conversion factor', 'Unnamed: 4_level_1'), ('Energy (J) (calculated)', 'Unnamed: 5_level_1'),
    ('Energy (J) (from logbook)', 'Unnamed: 6_level_1'), ('Pulse duration (ns)', 'Unnamed: 7_level_1'),
    ('Laser Power (W/cm^2)', 'Unnamed: 8_level_1'), ('Predicted Pressure from Swift & Kraus (GPa)', 'Unnamed: 9_level_1'),
    ('Predicted FSV (Swift & Kraus + Mitchell & Nellis)', 'Unnamed: 10_level_1'),
    ('Target', 'Unnamed: 11_level_1'), ('Sample no. (Julia)', 'Unnamed: 12_level_1')
]
visar1_cols = [
    ('VISAR 1', 'sweep time'), ('VISAR 1', 'gain'), ('VISAR 1', 'Etalon')
]
visar2_cols = [
    ('VISAR 2', 'sweep time'), ('VISAR 2', 'gain'), ('VISAR 2', 'Etalon')
]

columns_to_keep = main_cols + visar1_cols + visar2_cols
filtered_cols = [col for col in columns_to_keep if col in df.columns]
df_filtered = df[filtered_cols]

rows = []
for _, row in df_filtered.iterrows():
    # VISAR 1
    rows.append({
        **{k[0]: row[k] for k in main_cols},
        'VISAR': 1,
        'sweep_time': row.get(('VISAR 1', 'sweep time'), ''),
        'gain': row.get(('VISAR 1', 'gain'), ''),
        'etalon': row.get(('VISAR 1', 'Etalon'), ''),
    })
    # VISAR 2
    rows.append({
        **{k[0]: row[k] for k in main_cols},
        'VISAR': 2,
        'sweep_time': row.get(('VISAR 2', 'sweep time'), ''),
        'gain': row.get(('VISAR 2', 'gain'), ''),
        'etalon': row.get(('VISAR 2', 'Etalon'), ''),
    })

real_info_df = pd.DataFrame(rows)
real_info_df.to_csv(real_info_path, index=False)
print(f"Saved cleaned and reshaped shot log to {real_info_path}")

# Appending files to CSV

real_df = pd.read_csv(real_info_path, dtype=str)

for col in ['filename', 'Date', 'Shot no.', 'VISAR']:
    if col not in real_df.columns:
        real_df[col] = None

def extract_date_from_filename(fname):
    m = re.match(r"^(\d{4})", fname)
    if m:
        mmdd = m.group(1)
        if '0326' <= mmdd <= '0418':
            return mmdd

    matches = re.findall(r"(\d{4})", fname)
    for mmdd in matches:
        if '0326' <= mmdd <= '0418':
            return mmdd

    m = re.search(r"(20\d{6})", fname)
    if m:
        yyyymmdd = m.group(1)
        try:
            dt = datetime.strptime(yyyymmdd, "%Y%m%d")
            mmdd = dt.strftime("%m%d")
            if '0326' <= mmdd <= '0418':
                return mmdd
        except Exception:
            pass
    return None

def mmdd_to_datestr(mmdd, year='2025'):
    if mmdd and len(mmdd) == 4:
        dt = datetime.strptime(mmdd + year, "%m%d%Y")
        return f"{dt.month}/{dt.day}/{dt.year}"
    return None

def extract_shot(fname):
    m = re.search(r"[sS]hot\s*_*-*(\d+)", fname)
    return m.group(1) if m else None

def extract_visar(fname):
    m = re.search(r"[vV]isar\s*_*-*(\d+)", fname)
    return m.group(1) if m else None

def extract_visar_from_folder(folder):
    m = re.search(r"VISAR(\d+)", folder, re.IGNORECASE)
    return m.group(1) if m else None

def parse_filename(fname, folder):
    mmdd = extract_date_from_filename(fname)
    shot = extract_shot(fname)
    visar = extract_visar(fname)
    if not visar:
        visar = extract_visar_from_folder(folder)
    datesimple = mmdd_to_datestr(mmdd)
    return datesimple, shot, visar

folders = ['data/VISAR1', 'data/VISAR2']
all_files = []
for folder in folders:
    for fname in os.listdir(folder):
        if fname.lower().endswith('.tif'):
            all_files.append((folder, fname))

output_rows = []
used_indices = set()

real_df['Date_clean'] = real_df['Date'].astype(str).str.split().str[0]

for folder, fname in all_files:
    full_path = fname
    datesimple, shot, visar = parse_filename(fname, folder)
    matched = False
    if datesimple and shot and visar:
        mask = (
            (real_df['Date'] == datesimple) &
            (real_df['Shot no.'].astype(str) == shot) &
            (real_df['VISAR'].astype(str) == visar)
        )
        matches = real_df[mask]
        if not matches.empty:
            assigned = False
            for idx, row in matches.iterrows():
                if pd.isnull(row['filename']) or row['filename'] == '':
                    real_df.at[idx, 'filename'] = full_path
                    used_indices.add(idx)
                    assigned = True
                    matched = True
                    break
            if not assigned:
                new_row = matches.iloc[0].copy()
                new_row['filename'] = full_path
                output_rows.append(new_row)
                matched = True
    if not matched:
        new_row = {col: None for col in real_df.columns}
        new_row['filename'] = full_path
        new_row['Date'] = datesimple
        new_row['Shot no.'] = shot
        new_row['VISAR'] = visar
        output_rows.append(new_row)

if output_rows:
    real_df = pd.concat([real_df, pd.DataFrame(output_rows)], ignore_index=True)

for col in ['Date_clean', 'DateSimple', 'ShotExtracted', 'VisarExtracted']:
    if col in real_df.columns:
        real_df = real_df.drop(columns=[col])

def determine_type(fname):
    if pd.isnull(fname):
        return "Shot"
    if isinstance(fname, str):
        return "BeamRef" if "ref" in fname.lower() else "Shot"
    return "Shot"

real_df["Type"] = real_df["filename"].apply(determine_type)

real_df['Date_sort'] = pd.to_datetime(real_df['Date'], format='%m/%d/%Y', errors='coerce')
real_df['Shot_no_sort'] = pd.to_numeric(real_df['Shot no.'], errors='coerce')
real_df = real_df.sort_values(
    by=['Date_sort', 'Shot_no_sort'],
    ascending=[True, True],
    na_position='last'
)
real_df = real_df.drop(columns=['Date_sort', 'Shot_no_sort'])

real_df.to_csv(real_info_path, index=False)

#%% Synthetic Data CSV

#create info excel file for Synthetic analysis data
df = pd.DataFrame({"Name": [], #Name of the data (can be anything)
                   "Type": [], #type of the data (beam_ref, shot_ref, or shot)
                   "Fname": [], #file for the shot
                   "sweep_speed": [], #sweep speed for the shot
                   "slit_size": [], #slit size of the camera
                   "etalon": [], #the width of the etalon
                   "Ref": [] #name of the reference for the shot
                   })
# Add synthetic beam reference
df.loc[0] = [
    "SyntheticBeam",
    "beam_ref",
    "SyntheticData/20nsBeamReference.tif",
    20,
    500,
    20,
    nan
]

#synthetic shot reference
df.loc[1] = [
    "SyntheticShotRef",
    "shot_ref",
    "SyntheticData/20nsShotReference.tif",
    20,
    500,
    20,
    nan
]

#synthetic shot reference
df.loc[2] = [
    "SyntheticShot",
    "shot",
    "SyntheticData/20nsShot.tif",
    20,
    500,
    20,
    "SyntheticShotRef"
]

#save excel to Analysis folder
df.to_excel("Analysis/info.xlsx")