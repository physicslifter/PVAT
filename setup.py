"""
Script for setting up tutorials and analysis workflow
after downloading the repository

Generating CSV files in Data and SyntheticData to be used for analysis

"""
import os
import pandas as pd
from datetime import datetime
import re
import numpy as np

#create Analysis Folder
if not os.path.exists("Analysis"):
    os.mkdir("Analysis")

# JLF shot CSV reorganized

#Paths
#if path is in your real data folder
shot_log_path = "JLF 2025 shot log.xlsx" 
real_info_path = "real_info.csv"

#if path is in general folder with subfolders real data, synthetic data, analysis (see folders variable to comment out too)
#data_dir = "data"
#shot_log_path = os.path.join(data_dir, "JLF 2025 shot log.xlsx")
#real_info_path = os.path.join(data_dir, "real_info.csv")

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

# Moving 0326 Shot 5 to 03/27 date for Shot 5
filename_date_corrections = {
    '0326_1442_Shot5_Visar1.tif': '0327',
    '0326_1442_Shot5_Visar2.tif': '0327',
}

def extract_date_from_filename(fname):
    base = os.path.basename(fname)
    if base in filename_date_corrections:
        return filename_date_corrections[base]
    
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

folders = ["VISAR1", "VISAR2]
#folders = [os.path.join(data_dir, "VISAR1"), os.path.join(data_dir, "VISAR2")]

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
    if not isinstance(fname, str):
        return "Other" #is westbeam "Other" or "BeamRef"?
    lower = fname.lower()
    if "timingref" in lower or "timing_ref" in lower:
        return "BeamRef"
    if "shot" in lower and "ref" in lower:
        return "ShotRef"
    if "shot" in lower and "ref" not in lower:
        return "Shot"
    if "pin" in lower:
        return "Other"
    return "Other"

real_df["Type"] = real_df["filename"].apply(determine_type)

real_df['Date_sort'] = pd.to_datetime(real_df['Date'], format='%m/%d/%Y', errors='coerce')
real_df['Shot_no_sort'] = pd.to_numeric(real_df['Shot no.'], errors='coerce')
real_df = real_df.sort_values(
    by=['Date_sort', 'Shot_no_sort'],
    ascending=[True, True],
    na_position='last'
)
real_df = real_df.drop(columns=['Date_sort', 'Shot_no_sort'])

def is_valid(val):
    return val not in [None, '', 'nan', 'NaN'] and not (isinstance(val, float) and np.isnan(val))

def construct_name(row):
    shot = row.get('Shot no.', '')
    visar = row.get('VISAR', '')
    fname = row.get('filename', '')
    if is_valid(shot) and is_valid(visar):
        return f"Shot{int(shot)}_Visar{int(visar)}"
    elif isinstance(fname, str) and fname.lower().endswith('.tif'):
        return os.path.splitext(os.path.basename(fname))[0]
    else:
        return ""

real_df['Name'] = real_df.apply(construct_name, axis=1)

real_df.to_csv(real_info_path, index=False)


# Analysis CSV (to be generated/appended in GUI)

def append_analysis_csv(
    csv_path, 
    name, 
    type_, 
    fname, 
    sweep_speed, 
    slit_size, 
    etalon, 
    ref, 
    analysis_folder, 
    notes=""
):

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            "Name", "Type", "Fname", "sweep_speed", "slit_size", "etalon", 
            "Ref", "AnalysisFolder", "DateAnalyzed", "Notes"
        ])

    row = {
        "Name": name,
        "Type": type_,
        "Fname": fname,
        "sweep_speed": sweep_speed,
        "slit_size": slit_size,
        "etalon": etalon,
        "AnalysisFolder": analysis_folder,
        "DateAnalyzed": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Notes": notes
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Appended analysis for {name} to {csv_path}")
