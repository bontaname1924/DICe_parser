import vmbpy
import cv2
import time
import os
from datetime import datetime
import threading
import queue
import csv

# =============================================================================
# ---  INFORMATION  ---
#   FLOW:                       IMAGE ACQUISITION -> VERTICAL STABILISATION -> ROI SELECTION -> DICe EXECUTION -> PARAVIEW -> MERGE
#   SAVED-IMAGE SCHEDULE:      FOUR PHASES: STARTING CONSTANT, EXPANSION 1, EXPANSION 2, LATE STABLE
#   CAMERA:                     ALLIED VISION MAKO U-130B & COMPATIBLE
#   THERMAL CONDITIONING:       CONTINUOUS FRAME ACQUISITION
#   PREHEAT:                    USER-DETERMINED PREHEAT
#   DICe:                       MPI ENABLED VIA DOCKER
#   ROI:                        ONE, NO EXCLUSION
# =============================================================================

# --- 1. Set Dynamic Directory ---
user_home = os.path.expanduser("~")
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_directory = os.path.join(user_home, "AZ31DIC", "DICe", start_time)
results_directory = os.path.join(save_directory, "results")
os.makedirs(results_directory, exist_ok=True)


# =============================================================================
# ---  SHUTTER CONTROL PANEL  ---
# =============================================================================

start_interval = 0.1
preheat_duration_seconds = 10
temperature_poll_interval_seconds = 5.0
constant_duration_seconds = 60.0
phase1_duration_seconds = 2400.0
phase1_expansion = 1.001
phase2_expansion = 1.05
stable_interval = 1200.0

# =============================================================================
# --- 2. AUTO-GENERATE THE ENTIRE ANALYSIS SUITE ---
# =============================================================================

# A. Generate the initial _specs.txt file
memo_filename = os.path.join(save_directory, "_specs.txt")
memo_content = f"""-------------------- EXPERIMENT METADATA --------------------
Load (N):
Conditions:
Notes:

-------------------- DIC MEASUREMENT SYSTEM --------------------
DIC Configuration:                  2D-DIC, single camera
Camera Manufacturer:               Allied Vision
Camera Model:                       Mako U-130B
Lens Manufacturer:                 Edmund Optics
Lens Model:                        #63-741
Lens Type:                         Object-space telecentric
Conventional Focal Length:         Not applicable as operational scaling parameter
Primary Magnification:             0.5X
Aperture Used (f-number):           9
Stand-Off Distance (mm):           65
Field of View (mm x mm):           12.288 x 9.830
Image Scale (px/mm):               104.17

-------------------- IMAGE ACQUISITION --------------------
Frame-Acquisition Target Interval (s): {start_interval}
Frame-Acquisition Target Rate (Hz):    {1.0 / start_interval:g}
Preheat Duration (s):               {preheat_duration_seconds}
Temperature Polling Interval (s):   {temperature_poll_interval_seconds}
Saved-Image Sampling:              Variable interval
Saved-Image Schedule Basis:        Four-phase interval schedule
Schedule Phase 1:                  Starting constant interval
Schedule Phase 2:                  Expansion 1
Schedule Phase 3:                  Expansion 2
Schedule Phase 4:                  Late stable interval
Starting Constant Duration (s):    {constant_duration_seconds}
Expansion 1 Duration (s):          {phase1_duration_seconds}
Expansion 1 Factor:                {phase1_expansion}
Expansion 2 Factor:                {phase2_expansion}
Late Stable Interval (s):          {stable_interval}
Saved-Image Sampling Rate Range:   {1.0/start_interval:g} to {1.0/stable_interval:.6f} Hz
Per-Image Time Record:             results/master_log.csv, True_Elapsed_Time_s
Elapsed-Time Clock:                Host monotonic clock
Timestamp Definition:              Host elapsed time immediately after frame retrieval

-------------------- CAMERA AND IMAGE FORMAT --------------------
Sensor:                            On Semiconductor PYTHON 1300
Sensor ADC Nominal Bit Depth:      10-bit
Configured Acquisition Format:    Mono8
Stored Image Format:              8-bit BMP
Nominal Image Size (px):           1280 x 1024
Pixel Pitch (um x um):             4.8 x 4.8
Exposure Time:                     Appended from camera read-back after connection

-------------------- DIC PATTERN --------------------
Patterning Technique:              Aerosol-painted stochastic pattern by spray paint can
Feature-Size Measurement Method:   Electronic microscope camera
Approx. Pattern Feature Size (um): 25
Feature-Size Statistic/Definition: [ENTER measurement definition]
Microscope Model/Calibration:      [ENTER instrument and calibration]
"""
with open(memo_filename, 'w', encoding="utf-8", newline="\n") as f:
    f.write(memo_content)

# B. Generate run_dice.py with automated documentation
run_dice_filename = os.path.join(save_directory, "run_dice.py")
run_dice_code = r'''import os
import subprocess
import sys

# =============================================================================
# ---  USER INPUT PANEL  ---
# =============================================================================

# =============================================================================
# --- AUTO-MANAGED IMAGE RANGE AND ROI SETTINGS ---
# Updated by select_roi.py. Rerunning it replaces only this marked block.
# =============================================================================
# BEGIN SELECT_ROI MANAGED SETTINGS

START_IMAGE = "[CHANGE]"
END_IMAGE = "[CHANGE]"

# FORMAT: X Y, ordered TL, TR, BR, BL.
ROI_VERTICES = """\
[CHANGE]
[CHANGE]
[CHANGE]
[CHANGE]"""

# END SELECT_ROI MANAGED SETTINGS

SUBSET_SIZE = "51"
STEP_SIZE = "12"

# Define the strain window as a grid of subsets (e.g., 5 for a 5x5 grid).
# This must be an odd integer, 3 or greater. 5 is a robust default.
STRAIN_WINDOW_IN_SUBSETS = 5

# --- Correlation controls ---
#
# INITIALISATION METHODS:
# USE_FEATURE_MATCHING:
#     Uses detected image features to estimate the initial displacement.
# USE_FIELD_VALUES:
#     Uses displacement-field values already stored for each correlation point.
# USE_NEIGHBOR_VALUES:
#     Uses solved neighbouring subsets to initialise subsequent subsets.
# USE_NEIGHBOR_VALUE_FIRST_STEP_ONLY:
#     Uses neighbouring values for the first step, then field values thereafter.
#
# OPTIMISATION METHODS:
# GRADIENT_BASED:
#     Uses image gradients for efficient local subpixel optimisation.
# SIMPLEX:
#     Uses derivative-free simplex optimisation for a broader but slower search.
# SIMPLEX_THEN_GRADIENT_BASED:
#     Runs simplex first and then applies gradient-based subpixel refinement.
#
# SHAPE-FUNCTION TYPES:
# AFFINE:
#     Represents translation and first-order local deformation within each subset.
#
# Active settings:
INITIALIZATION_METHOD = "USE_FEATURE_MATCHING"
OPTIMIZATION_METHOD = "SIMPLEX_THEN_GRADIENT_BASED"
SHAPE_FUNCTION_TYPE = "AFFINE"

NUM_CORES = 16   # SET AS NUMBER OF P-CORES


# =============================================================================
# ---  AUTOMATIC CALCULATION & DOCUMENTATION ---
# =============================================================================

# --- Automatic Strain Window Calculation ---
try:
    step_size_int = int(STEP_SIZE)
    subset_size_int = int(SUBSET_SIZE)
    
    if STRAIN_WINDOW_IN_SUBSETS < 3 or STRAIN_WINDOW_IN_SUBSETS % 2 == 0:
        raise ValueError("STRAIN_WINDOW_IN_SUBSETS must be an odd integer of 3 or greater.")

    # L_VSG = (L_window - 1) * L_step + L_subset
    strain_window_pixels = (STRAIN_WINDOW_IN_SUBSETS - 1) * step_size_int + subset_size_int
    
    if strain_window_pixels % 2 == 0:
        strain_window_pixels += 1
    
    STRAIN_WINDOW_SIZE = strain_window_pixels

except (ValueError, TypeError):
    print("\n[!] ERROR: Please ensure SUBSET_SIZE, STEP_SIZE, and STRAIN_WINDOW_IN_SUBSETS are valid integers.")
    input("Press ENTER to exit...")
    sys.exit(1)

try:
    dic_specs_content = f"""-------------------- DIC ANALYSIS PARAMETERS --------------------

DIC Software:                       DICe v2.0
Interpolant:                        Keys Fourth (DICe default)
Matching Criterion:                 ZNSSD (DICe default)
Shape Function Type:               {SHAPE_FUNCTION_TYPE}
Initialisation Method:              {INITIALIZATION_METHOD}
Optimisation Method:                {OPTIMIZATION_METHOD}
Strain Calculation Method:          DICe VSG strain post-processor
Strain Formulation:                 Engineering strain
Image Filtering:                    None declared in generated DICe input
Displacement Pre-Filtering:         None declared in generated DICe input
Strain Post-Filtering:              None declared in generated DICe input
Reference Image Index:              {START_IMAGE}
End Image Index:                    {END_IMAGE}
Subset Size (px):                   {SUBSET_SIZE}
Step Size (px):                     {STEP_SIZE}
Strain Window (subsets):            {STRAIN_WINDOW_IN_SUBSETS}x{STRAIN_WINDOW_IN_SUBSETS}
Calculated VSG Size:                {STRAIN_WINDOW_SIZE}px
    ROI Vertices (X Y):
        {ROI_VERTICES}
"""
    
    if os.path.exists("_specs.txt"):
        with open("_specs.txt", "r", encoding="utf-8") as f:
            existing_content = f.read()
        
        # Find the marker and slice the text to keep only what comes before it
        marker = "-------------------- DIC ANALYSIS PARAMETERS --------------------"
        if marker in existing_content:
            existing_content = existing_content.split(marker)[0].rstrip() + "\n\n"
        else:
            existing_content = existing_content.rstrip() + "\n\n"
            
        # Overwrite the file with the preserved top half + the freshly updated DIC block
        with open("_specs.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(existing_content + dic_specs_content)
    else:
        # Fallback in case the file was accidentally deleted
        with open("_specs.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(dic_specs_content)

except Exception as e:
    print(f"\n[!] WARNING: Could not update DIC parameters in _specs.txt. Error: {e}")

# =============================================================================
# ---  DICe EXECUTION ---
# =============================================================================

current_dir = os.getcwd().replace('\\', '/').rstrip('/')
results_dir = f"{current_dir}/results"
os.makedirs(results_dir, exist_ok=True)

input_xml = f"""<ParameterList>
  <Parameter name="image_folder" type="string" value="/data" />
  <Parameter name="output_folder" type="string" value="/data/results" />
  <Parameter name="image_file_prefix" type="string" value="/image_" />
  <Parameter name="image_file_extension" type="string" value=".bmp" />
  <Parameter name="num_file_suffix_digits" type="int" value="5" />
  <Parameter name="reference_image_index" type="int" value="{START_IMAGE}" />
  <Parameter name="start_image_index" type="int" value="{START_IMAGE}" />
  <Parameter name="end_image_index" type="int" value="{END_IMAGE}" />
  <Parameter name="subset_file" type="string" value="/data/subset.txt" />
  <Parameter name="subset_size" type="int" value="{SUBSET_SIZE}" />
  <Parameter name="step_size" type="int" value="{STEP_SIZE}" />
  <Parameter name="correlation_parameters_file" type="string" value="/data/params.xml" />
</ParameterList>"""

params_xml = f"""<ParameterList>
    <Parameter name="write_exodus_output" type="bool" value="true" />

    <!-- Select the method used to obtain the initial displacement estimate. -->
    <Parameter name="initialization_method" type="string" value="{INITIALIZATION_METHOD}" />

    <!-- Select the numerical optimisation sequence used for subset matching. -->
    <Parameter name="optimization_method" type="string" value="{OPTIMIZATION_METHOD}" />

    <!-- Select the local deformation model applied within each subset. -->
    <Parameter name="shape_function_type" type="string" value="{SHAPE_FUNCTION_TYPE}" />

    <!-- Export displacement, correlation-quality, status, uncertainty, and VSG
         strain fields to support correlation and strain diagnostics. -->
    <ParameterList name="output_spec">
        <Parameter name="COORDINATE_X" type="bool" value="true" />
        <Parameter name="COORDINATE_Y" type="bool" value="true" />
        <Parameter name="DISPLACEMENT_X" type="bool" value="true" />
        <Parameter name="DISPLACEMENT_Y" type="bool" value="true" />
        <Parameter name="GAMMA" type="bool" value="true" />
        <Parameter name="SIGMA" type="bool" value="true" />
        <Parameter name="BETA" type="bool" value="true" />
        <Parameter name="STATUS_FLAG" type="bool" value="true" />
        <Parameter name="UNCERTAINTY" type="bool" value="true" />
        <Parameter name="VSG_STRAIN_XX" type="bool" value="true" />
        <Parameter name="VSG_STRAIN_YY" type="bool" value="true" />
        <Parameter name="VSG_STRAIN_XY" type="bool" value="true" />
    </ParameterList>

    <ParameterList name="post_process_vsg_strain">
        <Parameter name="strain_window_size_in_pixels" type="int" value="{STRAIN_WINDOW_SIZE}" />
        <Parameter name="strain_formulation" type="string" value="ENGINEERING" />
    </ParameterList>
</ParameterList>"""

subset_content = f"""begin region_of_interest
  begin boundary
    begin polygon
      begin vertices
{ROI_VERTICES}
      end vertices
    end polygon
  end boundary
end region_of_interest"""

with open("input.xml", "w", encoding="utf-8", newline="\n") as f:
    f.write(input_xml.strip())
with open("params.xml", "w", encoding="utf-8", newline="\n") as f:
    f.write(params_xml.strip())
with open("subset.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(subset_content.strip())

print(f"Using a {STRAIN_WINDOW_IN_SUBSETS}x{STRAIN_WINDOW_IN_SUBSETS} subset grid, calculated strain window size: {STRAIN_WINDOW_SIZE} pixels.")
print(f"DICe initialisation: {INITIALIZATION_METHOD}")
print(f"DICe optimisation: {OPTIMIZATION_METHOD}")
print(f"DICe shape function: {SHAPE_FUNCTION_TYPE}")
print(f"Image Sequence: {START_IMAGE} ~ {END_IMAGE}")
print(f"Executing DICe engine on {NUM_CORES} cores...")

# HPC LEASH: Prevent OpenMP from oversubscribing logical threads
os.environ["OMP_NUM_THREADS"] = "1"

try:

    docker_cmd = [
        "docker", "run", "--rm", 
        "-v", f"{current_dir}:/data", 
        "dice-mpi:v2.0", 
        "mpiexec", "--allow-run-as-root", "-n", str(NUM_CORES), 
        "dice", "-i", "/data/input.xml", "-v"
    ]

    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    
    with open("dice_log.txt", "w", encoding="utf-8") as f:
        f.write("--- STDOUT ---\n" + result.stdout + "\n--- STDERR ---\n" + result.stderr)

    if result.returncode != 0:
        print("\n" + "="*50)
        print("[!] FATAL ERROR OR MPI FAILURE [!]")
        print("See dice_log.txt for detailed error messages.")
        print(result.stderr)
        print("="*50 + "\n")
        sys.exit(1)
        
except FileNotFoundError:
    print("\n" + "="*50)
    print("[!] DOCKER ENVIRONMENT NOT FOUND [!]")
    print("The 'docker' command is not recognized by Windows. Ensure Docker Desktop is running.")
    print("="*50 + "\n")
    sys.exit(1)
except Exception as e:
    print(f"Critical execution error: {e}")
    sys.exit(1)

print("Execution complete! Check the 'results' folder and dice_log.txt.")
print("\n--- PARAVIEW EXPORT FILENAMES REQUIRED BY merge.py ---")
print("Global/full-ROI export:  results/data_index.csv")
print("Local-region exports:    results/pv_<name>.csv")
print("Example local exports:   results/pv_centre.csv, results/pv_left.csv")
print("Save every ParaView CSV directly inside the 'results' folder.")
print("The global filename must be exactly 'data_index.csv'.")
print("Each local filename must begin with 'pv_' and end with '.csv'.")
print("After exporting the CSV files, run merge.py from this experiment directory.")

input("\nPress ENTER to close the window...")
'''

with open(run_dice_filename, 'w', encoding="utf-8", newline="\n") as f:
    f.write(run_dice_code)

# C. Generate select_roi.py (interactive image-range and ROI selector)
selector_filename = os.path.join(save_directory, "select_roi.py")
selector_code = r'''#!/usr/bin/env python3
"""Select the DIC image range and rectangular ROI after imgstab.py."""
from __future__ import annotations
import ast, csv, math, os, re
from pathlib import Path
import cv2
import numpy as np

WIN = "AZ31 DIC Image Range and ROI"
RED, CYAN, WHITE, BLACK, GREEN = (0,0,255), (255,255,0), (255,255,255), (0,0,0), (0,220,0)
VW, VH, JUMP = 1180, 760, 10
BEGIN = "# BEGIN SELECT_ROI MANAGED SETTINGS"
END = "# END SELECT_ROI MANAGED SETTINGS"
SB = "-------------------- REFERENCE IMAGE AND ROI SELECTION --------------------"
SE = "-------------------- END REFERENCE IMAGE AND ROI SELECTION --------------------"
LEFT, RIGHT = {2424832,65361}, {2555904,65363}
UP, DOWN = {2490368,65362}, {2621440,65364}
BACK = {8,127}

def idx(p):
    m = re.fullmatch(r"image_(\d{5})\.bmp", p.name, re.I)
    if not m: raise ValueError(f"Invalid acquisition filename: {p.name}")
    return int(m.group(1))

def files_at(work):
    fs = sorted((x for x in work.iterdir() if re.fullmatch(r"image_\d{5}\.bmp", x.name, re.I)), key=idx)
    if not fs: raise FileNotFoundError("No image_#####.bmp files found")
    if len({idx(x) for x in fs}) != len(fs): raise ValueError("Duplicate image indices")
    return fs

def load(p):
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None: raise OSError(f"Could not load {p.name}")
    if a.ndim == 2: return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 4: return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    return a[:,:,:3].copy()

def fit(a):
    h,w = a.shape[:2]; z = min(VW/w, VH/h, 1.0); wh=(max(1,round(w*z)),max(1,round(h*z)))
    return (a.copy() if wh==(w,h) else cv2.resize(a,wh,interpolation=cv2.INTER_AREA)), z

def text(a, lines, colour=WHITE):
    y=24
    for line in lines:
        cv2.putText(a,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.53,BLACK,3,cv2.LINE_AA)
        cv2.putText(a,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.53,colour,1,cv2.LINE_AA); y+=22

def parse_dice(path):
    s=path.read_text(encoding="utf-8")
    sm=re.search(r'^\s*SUBSET_SIZE\s*=\s*["\'](\d+)["\']',s,re.M)
    tm=re.search(r'^\s*STEP_SIZE\s*=\s*["\'](\d+)["\']',s,re.M)
    if not sm or not tm: raise ValueError("SUBSET_SIZE and STEP_SIZE must be integer strings")
    if s.count(BEGIN)!=1 or s.count(END)!=1: raise ValueError("Managed settings markers missing or duplicated")
    return s,int(sm.group(1)),int(tm.group(1))

def valid_rows(work,name,h):
    p=work/'stabilisation_log.csv'
    if not p.exists(): return 0,h,"stabilisation_log.csv unavailable"
    try:
        for row in csv.DictReader(p.open(encoding='utf-8-sig',newline='')):
            if Path(row.get('fixed_file','')).name.lower()==name.lower():
                m=re.fullmatch(r'\s*(\d+)\s*:\s*(\d+)\s*',row.get('destination_rows',''))
                if m:
                    a,b=map(int,m.groups())
                    if 0<=a<b<=h: return a,b,""
        return 0,h,"reference valid rows absent from stabilisation log"
    except Exception as e: return 0,h,f"stabilisation log unreadable: {e}"

def detect(a,lo,hi):
    g=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(float); h,w=g.shape; tops=[]; bots=[]
    kernel=np.ones(7)/7; sep=max(10,round((hi-lo)*.15))
    for f0,f1 in ((.2,.4),(.4,.6),(.6,.8)):
        profile=np.convolve(g[:,round(w*f0):round(w*f1)].mean(1),kernel,mode='same')
        d=np.gradient(profile); region=d[lo:hi]; top=lo+int(np.argmax(region)); bottom=None
        for q in np.argsort(region):
            y=lo+int(q)
            if y>=top+sep: bottom=y; break
        if bottom is not None: tops.append(top); bots.append(bottom)
    if not tops: raise ValueError("no valid automatic edge pair")
    return round(float(np.median(tops))),round(float(np.median(bots))),f"band tops={tops}, bottoms={bots}"

def draw_adjust(a,b,active):
    v,z=fit(a); h,w=v.shape[:2]
    for n in ('top','bottom'):
        y=max(0,min(h-1,round(b[n]*z))); cv2.line(v,(0,y),(w-1,y),CYAN if n==active else RED,1,cv2.LINE_8)
    for n in ('left','right'):
        x=max(0,min(w-1,round(b[n]*z))); cv2.line(v,(x,0),(x,h-1),CYAN if n==active else RED,1,cv2.LINE_8)
    return v

def draw_summary(a,roi):
    v,z=fit(a); h,w=v.shape[:2]
    for y in (roi['top'],roi['bottom']): cv2.line(v,(0,round(y*z)),(w-1,round(y*z)),RED,1,cv2.LINE_8)
    for x in (roi['left'],roi['right']): cv2.line(v,(round(x*z),0),(round(x*z),h-1),RED,1,cv2.LINE_8)
    return v

def missing(fs,a,b):
    present={idx(x) for x in fs if a<=idx(x)<=b}; absent=sorted(set(range(a,b+1))-present)
    return b-a+1,len(present),absent

def desc(xs,n=12):
    if not xs:return 'None'
    q=', '.join(map(str,xs[:n])); return q if len(xs)<=n else f"{q} ... ({len(xs)} total)"

def errors(b,h,w,inset,subset,step):
    e=[]; rt=b['top']+inset; rb=b['bottom']-inset
    if not 0<=b['top']<b['bottom']<h:e.append('invalid specimen top/bottom order')
    if not 0<=rt<rb<h or rb-rt<subset:e.append('vertical ROI is too small after inset')
    if not 0<=b['left']<b['right']<w:e.append('invalid left/right order')
    if b['right']-b['left']<subset+2*step:e.append('ROI width is too small')
    return e

def managed(first,end,roi):
    return "\n".join([BEGIN,"",f'START_IMAGE = "{first}"',f'END_IMAGE = "{end}"',"",
        "# FORMAT: X Y, ordered TL, TR, BR, BL.",'ROI_VERTICES = """\\',
        f"{roi['left']} {roi['top']}",f"{roi['right']} {roi['top']}",
        f"{roi['right']} {roi['bottom']}",f"{roi['left']} {roi['bottom']}",'"""',"",END])

def replace_block(s,repl,begin=BEGIN,end=END):
    a=s.index(begin); b=s.index(end,a)+len(end); return s[:a]+repl+s[b:]

def verify(s,first,end,roi):
    ast.parse(s)
    if not re.search(rf'^START_IMAGE\s*=\s*"{first}"$',s,re.M):raise RuntimeError('START_IMAGE verification failed')
    if not re.search(rf'^END_IMAGE\s*=\s*"{end}"$',s,re.M):raise RuntimeError('END_IMAGE verification failed')
    expected=[(roi['left'],roi['top']),(roi['right'],roi['top']),(roi['right'],roi['bottom']),(roi['left'],roi['bottom'])]
    m=re.search(r'^ROI_VERTICES\s*=\s*"""\\\n(.*?)\n"""',s,re.M|re.S)
    if not m or [tuple(map(int,x.split())) for x in m.group(1).splitlines()]!=expected:raise RuntimeError('ROI verification failed')

def specs(firstp,endp,first,end,span,count,absent,subset,inset,b,roi):
    return f"""{SB}
Selection Script:                   select_roi.py
Reference Image File:               {firstp.name}
Reference Image Index:              {first}
End Image File:                     {endp.name}
End Image Index:                    {end}
Index Range:                        {first} to {end}
Index Span:                         {span}
Available Images in Range:          {count}
Missing Image Indices in Range:     {desc(absent,20)}

Subset Size (px):                   {subset}
ROI Edge Inset Rule:                ceil(1.5 x subset size)
ROI Edge Inset (px):                {inset}
Validated Specimen Top Y (px):      {b['top']}
Validated Specimen Bottom Y (px):   {b['bottom']}
ROI Left X (px):                    {roi['left']}
ROI Right X (px):                   {roi['right']}
ROI Top Y (px):                     {roi['top']}
ROI Bottom Y (px):                  {roi['bottom']}
ROI Width (px):                     {roi['right']-roi['left']}
ROI Height (px):                    {roi['bottom']-roi['top']}
ROI Vertices (X Y):
    Top Left:                       {roi['left']} {roi['top']}
    Top Right:                      {roi['right']} {roi['top']}
    Bottom Right:                   {roi['right']} {roi['bottom']}
    Bottom Left:                    {roi['left']} {roi['bottom']}
Reference BMP:                      reference_image.bmp
Reference PNG:                      reference_image.png
{SE}"""

def upsert(s,block):
    if SB in s and SE in s:return replace_block(s,block,SB,SE)
    return s.rstrip()+"\n\n"+block+"\n"

def write_all(work,dice,original,firstp,endp,subset,inset,b,roi,span,count,absent,image):
    first,end=idx(firstp),idx(endp); new=replace_block(original,managed(first,end,roi)); verify(new,first,end,roi)
    old=(work/'_specs.txt').read_text(encoding='utf-8') if (work/'_specs.txt').exists() else ''
    new_specs=upsert(old,specs(firstp,endp,first,end,span,count,absent,subset,inset,b,roi))
    dt,st=work/'run_dice.py.tmp',work/'_specs.txt.tmp'; dt.write_text(new,encoding='utf-8',newline='\n'); st.write_text(new_specs,encoding='utf-8',newline='\n')
    verify(dt.read_text(encoding='utf-8'),first,end,roi); os.replace(dt,dice); os.replace(st,work/'_specs.txt')
    legacy_backup=work/'run_dice.py.bak'
    if legacy_backup.exists(): legacy_backup.unlink()
    out=image.copy(); cv2.rectangle(out,(roi['left'],roi['top']),(roi['right'],roi['bottom']),RED,3,cv2.LINE_8)
    for name in ('reference_image.bmp','reference_image.png'):
        path=work/name
        if not cv2.imwrite(str(path),out) or cv2.imread(str(path),cv2.IMREAD_COLOR) is None:raise OSError(f"Could not verify {name}")
    verify(dice.read_text(encoding='utf-8'),first,end,roi)

def main():
    work=Path(__file__).resolve().parent; dice=work/'run_dice.py'; fs=files_at(work); original,subset,step=parse_dice(dice); inset=math.ceil(1.5*subset)
    state='range'; cur=0; fp=ep=None; image=auto=b=None; active='top'; note=notice=''
    cv2.namedWindow(WIN,cv2.WINDOW_AUTOSIZE); cv2.createTrackbar('Image',WIN,0,len(fs)-1,lambda x:None)
    while True:
        if state in ('range','range_summary'):
            cur=cv2.getTrackbarPos('Image',WIN); v,_=fit(load(fs[cur])); f='not set' if fp is None else str(idx(fs[fp])); e='not set' if ep is None else str(idx(fs[ep]))
            if state=='range': lines=['IMAGE RANGE SELECTION',f'Current: {fs[cur].name} [{cur+1}/{len(fs)}]',f'First/reference: {f} | End: {e}','Arrows or A/D: 1 | W/S: 10 | trackbar: browse','F: first | E: end | ENTER: review | BACKSPACE: clear | Q: cancel']
            else:
                span,count,absent=missing(fs,idx(fs[fp]),idx(fs[ep])); lines=['SELECTED IMAGE RANGE',f'Reference: {fs[fp].name} | End: {fs[ep].name}',f'Index span: {span} | Files: {count} | Missing: {desc(absent)}','ENTER: proceed | BACKSPACE: re-adjust | Q: cancel']
            if notice:lines.append('NOTICE: '+notice)
            text(v,lines)
        elif state=='adjust':
            v=draw_adjust(image,b,active); rt,rb=b['top']+inset,b['bottom']-inset
            lines=['BOUNDARY ADJUSTMENT',f'Reference: {fs[fp].name} | subset {subset}px | inset {inset}px',f'Specimen top {b["top"]} | bottom {b["bottom"]}',f'Calculated ROI top {rt} | bottom {rb} | left {b["left"]} | right {b["right"]}',f'ACTIVE: {active.upper()} (cyan); others red','T/B/L/R: activate | arrows or WASD: 1px | [ ]: 10px | 0: reset','ENTER: summary | BACKSPACE: image-range summary | Q: cancel','Detection: '+note]
            if notice:lines.append('NOTICE: '+notice)
            text(v,lines)
        else:
            roi={'top':b['top']+inset,'bottom':b['bottom']-inset,'left':b['left'],'right':b['right']}; v=draw_summary(image,roi)
            text(v,['FINAL SELECTION SUMMARY',f'Images: {idx(fs[fp])} to {idx(fs[ep])} | subset {subset}px | inset {inset}px',f'Validated specimen: top {b["top"]} | bottom {b["bottom"]}',f'Final ROI: L {roi["left"]} R {roi["right"]} T {roi["top"]} B {roi["bottom"]}',f'TL {roi["left"]} {roi["top"]} | TR {roi["right"]} {roi["top"]}',f'BR {roi["right"]} {roi["bottom"]} | BL {roi["left"]} {roi["bottom"]}','ENTER: confirm and write | BACKSPACE: adjust | Q: cancel'])
        cv2.imshow(WIN,v); key=cv2.waitKeyEx(20)
        if key in (27,ord('q'),ord('Q')):print('Cancelled. No files changed.');break
        if key==-1:continue
        notice=''
        if state=='range':
            if key in LEFT or key in (ord('a'),ord('A')):cur=max(0,cur-1);cv2.setTrackbarPos('Image',WIN,cur)
            elif key in RIGHT or key in (ord('d'),ord('D')):cur=min(len(fs)-1,cur+1);cv2.setTrackbarPos('Image',WIN,cur)
            elif key in UP or key in (ord('w'),ord('W')):cur=max(0,cur-JUMP);cv2.setTrackbarPos('Image',WIN,cur)
            elif key in DOWN or key in (ord('s'),ord('S')):cur=min(len(fs)-1,cur+JUMP);cv2.setTrackbarPos('Image',WIN,cur)
            elif key in (ord('f'),ord('F')):fp=cur;ep=None if ep is not None and idx(fs[ep])<idx(fs[fp]) else ep
            elif key in (ord('e'),ord('E')):ep=cur
            elif key in BACK: ep=None if ep is not None else ep; fp=None if ep is None and fp is not None else fp
            elif key in (10,13):
                if fp is None or ep is None:notice='Set first (F) and end (E)'
                elif idx(fs[fp])>idx(fs[ep]):notice='End must not precede first'
                else:state='range_summary'
        elif state=='range_summary':
            if key in BACK:state='range'
            elif key in (10,13):
                image=load(fs[fp]);h,w=image.shape[:2];lo,hi,warn=valid_rows(work,fs[fp].name,h)
                try:top,bottom,note=detect(image,lo,hi)
                except Exception as ex:top,bottom=round(h*.25),round(h*.75);note=f'automatic detection failed: {ex}'
                if warn:note+='; WARNING: '+warn
                auto={'top':top,'bottom':bottom,'left':round(w*.25),'right':round(w*.75)};b=auto.copy();active='top';state='adjust'
        elif state=='adjust':
            h,w=image.shape[:2]
            if key in BACK:state='range_summary'
            elif key in (ord('t'),ord('T')):active='top'
            elif key in (ord('b'),ord('B')):active='bottom'
            elif key in (ord('l'),ord('L')):active='left'
            elif key in (ord('r'),ord('R')):active='right'
            elif key==ord('0'):b[active]=auto[active]
            elif key in (10,13):
                es=errors(b,h,w,inset,subset,step);notice='; '.join(es)
                if not es:state='summary'
            else:
                d=-10 if key==ord('[') else 10 if key==ord(']') else 0
                if not d and active in ('top','bottom'):
                    d=-1 if key in UP or key in (ord('w'),ord('W')) else 1 if key in DOWN or key in (ord('s'),ord('S')) else 0
                elif not d:
                    d=-1 if key in LEFT or key in (ord('a'),ord('A')) else 1 if key in RIGHT or key in (ord('d'),ord('D')) else 0
                if d:b[active]=max(0,min((h if active in ('top','bottom') else w)-1,b[active]+d))
        else:
            if key in BACK:state='adjust'
            elif key in (10,13):
                roi={'top':b['top']+inset,'bottom':b['bottom']-inset,'left':b['left'],'right':b['right']};span,count,absent=missing(fs,idx(fs[fp]),idx(fs[ep]));write_all(work,dice,original,fs[fp],fs[ep],subset,inset,b,roi,span,count,absent,image)
                done=np.zeros((390,850,3),np.uint8);text(done,['SELECTION WRITTEN SUCCESSFULLY','Overwrote ROI settings in run_dice.py','Updated the ROI block in _specs.txt','Overwrote reference_image.bmp and reference_image.png','Press any key to close.'],GREEN);cv2.imshow(WIN,done);cv2.waitKeyEx(0);break
    cv2.destroyAllWindows()
if __name__=='__main__':
    try:main()
    except Exception as e:
        cv2.destroyAllWindows();print('\nFATAL ERROR:',e);input('Press ENTER to close...');raise
'''
with open(selector_filename, 'w', encoding="utf-8", newline="\n") as f:
    f.write(selector_code)

# D. Generate merge.py (FULLY AUTOMATED)
merge_filename = os.path.join(save_directory, "merge.py")
merge_code = r'''import os
import glob
import shutil
import pandas as pd
import sys
import re

# =============================================================================
# ---  MERGE RESULTS WORKFLOW ---
# This script performs two main tasks:
# 1. File Cleanup: Moves all DICe output files into the 'results' directory.
# 2. Data Merge: Combines the master_log.csv with data_index.csv, AND
#    dynamically appends any subset CSVs prefixed with 'pv_'.
# =============================================================================

def get_dice_start_index():
    """
    Parses 'run_dice.py' to automatically find the START_IMAGE index.
    """
    run_dice_path = 'run_dice.py'
    print(f"Attempting to read START_IMAGE from '{run_dice_path}'...")
    
    try:
        with open(run_dice_path, 'r') as f:
            content = f.read()
            match = re.search(r"^\s*START_IMAGE\s*=\s*['\"]?(\d+)['\"]?", content, re.MULTILINE)
            
            if match:
                start_index = int(match.group(1))
                print(f"Successfully read START_IMAGE = {start_index}")
                return start_index
            else:
                print("\n[!] FATAL ERROR: Could not automatically find START_IMAGE in 'run_dice.py'.")
                sys.exit(1)
                
    except FileNotFoundError:
        print(f"\n[!] FATAL ERROR: '{run_dice_path}' not found in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] An unexpected error occurred while reading 'run_dice.py': {e}")
        sys.exit(1)

start_index = get_dice_start_index()

# --- 1. FILE CLEANUP ---
print("\n--- Task 1: Cleaning up DICe output files ---")
current_dir = os.getcwd()
results_dir = os.path.join(current_dir, "results")

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

files_to_move = list(set(
    glob.glob(os.path.join(current_dir, "*DICe_solution*")) +
    glob.glob(os.path.join(current_dir, "*resultstiming*")) +
    glob.glob(os.path.join(current_dir, "*.e")) +
    glob.glob(os.path.join(current_dir, "dice_log.txt"))
))

if not files_to_move:
    print("No DICe output files found in the main directory. Skipping file move.")
else:
    for file_path in files_to_move:
        file_name = os.path.basename(file_path)
        destination = os.path.join(results_dir, file_name)
        if os.path.exists(destination):
            os.remove(destination)
        shutil.move(file_path, destination)
        print(f"Moved: {file_name} -> results/")
print("File cleanup complete.\n")

# --- 2. DATA MERGE ---
print("--- Task 2: Merging temporal and strain data ---")
# These variables remain untouched to maintain compatibility with your parent script
master_log_path = os.path.join(results_dir, "master_log.csv")
strain_index_path = os.path.join(results_dir, "data_index.csv")
output_path = os.path.join(results_dir, "strain_time.csv")

if not os.path.exists(master_log_path):
    print(f"[!] FATAL ERROR: Cannot find master_log.csv in '{results_dir}'")
    sys.exit(1)
if not os.path.exists(strain_index_path):
    print(f"[!] FATAL ERROR: Cannot find data_index.csv in '{results_dir}'")
    sys.exit(1)

try:
    master_df = pd.read_csv(master_log_path)
    strain_df = pd.read_csv(strain_index_path)
    print("Successfully loaded master_log.csv and strain_index.csv.")

    paraview_time_col = "Time"
    if paraview_time_col not in strain_df.columns:
        print(f"[!] FATAL ERROR: Could not find '{paraview_time_col}' in strain_index.csv.")
        sys.exit(1)

    # Dynamically rename the global strain columns (e.g., avg(STRAIN(0)) -> avg(STRAIN(0))_Global)
    global_rename = {col: f"{col}_Global" for col in strain_df.columns if col != paraview_time_col}
    strain_df = strain_df.rename(columns=global_rename)

    # Look for any additional subset CSVs exported from ParaView using the 'pv_' prefix
    subset_files = glob.glob(os.path.join(results_dir, "pv_*.csv"))
    
    for file_path in subset_files:
        file_name = os.path.basename(file_path)
        print(f"Merging subset data: {file_name}")
        temp_df = pd.read_csv(file_path)
        
        # Extract a clean name (e.g., "pv_centre.csv" -> "centre")
        clean_name = file_name.replace('pv_', '').replace('.csv', '')
        
        # Rename columns to prevent clashes (e.g., avg(STRAIN(0)) -> avg(STRAIN(0))_centre)
        subset_rename = {col: f"{col}_{clean_name}" for col in temp_df.columns if col != paraview_time_col}
        temp_df = temp_df.rename(columns=subset_rename)
        
        # Merge this subset into the main strain dataframe side-by-side
        strain_df = pd.merge(strain_df, temp_df, on=paraview_time_col, how='outer')

    # True Image Index = (ParaView "Time" Step - 1) + START_IMAGE
    strain_df['Image_Number'] = (strain_df[paraview_time_col].astype(int) - 1) + start_index
    
    print(f"Aligning data using automatically detected start index: {start_index}.")

    # Final merge with master log
    final_df = pd.merge(
        master_df,
        strain_df,
        on='Image_Number',
        how='inner'
    )

    # Drop the now-redundant ParaView "Time" column
    final_df = final_df.drop(columns=[paraview_time_col])
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("SUCCESS!")
    print(f"Final merged data saved to: {output_path}")
    print("="*50)

except Exception as e:
    print(f"\n[!] An error occurred during the data merge process: {e}")
    sys.exit(1)

input("\nPress ENTER to close the window...")
'''
with open(merge_filename, 'w', encoding="utf-8", newline="\n") as f:
    f.write(merge_code)

# E. Generate imgstab.py (integer-only vertical image stabiliser)
imgstab_filename = os.path.join(save_directory, "imgstab.py")
imgstab_code = r'''#!/usr/bin/env python3
"""Integer-only vertical stabilisation for DIC BMP sequences.

Place this script beside the source BMP files and run it. The script moves all
source BMPs into ./original_images, then writes fixed BMPs with the same names
back beside the script. Output pixels are changed only by whole-row array copies;
newly exposed rows are zero-filled black.
"""
from __future__ import annotations
import argparse, csv, hashlib, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

@dataclass
class TrackingResult:
    lag_y: int
    score: float
    second_score: float
    confidence_margin: float

def natural_key(p: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]

def load_image(path: Path):
    with Image.open(path) as im:
        mode = im.mode
        a = np.asarray(im).copy()
    if a.dtype not in (np.uint8, np.uint16) or a.ndim not in (2, 3):
        raise ValueError(f"Unsupported image format in {path.name}: {a.shape}, {a.dtype}")
    return a, mode

def grey_for_tracking(a: np.ndarray):
    if a.ndim == 2:
        return a.astype(np.float64)
    return a[..., :3].astype(np.float64).mean(axis=2)

def edge_profile(a: np.ndarray, x0: int, x1: int):
    g = grey_for_tracking(a)
    if not (0 <= x0 < x1 <= g.shape[1]):
        raise ValueError("Invalid tracking band")
    # Average many columns so random black speckles cancel while the long,
    # continuous specimen boundaries remain strong. Median was too sensitive
    # when more than half of a row segment happened to contain dark speckles.
    rows = np.mean(g[:, x0:x1], axis=1)

    # Smooth only this temporary tracking signal, never the output image.
    # A seven-row box average suppresses speckle-scale fluctuations.
    kernel = np.ones(7, dtype=np.float64) / 7.0
    rows = np.convolve(rows, kernel, mode="same")

    # Keep the derivative sign: the top edge is positive and the bottom edge
    # negative. This prevents unrelated internal speckle transitions from
    # looking equivalent to both specimen boundaries.
    edge = np.gradient(rows)
    edge -= np.mean(edge)
    norm = np.linalg.norm(edge)
    if norm == 0:
        raise ValueError("No detectable specimen edge in tracking band")
    return edge / norm

def overlap(ref, cur, lag, margin):
    lo, hi = margin, len(ref) - margin
    if lag >= 0:
        return ref[lo:hi-lag], cur[lo+lag:hi]
    return ref[lo-lag:hi], cur[lo:hi+lag]

def score(a, b):
    if len(a) < 8:
        return -np.inf
    a, b = a-a.mean(), b-b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else -np.inf

def estimate_lag(ref, cur, max_shift, margin):
    values = []
    for lag in range(-max_shift, max_shift + 1):
        a, b = overlap(ref, cur, lag, margin)
        values.append((score(a, b), lag))
    values.sort(reverse=True)
    best, lag = values[0]
    second = values[1][0]
    return TrackingResult(lag, best, second, best-second)

def translate_y(a: np.ndarray, correction: int):
    h = a.shape[0]
    if abs(correction) >= h:
        raise ValueError(f"Correction {correction} px removes the whole image")
    out = np.zeros_like(a)
    if correction == 0:
        out[...] = a
        return out, (0, h), (0, h)
    if correction > 0:
        src, dst = (0, h-correction), (correction, h)
    else:
        src, dst = (-correction, h), (0, h+correction)
    out[dst[0]:dst[1], ...] = a[src[0]:src[1], ...]
    return out, src, dst

def sha256_rows(a, rows):
    data = np.ascontiguousarray(a[rows[0]:rows[1], ...]).tobytes()
    return hashlib.sha256(data).hexdigest()

def save_bmp_verified(a, mode, path):
    Image.fromarray(a, mode=mode).save(path, format="BMP")
    reopened, reopened_mode = load_image(path)
    if reopened.shape != a.shape or reopened_mode != mode or not np.array_equal(reopened, a):
        raise RuntimeError(f"Saved BMP verification failed: {path.name}")

def main():
    ap = argparse.ArgumentParser(description="Stabilise BMPs beside this script using integer row copies")
    ap.add_argument("--pattern", default="*.bmp")
    ap.add_argument("--track-start", type=float, default=0.70)
    ap.add_argument("--track-end", type=float, default=0.95)
    ap.add_argument("--max-shift", type=int, default=300)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.70)
    ap.add_argument("--min-margin", type=float, default=0.005)
    args = ap.parse_args()
    if not 0 <= args.track_start < args.track_end <= 1:
        raise ValueError("Require 0 <= track-start < track-end <= 1")

    work = Path(__file__).resolve().parent
    originals = work / "original_images"
    files = sorted([p for p in work.glob(args.pattern) if p.is_file()], key=natural_key)
    if not files:
        raise FileNotFoundError(f"No {args.pattern} files found beside the script")
    if originals.exists() and any(originals.iterdir()):
        raise FileExistsError("original_images is not empty. Rename or move it before a new run.")
    originals.mkdir(exist_ok=True)

    print(f"Moving {len(files)} original image(s) into: {originals}")
    moved = []
    for p in files:
        q = originals / p.name
        p.replace(q)
        moved.append(q)

    reference, ref_mode = load_image(moved[0])
    h, w = reference.shape[:2]
    x0, x1 = round(w*args.track_start), round(w*args.track_end)
    ref_profile = edge_profile(reference, x0, x1)
    log_path = work / "stabilisation_log.csv"
    fields = ["frame","original_file","fixed_file","observed_lag_y_px",
              "applied_correction_y_px","correlation_score","confidence_margin",
              "tracking_ok","source_rows","destination_rows","copied_pixels_exact",
              "valid_pixel_sha256"]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, p in enumerate(moved):
            a, mode = load_image(p)
            if a.shape != reference.shape or a.dtype != reference.dtype or mode != ref_mode:
                raise ValueError(f"Image format mismatch: {p.name}")
            result = TrackingResult(0, 1.0, np.nan, np.nan) if i == 0 else estimate_lag(
                ref_profile, edge_profile(a, x0, x1), args.max_shift, args.margin)
            correction = -result.lag_y
            fixed, src, dst = translate_y(a, correction)
            exact = np.array_equal(a[src[0]:src[1], ...], fixed[dst[0]:dst[1], ...])
            if not exact:
                raise RuntimeError(f"Pixel-copy check failed: {p.name}")
            output = work / p.name
            save_bmp_verified(fixed, mode, output)
            ok = i == 0 or (result.score >= args.min_score and result.confidence_margin >= args.min_margin)
            writer.writerow({"frame":i,"original_file":f"original_images/{p.name}",
                "fixed_file":p.name,"observed_lag_y_px":result.lag_y,
                "applied_correction_y_px":correction,"correlation_score":f"{result.score:.8f}",
                "confidence_margin":"" if i == 0 else f"{result.confidence_margin:.8f}",
                "tracking_ok":ok,"source_rows":f"{src[0]}:{src[1]}",
                "destination_rows":f"{dst[0]}:{dst[1]}","copied_pixels_exact":exact,
                "valid_pixel_sha256":sha256_rows(fixed, dst)})
            print(f"[{i+1:04d}/{len(moved):04d}] {'OK' if ok else 'REVIEW':6s} {p.name}: "
                  f"observed={result.lag_y:+d}px, correction={correction:+d}px, score={result.score:.5f}")
    print(f"\nFinished. Originals: {originals}")
    print(f"Fixed BMPs remain in: {work}")
    print(f"Audit log: {log_path}")

if __name__ == "__main__":
    main()
'''
with open(imgstab_filename, 'w', encoding="utf-8", newline="\n") as f:
    f.write(imgstab_code)

# F. Generate focus_monitor.py (Insert the monitor code string here)
monitor_filename = os.path.join(save_directory, "monitor.py")
monitor_code = r'''"""Live image monitor for experimental camera focus adjustment.

Designed around 1280 x 1024 experimental images while remaining usable with
other resolutions. The whole-frame view preserves aspect ratio. The focus view
uses a true 1:1 crop, meaning that OpenCV performs no image resampling.
"""

import re
import time
from pathlib import Path

import cv2


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

WINDOW_NAME = "AZ31 DIC Live Monitor"

# Directory containing the images. Path.cwd() means the directory from which
# this monitor script is launched.
IMAGE_DIRECTORY = Path.cwd()

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".png",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
}

# Polling interval for detecting new files.
POLL_INTERVAL_S = 0.01

# Number of consecutive polls for which a file must remain unchanged before
# being loaded. This helps avoid reading an image that is still being written.

# Both display modes use the same viewport dimensions. A 1000 x 800 viewport
# has the same 5:4 aspect ratio as the 1280 x 1024 experimental images.
VIEW_WIDTH = 1000
VIEW_HEIGHT = 800

# Pan distance in source-image pixels per key press during 1:1 focus view.
PAN_STEP = 25

# Expected dimensions are used only for a console warning. They do not limit
# operation with images of other sizes.
EXPECTED_IMAGE_WIDTH = 1280
EXPECTED_IMAGE_HEIGHT = 1024


# -----------------------------------------------------------------------------
# File discovery and loading
# -----------------------------------------------------------------------------

def extract_image_index(path: Path):
    """Return the last group of digits in a filename, preserving leading zeros."""
    matches = re.findall(r"\d+", path.stem)
    return matches[-1] if matches else None


def find_latest_image(directory: Path):
    """Return the final image having the greatest numerical image index."""
    try:
        indexed_candidates = []
        for path in directory.iterdir():
            if not path.is_file():
                continue
            match = re.fullmatch(r"image_(\d+)\.bmp", path.name, re.IGNORECASE)
            if match:
                indexed_candidates.append((int(match.group(1)), path))
    except OSError:
        return None
    return max(indexed_candidates, key=lambda item: item[0])[1] if indexed_candidates else None


def make_file_signature(path: Path):
    """Return values that change when an image file is replaced or modified."""
    try:
        stat = path.stat()
        return path, stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def bmp_file_is_complete(path: Path):
    """Return True when a BMP has reached the size declared in its header."""
    try:
        actual_size = path.stat().st_size
        if actual_size < 6:
            return False
        with path.open("rb") as bmp_file:
            header = bmp_file.read(6)
        if len(header) != 6 or header[:2] != b"BM":
            return False
        declared_size = int.from_bytes(header[2:6], "little", signed=False)
        return declared_size >= 6 and actual_size >= declared_size
    except OSError:
        return False


def load_image(path: Path):
    """Load the stored image at its native resolution and bit depth."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None or image.size == 0:
        return None

    return image


# -----------------------------------------------------------------------------
# Image-view construction
# -----------------------------------------------------------------------------

def geometric_centre(image):
    """Return the geometric centre of the source image."""
    height, width = image.shape[:2]
    return width // 2, height // 2


def clamp(value, lower, upper):
    """Restrict a value to an inclusive interval."""
    return max(lower, min(value, upper))


def create_whole_frame_view(image):
    """Fit the complete image within the viewport without changing aspect ratio."""
    source_height, source_width = image.shape[:2]

    scale = min(
        VIEW_WIDTH / source_width,
        VIEW_HEIGHT / source_height,
        1.0,
    )

    output_width = max(1, round(source_width * scale))
    output_height = max(1, round(source_height * scale))

    if output_width == source_width and output_height == source_height:
        return image.copy()

    return cv2.resize(
        image,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def create_focus_view(image, centre_x, centre_y):
    """Create an unscaled 1:1 crop around the requested source-image position."""
    source_height, source_width = image.shape[:2]

    crop_width = min(VIEW_WIDTH, source_width)
    crop_height = min(VIEW_HEIGHT, source_height)

    # This form correctly supports both odd and even crop dimensions.
    minimum_centre_x = crop_width // 2
    maximum_centre_x = source_width - (crop_width - crop_width // 2)
    minimum_centre_y = crop_height // 2
    maximum_centre_y = source_height - (crop_height - crop_height // 2)

    centre_x = clamp(int(centre_x), minimum_centre_x, maximum_centre_x)
    centre_y = clamp(int(centre_y), minimum_centre_y, maximum_centre_y)

    x0 = centre_x - crop_width // 2
    y0 = centre_y - crop_height // 2
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    view = image[y0:y1, x0:x1].copy()
    bounds = (x0, y0, x1, y1)

    return view, centre_x, centre_y, bounds


# -----------------------------------------------------------------------------
# Window title and keyboard handling
# -----------------------------------------------------------------------------

def create_window_title(path, source_image, focus_mode, crop_bounds=None):
    """Construct a title containing the current image index and view mode."""
    source_height, source_width = source_image.shape[:2]
    image_index = extract_image_index(path)
    index_text = image_index if image_index is not None else "not detected"

    if focus_mode and crop_bounds is not None:
        x0, y0, x1, y1 = crop_bounds
        mode_text = f"Focus zoom 1:1 | ROI x={x0}:{x1}, y={y0}:{y1}"
    else:
        mode_text = "Whole frame"

    return (
        f"{WINDOW_NAME} | Index: {index_text} | {path.name} | "
        f"Source: {source_width} x {source_height} | {mode_text}"
    )


def set_window_title(title):
    """Set the title where supported by the installed OpenCV GUI backend."""
    try:
        cv2.setWindowTitle(WINDOW_NAME, title)
    except (cv2.error, AttributeError):
        pass


def key_is(key, character):
    """Match either case of a character key."""
    return key in (ord(character.lower()), ord(character.upper()))


# Common extended-key values returned by cv2.waitKeyEx on Windows and Linux.
LEFT_KEYS = {2424832, 65361}
RIGHT_KEYS = {2555904, 65363}
UP_KEYS = {2490368, 65362}
DOWN_KEYS = {2621440, 65364}


# -----------------------------------------------------------------------------
# Main monitor
# -----------------------------------------------------------------------------

def run_monitor():
    print("--- AZ31 DIC LIVE MONITOR ---")
    print(f"Monitoring: {IMAGE_DIRECTORY}")
    print(f"Expected source resolution: {EXPECTED_IMAGE_WIDTH} x {EXPECTED_IMAGE_HEIGHT}")

    print("\n--- CONTROLS ---")
    print("  q or ESC    : Close monitor")
    print("  z           : Toggle whole frame / 1:1 focus zoom")
    print("  Arrow keys  : Pan during 1:1 focus zoom")
    print("  W A S D     : Alternative pan controls")
    print("  c           : Re-centre the 1:1 focus view")
    print("  r           : Force the current image to reload")
    print()

    # WINDOW_AUTOSIZE keeps the GUI client area matched to the displayed array,
    # preventing accidental stretching to a different aspect ratio.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    current_path = None
    loaded_signature = None

    source_image = None
    focus_mode = False
    focus_x = None
    focus_y = None
    force_reload = False

    while True:
        candidate = find_latest_image(IMAGE_DIRECTORY)
        candidate_signature = (
            make_file_signature(candidate) if candidate is not None else None
        )

        if candidate_signature is not None and bmp_file_is_complete(candidate):
            needs_loading = candidate_signature != loaded_signature or force_reload

            if needs_loading:
                loaded = load_image(candidate)

                if loaded is not None:
                    previous_shape = source_image.shape[:2] if source_image is not None else None
                    source_image = loaded
                    current_path = candidate
                    loaded_signature = candidate_signature
                    force_reload = False

                    # Centre the first image. Retain the selected focus location
                    # for later images unless their dimensions change.
                    if focus_x is None or focus_y is None or previous_shape != source_image.shape[:2]:
                        focus_x, focus_y = geometric_centre(source_image)

                    height, width = source_image.shape[:2]
                    image_index = extract_image_index(current_path)

                    print(
                        f"Loaded: {current_path.name} | "
                        f"index={image_index} | "
                        f"{width} x {height} | "
                        f"dtype={source_image.dtype}"
                    )

                    if (
                        width != EXPECTED_IMAGE_WIDTH
                        or height != EXPECTED_IMAGE_HEIGHT
                    ):
                        print(
                            "Warning: image dimensions differ from the expected "
                            f"{EXPECTED_IMAGE_WIDTH} x "
                            f"{EXPECTED_IMAGE_HEIGHT}."
                        )

        if source_image is not None and current_path is not None:
            crop_bounds = None

            if focus_mode:
                (
                    display_image,
                    focus_x,
                    focus_y,
                    crop_bounds,
                ) = create_focus_view(source_image, focus_x, focus_y)
            else:
                display_image = create_whole_frame_view(source_image)

            title = create_window_title(
                current_path,
                source_image,
                focus_mode,
                crop_bounds,
            )
            set_window_title(title)
            cv2.imshow(WINDOW_NAME, display_image)

        key = cv2.waitKeyEx(10)

        if key in (27, ord("q"), ord("Q")):
            break

        if key_is(key, "z"):
            focus_mode = not focus_mode

            # Enter each focus session at the exact source-image centre.
            if focus_mode and source_image is not None:
                focus_x, focus_y = geometric_centre(source_image)

            print(
                "View:",
                "focus zoom, 1:1 pixels" if focus_mode else "whole frame",
            )

        elif key_is(key, "r"):
            force_reload = True

        elif focus_mode and source_image is not None:
            if key_is(key, "c"):
                focus_x, focus_y = geometric_centre(source_image)
                print(f"Focus view centred at ({focus_x}, {focus_y})")

            elif key in LEFT_KEYS or key_is(key, "a"):
                focus_x -= PAN_STEP

            elif key in RIGHT_KEYS or key_is(key, "d"):
                focus_x += PAN_STEP

            elif key in UP_KEYS or key_is(key, "w"):
                focus_y -= PAN_STEP

            elif key in DOWN_KEYS or key_is(key, "s"):
                focus_y += PAN_STEP

        time.sleep(POLL_INTERVAL_S)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_monitor()
'''

with open(monitor_filename, 'w', encoding="utf-8", newline="\n") as f:
    f.write(monitor_code)

# G. Launch the Focus Monitor in a separate process
import subprocess
import sys

print("Launching Focus Monitor in a new console window...")
subprocess.Popen(
    [sys.executable, monitor_filename],
    cwd=save_directory,
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)

# =============================================================================
# --- 3. EXECUTE CAMERA CAPTURE ---
# =============================================================================       

def setup_camera(cam: vmbpy.Camera):
    cam.get_feature_by_name('TriggerMode').set('Off')
    cam.get_feature_by_name('PixelFormat').set('Mono8')

def _safe_camera_method(cam, method_name):
    """Return camera-identification data without modifying camera state."""
    try:
        method = getattr(cam, method_name)
        value = method()
        return str(value) if value not in (None, "") else "N/A"
    except Exception:
        return "N/A"


def _read_camera_feature(cam, feature_name):
    """Read one GenICam feature and its unit without changing its value."""
    try:
        feature = cam.get_feature_by_name(feature_name)
        if hasattr(feature, "is_readable") and not feature.is_readable():
            return "N/A"
        value = feature.get()
        try:
            unit = feature.get_unit()
        except Exception:
            unit = ""
        return f"{value} {unit}".strip()
    except Exception:
        return "N/A"


def append_camera_report_to_specs(cam):
    """Append camera-reported, read-only acquisition metadata to _specs.txt."""
    width = _read_camera_feature(cam, "Width")
    height = _read_camera_feature(cam, "Height")
    block = f"""
-------------------- CAMERA-REPORTED SETTINGS --------------------
Camera Name:                        {_safe_camera_method(cam, 'get_name')}
Camera Model:                       {_safe_camera_method(cam, 'get_model')}
Camera Serial Number:               {_safe_camera_method(cam, 'get_serial')}
Camera ID:                          {_safe_camera_method(cam, 'get_id')}
Active Image Width (px):            {width}
Active Image Height (px):           {height}
Pixel Format:                       {_read_camera_feature(cam, 'PixelFormat')}
Exposure Time:                      {_read_camera_feature(cam, 'ExposureTime')}
Trigger Mode:                       {_read_camera_feature(cam, 'TriggerMode')}
Gain:                               {_read_camera_feature(cam, 'Gain')}
Black Level:                        {_read_camera_feature(cam, 'BlackLevel')}
Offset X:                           {_read_camera_feature(cam, 'OffsetX')}
Offset Y:                           {_read_camera_feature(cam, 'OffsetY')}
Binning Horizontal:                 {_read_camera_feature(cam, 'BinningHorizontal')}
Binning Vertical:                   {_read_camera_feature(cam, 'BinningVertical')}
"""
    with open(memo_filename, "a", encoding="utf-8", newline="\n") as f:
        f.write(block)


def format_temperature(value, include_unit=False):
    """Format a measured temperature, or N/A when no reading is available."""
    if value is None:
        return "N/A"
    text = f"{value:.2f}"
    return f"{text} deg C" if include_unit else text


def append_consistency_check_to_specs(csv_path):
    """Compare image and ledger indices without modifying either dataset."""
    image_indices = []
    image_scan_error = None
    try:
        for name in os.listdir(save_directory):
            if name.lower().startswith("image_") and name.lower().endswith(".bmp"):
                index_text = name[6:-4]
                if index_text.isdigit():
                    image_indices.append(int(index_text))
    except Exception as exc:
        image_scan_error = str(exc)

    ledger_indices = []
    ledger_read_error = None
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ledger_indices.append(int(row["Image_Number"]))
    except Exception as exc:
        ledger_read_error = str(exc)

    image_set = set(image_indices)
    ledger_set = set(ledger_indices)
    missing_images = sorted(ledger_set - image_set)
    unlogged_images = sorted(image_set - ledger_set)
    duplicate_ledger_indices = sorted(
        index for index in ledger_set if ledger_indices.count(index) > 1
    )
    highest_index = max(image_set | ledger_set) if (image_set or ledger_set) else -1
    expected_set = set(range(highest_index + 1))
    missing_sequence_indices = sorted(expected_set - (image_set & ledger_set))
    matched_indices = ledger_set & image_set

    consistency_pass = (
        image_scan_error is None
        and ledger_read_error is None
        and len(image_indices) > 0
        and len(ledger_indices) > 0
        and len(image_indices) == len(ledger_indices)
        and not missing_images
        and not unlogged_images
        and not duplicate_ledger_indices
        and not missing_sequence_indices
    )

    def describe(indices):
        if not indices:
            return "None"
        if len(indices) <= 20:
            return ", ".join(str(i) for i in indices)
        return ", ".join(str(i) for i in indices[:20]) + f" ... ({len(indices)} total)"

    block = f"""
-------------------- ACQUISITION CONSISTENCY CHECK --------------------
Status:                              {'PASS' if consistency_pass else 'FAIL'}
Master-Log Data Rows:                {len(ledger_indices)}
BMP Image Files:                     {len(image_indices)}
Matched Unique Image Indices:        {len(matched_indices)}
Ledger Indices Missing BMP Files:    {describe(missing_images)}
BMP Indices Missing Ledger Entries:  {describe(unlogged_images)}
Missing Sequential Indices:          {describe(missing_sequence_indices)}
Duplicate Ledger Indices:            {describe(duplicate_ledger_indices)}
Image Scan Error:                    {image_scan_error or 'None'}
Ledger Read Error:                   {ledger_read_error or 'None'}
"""
    try:
        with open(memo_filename, "a", encoding="utf-8", newline="\n") as f:
            f.write(block)
    except Exception:
        consistency_pass = False
    return consistency_pass


def run_three_phase_capture():
    # Setup background saver thread (disk I/O only)
    image_queue = queue.Queue()
    writer_failures = []

    def image_writer_worker():
        while True:
            item = image_queue.get()
            try:
                if item is None:
                    return
                filename, image_data = item
                if not cv2.imwrite(filename, image_data):
                    writer_failures.append(
                        f"cv2.imwrite returned False for {filename}"
                    )
            except Exception as exc:
                filename = item[0] if item is not None else "unknown"
                writer_failures.append(f"{filename}: {exc}")
            finally:
                image_queue.task_done()

    writer_thread = threading.Thread(target=image_writer_worker, daemon=True)
    writer_thread.start()

    # Initialise The Master Ledger in the 'results' directory
    csv_path = os.path.join(results_directory, "master_log.csv")
    with open(csv_path, 'w', encoding='utf-8', newline="\n") as f:
        f.write("Image_Number,True_Elapsed_Time_s,Phase,Temperature_C\n")

# temperature variable to prevent over-polling
    global_temp = [None]
    temp_param_found = [None]

    def update_temperature(cam):
        if temp_param_found[0] == "UNSUPPORTED": return 
        if temp_param_found[0] is not None:
            try: global_temp[0] = cam.get_feature_by_name(temp_param_found[0]).get()
            except Exception: pass
            return

        # GenICam/Vimba standard temperature parameters
        params_to_try = ['DeviceTemperature', 'SensorTemperature', 'MainboardTemperature', 'Temperature']
        for param in params_to_try:
            try:
                global_temp[0] = cam.get_feature_by_name(param).get()
                temp_param_found[0] = param
                print(f"\n[SYSTEM] Thermal sensor locked using parameter: '{param}'")
                return
            except Exception: continue
        
        print("\n[WARNING] Temperature polling failed. Disabling sensor polling.")
        temp_param_found[0] = "UNSUPPORTED"

    def get_frame_without_timeout_termination(cam, timeout_ms=500):
        """Return None after a timeout, allowing the next acquisition attempt."""
        try:
            return cam.get_frame(timeout_ms=timeout_ms)
        except Exception as exc:
            exception_name = type(exc).__name__.lower()
            exception_text = str(exc).lower()
            if "timeout" in exception_name or "timeout" in exception_text or "timed out" in exception_text:
                return None
            raise

    with vmbpy.VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("ERROR: No cameras found. Check USB connection.")
            return

        with cams[0] as cam:
            print(f"Connected to {cam.get_name()}")
            print(f"Saving all data to: \n{save_directory}")
            setup_camera(cam)
            append_camera_report_to_specs(cam)
            
            print("\n" + "="*50)
            print("--- ACTIVE PREHEAT INITIATED ---")
            print("="*50)
            
            preheat_duration = preheat_duration_seconds
            preheat_start = time.monotonic()
            target_shot_time = preheat_start
            frame_count = 0
            last_temp_update_time = time.monotonic()
            
            # --- PREHEAT PHASE ---
            while (time.monotonic() - preheat_start) < preheat_duration:
                # The anti-drift logic
                time_left = target_shot_time - time.monotonic()
                if time_left > 0:
                    time.sleep(time_left)
                    
                frame = get_frame_without_timeout_termination(cam, timeout_ms=500)
                target_shot_time += start_interval
                if frame is None:
                    continue
                
                if (time.monotonic() - last_temp_update_time) >= temperature_poll_interval_seconds:
                    update_temperature(cam)
                    print(f"Preheating ({start_interval}s target)... Temp: {format_temperature(global_temp[0], include_unit=True)}")
                    last_temp_update_time = time.monotonic()
                frame_count += 1

            print(f"\n[READY] {preheat_duration / 60:.1f}-minute preheat complete.")
            print("Continuous frame acquisition is maintaining the thermal conditioning state.")
            
            user_ready = [False]
            def wait_for_enter():
                input("\n>>> PRESS ENTER WHEN READY TO START CONSTANT PHASE <<<\n")
                user_ready[0] = True
            threading.Thread(target=wait_for_enter, daemon=True).start()
            print("Continuous frame acquisition will continue until Enter is pressed.")
            print(
                "Temperature will be reported every "
                f"{temperature_poll_interval_seconds:g} seconds until Enter is pressed."
            )

            target_shot_time = time.monotonic()
            # --- HOLDING PATTERN ---
            while not user_ready[0]:
                time_left = target_shot_time - time.monotonic()
                if time_left > 0:
                    time.sleep(time_left)
                    
                frame = get_frame_without_timeout_termination(cam, timeout_ms=500)
                target_shot_time += start_interval
                if frame is None:
                    continue
                
                now = time.monotonic()
                if (now - last_temp_update_time) >= temperature_poll_interval_seconds:
                    update_temperature(cam)
                    holding_elapsed = int(now - preheat_start)
                    print(
                        "Thermal holding pattern "
                        f"(total thermalisation: {holding_elapsed}s) | "
                        "Temperature: "
                        f"{format_temperature(global_temp[0], include_unit=True)}"
                    )
                    last_temp_update_time = now
                frame_count += 1

            reference_temp = global_temp[0]

            print("\n" + "="*50)
            print(
                "EXPERIMENT STARTED. Camera temperature: "
                f"{format_temperature(reference_temp, include_unit=True)}"
            )
            print("="*50)

            # --- MAIN EXPERIMENT ---
            current_interval = start_interval
            image_count = 0 
            stopwatch_start = time.monotonic()
            target_shot_time = stopwatch_start
            next_save_time = 0.0  # The mathematical schedule for when to keep an image
            current_phase = "Constant"
            last_temp_update_time = stopwatch_start
            
            try:
                while True:
                    # Anti-drift logic
                    time_left = target_shot_time - time.monotonic()
                    if time_left > 0:
                        time.sleep(time_left)
                    
                    frame = get_frame_without_timeout_termination(cam, timeout_ms=500)
                    target_shot_time += start_interval
                    if frame is None:
                        continue
                    # Timestamp immediately after successful frame retrieval and
                    # before temperature access or image conversion.
                    frame_retrieval_time = time.monotonic()
                    elapsed_time = frame_retrieval_time - stopwatch_start

                    # Phase Determination (Just a label check every frame)
                    if elapsed_time < constant_duration_seconds:
                        current_phase = "Constant"
                    elif elapsed_time < (constant_duration_seconds + phase1_duration_seconds):
                        current_phase = "Phase 1"
                    else:
                        current_phase = "Phase 2"

                    # Poll temperature by monotonic time after frame timestamping.
                    # The low-frequency query cannot alter the timestamp already
                    # assigned to this frame; anti-drift handles any brief delay.
                    if (frame_retrieval_time - last_temp_update_time) >= temperature_poll_interval_seconds:
                        update_temperature(cam)
                        last_temp_update_time = frame_retrieval_time

                    # DYNAMIC SOFTWARE LOGIC: Only save if we cross the scheduled threshold
                    if elapsed_time >= next_save_time:
                        cv_image = frame.as_opencv_image().copy() 
                        
                        filename = os.path.join(save_directory, f"image_{image_count:05d}.bmp")
                        image_queue.put((filename, cv_image))
                        
                        # Write to Master Ledger
                        with open(csv_path, 'a', encoding='utf-8', newline="\n") as f:
                            f.write(f"{image_count},{elapsed_time:.3f},{current_phase},{format_temperature(global_temp[0])}\n")
                        
                        print(f"[{current_phase}] SAVED image_{image_count:05d}.bmp | Next Target Interval: {current_interval:.5f}s | True Time: {elapsed_time:.3f}s | Temp: {format_temperature(global_temp[0], include_unit=True)}")
                        
                        image_count += 1
                        next_save_time += current_interval # Schedule the next save based on current math
                        
                        # MATH UPDATE: Only expand the geometric multiplier exactly when an image is saved
                        if current_phase == "Phase 1":
                            current_interval = min(current_interval * phase1_expansion, stable_interval)
                        elif current_phase == "Phase 2":
                            current_interval = min(current_interval * phase2_expansion, stable_interval)

                    frame_count += 1
                    
            except KeyboardInterrupt:
                print(f"\nTest stopped manually. Saved {image_count} images.")
            finally:
                # Preserve FIFO ordering and the existing image-number/ledger logic.
                # The sentinel is queued only after acquisition has stopped, so every
                # earlier queued image is processed before the writer terminates.
                image_queue.put(None)
                writer_thread.join()
                if writer_failures:
                    print("[WARNING] One or more image writes failed:")
                    for failure in writer_failures[:20]:
                        print(f"  {failure}")
                consistency_pass = append_consistency_check_to_specs(csv_path)
                if writer_failures:
                    consistency_pass = False
                    with open(memo_filename, "a", encoding="utf-8", newline="\n") as f:
                        f.write(
                            "Writer Failures:                     "
                            f"{len(writer_failures)}\n"
                        )
                print(
                    "Acquisition consistency: "
                    f"{'PASS' if consistency_pass else 'FAIL'}"
                )
                print(f"Master Ledger: {csv_path}")

if __name__ == '__main__':
    try:
        run_three_phase_capture()
    except Exception as e:
        print("\n" + "="*50 + f"\nFATAL ERROR: {e}\n" + "="*50 + "\n")
    input("Press ENTER to close window...")
