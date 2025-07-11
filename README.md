# PVAT
Python VISAR Analysis Tool

# Dependencies
- matplotlib
- pandas
- numpy
- PIL
- scipy
  
# Tutorials  
## Align a Beam reference  
Use the below code to align a beam reference. We use the synthetically generated 20ns beam reference
```
from VISAR import RefImage
from InteractivePlots import BeamAligner

ref_file = "SyntheticData/20nsBeamReference.tif"
ref = RefImage(fname = ref_file, sweep_speed = 20, slit_size = 500)
aligner = BeamAligner(ref)
aligner.initialize_plot()
aligner.show_plot()
```
## Generate synthetic data  

```
from SyntheticData import SyntheticBeamCalibration
 
simulated = SyntheticBeamCalibration(sweep_speed = 20, slit_size = 500, time_points = 1000, space_points = 500)  
simulated.generate_background(500)  
simulated.generate_beam(3.5, 1, 200, max_loc = 450)  
synthetic_img = VISARImage(fname = None, data = simulated.data, sweep_speed = simulated.sweep_speed, slit_size = simulated.slit_size)  
fig = plt.subplots()  
ax = plt.subplot(1, 1, 1)  
ax.set_title("Simulated Image")  
synthetic_img.show_data(ax, minmax = (simulated.data.min(), simulated.data.max()))  
plt.show()  
```
