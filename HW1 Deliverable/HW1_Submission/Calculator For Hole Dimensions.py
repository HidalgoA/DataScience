import numpy as np
import pandas as pd 

#Dimensions of Beam in mm
length = 100 
width = 20
thickness = 1

#Beam Volume mm^2
Volume = length * width * thickness 

#Required Volume Fractions starting from 0.1 to 0.7 with 0.05 increments
VolumeFractions = np.arange(0.1, 0.75, 0.05)
#VolumeFractions =  np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])


#Total Calculated Volume of holes 
VolumeOfHoles = VolumeFractions * Volume

#Randomly creates array of number of holes ranging between 1 and 4
#NumHoles = np.random.randint(4, 11, size = len(VolumeFractions)) for finding good values at first
#NumHoles =  [3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5] for testing
NumHoles =  [8, 4, 8, 7, 4, 4, 9, 4, 5, 11, 9, 8, 10] 
#            G, G, G, G, G, G, G, G, G, M, M, M , M
"""
Total Hole Volume = number of holes * volume of single hole

Where the Total Hole Volume is calculated from 0.1, 0.15, 0.20, etc times the volume of the beam
Where number of holes is randomly picked
Where volume of single hole is pi * r^2 * thickness 

Solve the equation for r to determine the required radius in mm
"""

radius = []
for HoleVolume, HoleCount in zip(VolumeOfHoles, NumHoles):

    r = np.sqrt(HoleVolume / (np.pi * thickness * HoleCount))

    radius.append(r) 

results = list(zip(VolumeFractions, VolumeOfHoles, NumHoles, radius))

df = pd.DataFrame(results, columns = ["Volume Fraction", "Volume Removed", "Number of Holes", "Hole Radius"])
print(df)




"""
Now determine varience of Z
"""
#E[Z] (mean of Z)
E_Z = np.mean(Z) 

#E[X^2] (mean of squared Z values)
E_Z2 = np.mean(Z**2)

#variance using the formula E[Z^2] - (E[Z])^2
var_Z_formula = E_Z2 - (E_Z ** 2)
print(f"Variance of Z from manual calc: {var_Z_formula:.3f}")

#Using np.var function to compare
var_z = np.var(Z, ddof =0)
print(f"Variance of Z from np.var: {var_z:.3f}")