from scipy.optimize import curve_fit
import numpy as np

d = 1e-3
l = 532e-9

generations = 5


def model_N(deg, n):
    N = (d / l) * (n / n-1) * deg**2    
    return N

a = np.linspace(0,10,20) # DEGREES
n = # DATA

sigma = 0

for i in range(5)
    popt, pcov = curve_fit(model_N, a, n,p0=[[0,2],[0,5]], sigma=sigma)
                           
    # bereken fout
    # sigma = abs(N_praktish - N_theoretisch)
    
print(popt,pcov)