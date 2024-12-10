import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

file: str = "/Users/nathanphang/metingen.csv" 

df = pd.read_csv(file, names=["hoek", "N1", "N2", "N3"])
 
df["Ngem"] = (df["N1"] + df["N2"] + df["N3"])/3
df["stdv"] = (((df["N1"] - df["Ngem"])**2 + (df["N2"] - df["Ngem"])**2 + (df["N3"] - df["Ngem"])**2)/ 2)**(1/2)


Nlijst = df["Ngem"]

df["radialen"] = df["hoek"] * (2*np.pi)/360

radialenlijst = df["radialen"].to_numpy()

def model(x, n):
    n_lucht = 1.0029
    d: float = 2e-3
    lambda: float = 532e-9

    theta = np.arcsin(np.sin(x)*n_lucht/n)


    N = 2*d/lambda*(n/np.cos(theta) + np.tan(x)*np.sin(x)*n_lucht - np.tan(theta)*np.sin(x)*n_lucht - (n-n_lucht) -n_lucht/np.cos(x))

return N 

popt, pcov = curve_fit(model, df["radialen"], df["Ngem"], pO=[1.5], sigma=df["stdv"], absolute_sigma=True)
