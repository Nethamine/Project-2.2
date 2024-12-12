import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

file_path: str = "/Users/nathanphang/metingen.csv" 

df = pd.read_csv(file_path, sep=';', header=0)  # Gebruik ',' als scheidingsteken en laat de eerste rij als header.

df.columns = ["hoek", "N1", "N2", "N3"]

df["hoek"] = pd.to_numeric(df["hoek"], errors='coerce')
df["N1"] = pd.to_numeric(df["N1"], errors='coerce')
df["N2"] = pd.to_numeric(df["N2"], errors='coerce')
df["N3"] = pd.to_numeric(df["N3"], errors='coerce')

df = df.dropna()

df["Ngem"] = (df["N1"] + df["N2"] + df["N3"]) / 3
df["stdv"] = (((df["N1"] - df["Ngem"])**2 + (df["N2"] - df["Ngem"])**2 + (df["N3"] - df["Ngem"])**2) / 2)**(1/2)
df["radialen"] = df["hoek"] * (2 * np.pi) / 360

def model(x, n):
    n_lucht = 1.0029
    d: float = 2e-3
    lambda_ = 532e-9  # Verander 'lambda' naar 'lambda_'

    # Berekening van theta
    theta = np.arcsin(np.sin(x) * n_lucht / n)

    # Modelvergelijking
    N = (2 * d / lambda_) * (n / np.cos(theta) + np.tan(x) * np.sin(x) * n_lucht - np.tan(theta) * np.sin(x) * n_lucht - (n - n_lucht) - n_lucht / np.cos(x))
    return N

# Gebruik de correcte parameter 'p0' in plaats van 'pO'
popt, pcov = curve_fit(
    model,
    df["radialen"],
    df["Ngem"],
    p0=[1.5],  # Startschatting voor n
    sigma=df["stdv"],
    absolute_sigma=True
)

print("Optimale parameters:", popt)
print("Covariantiematrix:", pcov)

# Genereer een reeks van x-waarden (hoek in radialen)
x_vals = np.linspace(min(df["radialen"]), max(df["radialen"]), 100)

# Bereken de bijbehorende y-waarden voor de modelfunctie, gebruikmakend van de optimale parameter
n_optimal = popt[0]  # De optimale parameter die door curve_fit wordt gegeven
y_vals = model(x_vals, n_optimal)

# Plot de gemeten data en het model
plt.errorbar(df["radialen"], df["Ngem"], yerr=df["stdv"], fmt='o', label="Gemeten data", markersize=5)
plt.plot(x_vals, y_vals, label=f"Model fit (n = {n_optimal:.3f})", color='red')
plt.xlabel('Hoek (radialen)')
plt.ylabel('N')
plt.legend()
plt.title('Model Fit vs Gemeten Data')
plt.grid(True)
plt.show()
