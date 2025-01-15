import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# DAAN PARAMATERS:
line_color = "orange" # KLEUR VAN DE LIJN
scatter_color = "blue" # KLEUR VAN DE SCATTER PUNTEN
errorbar_color = "gray" # KLEUR VAN DE ERROR BALKEN

# Gegeven data
dT = np.array([1.5, 2, 3.03, 4.23, 5.07])  # Temperatuurverschillen (in Kelvin)
dN = np.array([4.87, 8.03, 11.9, 14.53, 17.33]) # Aantal franjes
dT_fout = np.array([0.36, 0.10, 0.21, 0.38, 0.23]) # Stdev in Tempratuur
dN_fout = np.array([0.35, 0.75, 0.46, 0.67, 1.05]) # Stdev in aantal franjes

lambda_ = 532e-9       # Golflengte in meters
l0 = 0.1               # Oorspronkelijke lengte in meters

# dL berekenen
dL = dN * lambda_ / 2

# Lineaire functie voor curve fitting
def linear_fit(x, a):
    return a * x

# Curve fit uitvoeren
popt, pcov = curve_fit(linear_fit, dT, dL)
a_fit = popt[0]  # Richtingscoëfficiënt a

# Bereken de standaardfout
a_error = np.sqrt(np.diag(pcov))[0]

# Bereken lineaire uitzettings-coefficient
alpha = a_fit / l0 

# Resultaten printen
print(f"Lineaire uitzettingscoëfficiënt α: {alpha:.2e} 1/K")
print(f"Standaardfout van a: {a_error:.2e} m/(m·K)")

# Plot de data met grijze spreidingsbalkjes
plt.errorbar(dT, dL, xerr=dT_fout, yerr=dN_fout * lambda_ / 2, fmt='o', label="Gemeten data", color=scatter_color, ecolor=errorbar_color, capsize=5)

# Plot de lineaire fit
plt.plot(dT, linear_fit(dT, *popt), label=f"Alpha={alpha:.2e}", color=line_color)

plt.xlabel("ΔT (K)")
plt.ylabel("ΔL (m)")
plt.legend()
plt.grid()
plt.title("Lineaire fit voor uitzettingscoëfficiënt")


plt.savefig("linearcoefficient.png", dpi=800)

plt.show()