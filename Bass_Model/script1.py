import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from helper_functions import bass_model

# 1. LOAD DATA using relative path
# Data is stored in the 'data' subfolder
df = pd.read_excel('data/film_cameras.xlsx')

# Clean data: Convert n/a to NaN and drop empty rows
df['Film Cameras Total'] = pd.to_numeric(df['Film Cameras Total'], errors='coerce')
df = df.dropna(subset=['Film Cameras Total'])

years = df['Year'].values.astype(int)
sales = df['Film Cameras Total'].values
t = np.arange(1, len(sales) + 1)

# 2. PARAMETER ESTIMATION
# Initial guesses for the optimizer
initial_guess = [0.01, 0.4, 800000] 

params, _ = curve_fit(bass_model, t, sales, p0=initial_guess)
p_hist, q_hist, M_hist = params

print(f"Historical Parameters Estimated:")
print(f"p: {p_hist:.6f}, q: {q_hist:.6f}, M: {M_hist:.0f}")
