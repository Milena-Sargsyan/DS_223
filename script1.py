# 1. LOAD AND PREPARE DATA
# Using the full dataset (1977-2007) to capture the full lifecycle
df = pd.read_excel('film_cameras.xlsx')

# 1. Convert 'n/a' to actual NaN and drop those rows
df['Film Cameras Total'] = pd.to_numeric(df['Film Cameras Total'], errors='coerce')
df = df.dropna(subset=['Film Cameras Total'])

# 2. Extract cleaned data
years = df['Year'].values.astype(int)
sales = df['Film Cameras Total'].values
t = np.arange(1, len(sales) + 1)

print(f"Data cleaned. Analyzing years {years.min()} to {years.max()}.")


# 3. ESTIMATE PARAMETERS (Step 4)
# Initial guesses: p (innovation), q (imitation), M (market potential in 1000s)
# Based on the existing data in the excel file, total cumulative sales by 2007 are ~ 734000 units.
initial_guess = [0.01, 0.4, 800000] 

params, _ = curve_fit(bass_model, t, sales, p0=initial_guess)
p_hist, q_hist, M_hist = params

print(f"Historical Parameters (Look-alike):\np: {p_hist:.6f}\nq: {q_hist:.6f}\nM: {M_hist:.0f} (units in 1000s)")
