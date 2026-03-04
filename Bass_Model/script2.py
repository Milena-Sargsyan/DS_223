import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import bass_model

# 1. SETUP PARAMETERS (derived from script1 and Fermi logic)
p, q = 0.010203, 0.162198
M_pentax = 5000  
t_future = np.arange(1, 21)

# 2. RUN FORECAST
pentax_forecast = bass_model(t_future, p, q, M_pentax)
cum_pentax = np.cumsum(pentax_forecast)

# 3. VISUALIZATION & SAVING
plt.figure(figsize=(10, 6))
plt.bar(t_future, pentax_forecast, color='green', alpha=0.7, label='Annual Sales')
plt.plot(t_future, cum_pentax, color='blue', marker='o', label='Cumulative Adopters')
plt.title('Pentax 17 Predicted Diffusion Path')
plt.xlabel('Years from Launch')
plt.ylabel('Units (1,000s)')
plt.legend()

# Save image to the img/ folder using a relative path
plt.savefig('img/image2.png')
plt.show()

# 4. EXPORT DATA
forecast_table = pd.DataFrame({
    'Year': t_future,
    'Annual_Adopters': pentax_forecast.round(2),
    'Total_Adopters': cum_pentax.round(2)
})

# Save CSV to the data/ folder using a relative path
forecast_table.to_csv('data/pentax_predictions.csv', index=False)
print("Forecast saved to data/pentax_predictions.csv")
