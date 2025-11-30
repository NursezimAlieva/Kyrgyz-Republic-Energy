import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Import dataframes ---
try:
    allEnergy = pd.read_excel('Production of electricity.xlsx')
    gas = pd.read_excel('Natural gas production volume (stat kg).xlsx')
    coal = pd.read_excel('Volume of coal and lignite production (stat kg).xlsx')
    energyUse = pd.read_excel('Energy use in Kyrgyz Republic(WBOD).xlsx')
except FileNotFoundError as e:
    print(f"Error: Unable to find file. Make sure all Excel files are in the working directory. Error: {e}")
    exit()

# --- 2. Function for transforming and cleaning data ---
def transform_and_clean_robust(df, value_col_name, row_index=0):
    """Converts data from wide to long format, stripping 'Year' and 'Value'."""
    data_series = df.iloc[row_index].copy()

    # We try to take columns starting from the 4th (index 3), as in the previous structure
    year_columns = data_series.index[3:]

    df_long = data_series[year_columns].to_frame(name=value_col_name)
    df_long.index.name = 'Year'
    df_long = df_long.reset_index()

    df_long['Year'] = df_long['Year'].astype(str).str.replace(r'[^\d]+', '', regex=True).str.strip()
    df_long = df_long[df_long['Year'] != '']
    df_long['Year'] = df_long['Year'].astype(int)

    # Value conversion with handling of dashes/non-numbers
    df_long[value_col_name] = pd.to_numeric(df_long[value_col_name], errors='coerce')

    df_long = df_long.dropna(subset=[value_col_name, 'Year'])

    return df_long

# Applying transformation
df_energy = transform_and_clean_robust(allEnergy, 'Consumption_MWh', row_index=0)
df_gas = transform_and_clean_robust(gas, 'Production_Gas', row_index=0)
df_coal = transform_and_clean_robust(coal, 'Production_Coal', row_index=0)
df_use = transform_and_clean_robust(energyUse, 'Usage_per_capita', row_index=0)

# --- 3. Union and CORRECT IMPUTATION ---

df_combined = df_energy.copy()
df_combined = pd.merge(df_combined, df_gas, on='Year', how='left') # left merge on df_energy
df_combined = pd.merge(df_combined, df_coal, on='Year', how='left')
df_combined = pd.merge(df_combined, df_use, on='Year', how='left')

# We leave only those years for which there is data on the target variable
df_final = df_combined.dropna(subset=['Consumption_MWh'])

# Sorting by year for correct time series imputation
df_final = df_final.sort_values(by='Year')

# Replacing aggressive dropna() with robust imputation
feature_cols = ['Production_Gas', 'Production_Coal', 'Usage_per_capita']

for col in feature_cols:
    # 1. Forward Fill: Filling in the blanks with the last known value
    df_final[col] = df_final[col].fillna(method='ffill')

    # 2. Backward Fill: Filling in the gaps at the beginning of the row with the first known value
    df_final[col] = df_final[col].fillna(method='bfill')

    # 3. Final Fallback: If the column is completely empty (which is unlikely), fill it with zero.
    mean_val = df_final[col].mean()
    fallback_value = mean_val if pd.notna(mean_val) else 0
    df_final[col] = df_final[col].fillna(fallback_value)


print(f"Ready data for training: {len(df_final)} rows (with {df_final['Year'].min()} on {df_final['Year'].max()}).")


# --- 4. Modeling and Forecasting ---
if len(df_final) < 2:
    print("\nERROR: Not enough data for training after cleaning.")
else:
    features = ['Year', 'Production_Gas', 'Production_Coal', 'Usage_per_capita']
    target = 'Consumption_MWh'

    # X is now NaN-FREE
    X = df_final[features]
    y = df_final[target]

    # Model training
    model = LinearRegression()
    model.fit(X, y)

    # 5. Generating data for forecasting (5 years)
    last_year = df_final['Year'].max()
    future_years = pd.DataFrame({
        'Year': range(last_year + 1, last_year + 6)
    })

    # We use the average of the last 5 years of observed data for future signs
    for col in features[1:]:
        mean_val = df_final[col].tail(5).mean()
        future_years[col] = mean_val

    # Making a forecast
    forecast_X = future_years[features]
    forecast_y = model.predict(forecast_X)

    # Forming the final result
    df_forecast = future_years[['Year']].copy()
    df_forecast['Forecast_Consumption_MWh'] = forecast_y.round(1)

    print("\n=============================================")
    print("5-year electricity consumption forecast:")
    print("=============================================")
    print(df_forecast)

    # 6. Visualization of results
    plt.figure(figsize=(12, 7))
    plt.plot(df_final['Year'], df_final['Consumption_MWh'], label='Actual consumption', marker='o', color='blue')
    plt.plot(df_forecast['Year'], df_forecast['Forecast_Consumption_MWh'], label='Forecast (5 years)', marker='s', linestyle='--', color='red')
    plt.title('Electricity consumption forecast')
    plt.xlabel('Year')
    plt.ylabel('Consumption(MWh)')
    plt.legend()
    plt.grid(True)
    plt.show()