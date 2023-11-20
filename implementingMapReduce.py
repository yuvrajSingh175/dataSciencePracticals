import pandas as pd

# Sample data
dateNtemp = [(1012023, 13.2), (2012023, 30.2), (3012023, 14.8), (4012023, 31)]

# Create a DataFrame
df = pd.DataFrame(dateNtemp, columns=["date", "temperature"])

# Define a function to categorize as hot or cold
def categorize_temperature(temp):
    return "hot" if temp >= 30 else "cold"

# Apply the function to create a new column
df["weather_status"] = df["temperature"].apply(categorize_temperature)

# Print the result
print(df)
