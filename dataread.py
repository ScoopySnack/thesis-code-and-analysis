import pandas as pd

df = pd.read_excel(r"C:\Users\atzyn\Documents\MASTER\THESIS\data.xlsx")

# Set the proper column names (from the second row)
df.columns = df.iloc[0]
df = df[1:]

# Drop columns that are entirely empty
df = df.dropna(axis=1, how='all')

# Show current columns
print("Columns before renaming:", df.columns)

# Then rename accordingly (once you're sure)
df.columns = ["Nr", "Name", "Nr of Carbons", "Density"]
print("Columns after renaming:", df.columns)

# Convert numeric columns to proper types
df["Nr"] = pd.to_numeric(df["Nr"])
df["Nr of Carbons"] = pd.to_numeric(df["Nr of Carbons"])
df["Density"] = pd.to_numeric(df["Density"])

print(df.head())
