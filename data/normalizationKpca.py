import pandas as pd
from sklearn import preprocessing
import numpy as np

df=pd.read_csv('../archive/alkanes_core_with_smiles_final_with_graph.csv')

features = ["Density", "degree_entropy", "compression_ratio"]
df_numeric = df.select_dtypes(exclude=["string", "object"])

# Save back to CSV if needed
df_numeric.to_csv("numeric_only.csv", index=False)

print("Dropped string columns. New shape:", df_numeric.shape)