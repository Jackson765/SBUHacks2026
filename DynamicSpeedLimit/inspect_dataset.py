import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\iamhi\OneDrive\Documents\Hackathon Speed limit\DynamicSpeedLimit\model_b\data\METR-LA.csv")
df = pd.read_csv(p)
print('shape=', df.shape)
print(df.head().to_string(index=False))
print('\ncolumns=', list(df.columns[:20]))
print('\ndtypes=\n', df.dtypes.head(20))
