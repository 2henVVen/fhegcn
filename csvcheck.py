import pandas as pd
from pathlib import Path

data_dir = Path("./BRCA")

files = [
    "1_tr.csv",
    "1_te.csv",
    "1_featname.csv",
    "labels_tr.csv",
    "labels_te.csv",
]

for f in files:
    path = data_dir / f
    print(f"\n===== {f} =====")
    
    df = pd.read_csv(path, header=None)
    
    print("shape:", df.shape)
    print(df.head())