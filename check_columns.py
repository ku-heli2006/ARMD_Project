import pandas as pd
df = pd.read_csv('data/microbiology_cultures_priorprocedures.csv', low_memory=False)
print(df.columns.tolist())