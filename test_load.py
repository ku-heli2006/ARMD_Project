import pandas as pd
print("Starting...")
df = pd.read_csv('data/microbiology_cultures_cohort.csv', low_memory=False)
print("Done loading!")
print(df.shape)