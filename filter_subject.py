import pandas as pd
import sys

df = pd.read_csv("temperatures.csv")

print(df[df["subject_id"]==int(sys.argv[1])])
