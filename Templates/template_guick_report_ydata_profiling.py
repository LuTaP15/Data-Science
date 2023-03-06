# Import libraries
import pandas as pd
from ydata_profiling import ProfileReport

# Load the data
df = pd.read_csv("./../data/autos_prepared.csv")

# Produce and save the profiling report
profile = ProfileReport(df, title="Car Report")
profile.to_file("./../data/report.html")