# Libaries
import sweetviz as sv
import pandas as  pd

# Load dataset
df = pd.read_csv('.././data/iris_data.csv')

# Create Analysis report
iris_report = sv.analyze(df)

# Display report
iris_report.show_html('Iris.html')

# Compare two datasets
df1 = sv.compare(df[100:], df[:100])
df1.show_html('Compare.html')