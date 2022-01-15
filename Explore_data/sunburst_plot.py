"""
Create a sunburst chart plot
"""

# Libaries
import plotly.graph_objects as go
import pandas as pd

# Create data
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')
df = pd.read_csv('.././data/machine_learning.csv')

# Sunburst chart plot
fig = go.Figure(go.Sunburst(
        ids=df.ids,
        labels=df.labels,
        parents=df.parents,
        insidetextorientation='radial'))
fig.show()