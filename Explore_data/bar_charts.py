"""
Exploring data analysis with bar charts
"""

import pandas as pd
import plotly.express as px
import random

# Create data
expense_data = {
    "Person": random.choices(["A", "B"], k=30),
    "Amount": random.sample(range(100, 200), 10) + random.sample(range(0, 99), 10) + random.sample(range(49, 499), 10),
    "Category": ["Groceries"] * 10 + ["Restaurant"] * 10 + ["Appliances"] * 10,
    "Date": pd.to_datetime(pd.date_range('2020-01-01','2020-10-01', freq='MS').tolist() * 3)
}
df = pd.DataFrame(data=expense_data)

df_grouped = df.groupby(by=[pd.Grouper(key="Date", freq="1M"), "Category"])["Amount"]
df_grouped = df_grouped.sum().reset_index()

# Plot bar charts
barmodes = ["stack", "group", "overlay", "relative"]
for barmode in barmodes:
    fig_bar = px.bar(df_grouped, x="Date", y="Amount", color="Category", barmode=barmode)
    fig_bar.show()

####################################################
# Bar chart with text inside
df_people = df.groupby(by=[pd.Grouper(key="Date", freq="1M"), "Person"])["Amount"]
df_people = df_people.sum().reset_index()

fig_people = px.bar(df_people, x="Date", y="Amount", color="Person", barmode="stack", text="Amount")
fig_people.show()

####################################################
# Horizontal bar chart
df_category = df.groupby(["Category", "Person"]).sum().reset_index()

fig_category = px.bar(df_category, x="Amount", y="Person", color="Category", text="Amount", orientation="h")
fig_category = fig_category.update_traces(insidetextanchor="middle", texttemplate="$%{text}")
fig_category = fig_category.update_xaxes(visible=False, showticklabels=False)
fig_category.show()