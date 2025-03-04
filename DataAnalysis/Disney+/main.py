#%%
import numpy as np
import pandas as pd

data = pd.read_csv("disney_movies_total_gross.csv", parse_dates=['release_date'])
print(data)
#%%
data.sort_values(["inflation_adjusted_gross"], ascending=False)
print(data.head(10))
#%%

data['release_year'] = data['release_date'].dt.year
group = data.groupby(['genre', 'release_year'], as_index=True)['genre', 'release_year','total_gross', 'inflation_adjusted_gross']
print(group.head())

# %%
