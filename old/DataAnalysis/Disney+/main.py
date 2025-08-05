#%%
# Step 1: load the dataset
import numpy as np
import pandas as pd

data = pd.read_csv("disney_movies_total_gross.csv", parse_dates=['release_date'])
print(data)
#%%
# Step 2: top grossing films
data.sort_values(["inflation_adjusted_gross"], ascending=False)
print(data.head(10))
#%%
# Step 3: genre trends
data['release_year'] = data['release_date'].dt.year
group = data.groupby(['genre', 'release_year'], as_index=True)
#print(group.head())
genre_yearly = group[['total_gross', 'inflation_adjusted_gross']].sum().reset_index()
print(genre_yearly.head(10))

#%%
# Step 4: plotting the trends of genre over the years
import seaborn as sns
sns.relplot(genre_yearly, x='release_year', y='inflation_adjusted_gross', kind='line', hue='genre')
# %%
# Step 5: Prepare dummy variables for a linear regression model
genre_dummies = pd.get_dummies(data['genre'], drop_first=True)
print(genre_dummies.head())
# %%
# Step 6: Fit a linear regression model with genre_dummies and inflation_adjusted_gross
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(genre_dummies, data['inflation_adjusted_gross'])

action = regr.intercept_
adventure = regr.coef_[0]

print((action, adventure))
#%%
# Step 7: Set up an array of indices and initialize replicated arrays for bootstrap
import numpy as np

inds = np.arange(0, len(data['genre']))
size = 500
bs_action_reps = np.empty(size)
bs_adventure_reps = np.empty(size)

print(size, bs_action_reps.shape, bs_adventure_reps.shape)
#%%
# Step 8: Perform paired bootstrap for linear regression
for i in range(size):
    bs_inds = np.random.choice(inds, size=len(inds))
    
    bs_genre = data['genre'].iloc[bs_inds]
    bs_gross = data['inflation_adjusted_gross'].iloc[bs_inds]
    
    bs_dummies = pd.get_dummies(bs_genre, drop_first=True)
    
    regr = LinearRegression()
    regr.fit(bs_dummies, bs_gross)
    
    bs_action_reps[i] = regr.intercept_
    bs_adventure_reps[i] = regr.coef_[0]  
#%%
# Step 9: Calculate 95% confidence intervals
confidence_interval_action = np.percentile(bs_action_reps, [2.5, 97.5])
confidence_interval_adventure = np.percentile(bs_adventure_reps, [2.5, 97.5])

print(confidence_interval_action)
print(confidence_interval_adventure)
#%%

# Step 10: Should Disney make more action and adventure movies?
# If both confidence intervals exclude zero, we can conclude there's a significant relationship
more_action_adventure_movies = (confidence_interval_action[0] > 0 and 
                               confidence_interval_adventure[0] > 0)
print(more_action_adventure_movies)
# %%
