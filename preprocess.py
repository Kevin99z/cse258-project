import pandas as pd

# Read in the raw data
data = pd.read_csv('RAW_interactions.csv')
# use only user_id,recipe_id, rating
data = data[['user_id', 'recipe_id', 'rating']]
# map user_id and recipe_id to integer starting from 0
data['user_id'] = data['user_id'].astype('category').cat.codes
data['recipe_id'] = data['recipe_id'].astype('category').cat.codes
# rename the columns
data.columns = ['user', 'item', 'rating']
# randomly split the data into train, valid and test
train = data.sample(frac=0.8, random_state=200)
valid = data.drop(train.index).sample(frac=0.5, random_state=200)
test = data.drop(train.index).drop(valid.index)
# save the data
train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)
test.to_csv('test.csv', index=False)

