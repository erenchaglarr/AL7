from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
#print(mushroom.metadata) 
  
# variable information 
#print(mushroom.variables) 

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

one_hot_encoder = enc.fit_transform(X)

one_hot_df = pd.DataFrame(one_hot_encoder, columns=enc.get_feature_names_out(X.columns))

df_sklearn = pd.concat([one_hot_df, y], axis=1)

print(df_sklearn)
