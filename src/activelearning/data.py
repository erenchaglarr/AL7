from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder

# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
#print(mushroom.metadata) 
  
# variable information 
#print(mushroom.variables) 

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
print(enc.categories_)

