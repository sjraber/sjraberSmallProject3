import pandas as pd
from IPython.display import display


data = pd.read_csv("card_transdata.csv", header=0, index_col=None)

data = data.dropna()

cols = ["distance_from_home", "distance_from_last_transaction", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]

data.drop(cols,axis=1, inplace=True)


data.info()
display(data)

data.to_csv("clean2.csv", index=False)

