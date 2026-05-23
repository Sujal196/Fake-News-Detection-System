import sqlite3
from numpy import select
import pandas as pd

df=pd.read_sql_query("select * from history", sqlite3.connect('predictions.db'))
print(df)