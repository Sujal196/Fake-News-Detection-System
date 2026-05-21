import sqlite3
import pandas as pd

connection = sqlite3.connect('predictions.db')


df= pd.read_sql("select * from history;",connection)
    
print(df)