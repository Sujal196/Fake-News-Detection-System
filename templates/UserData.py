import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load local environment variables if present (.env)
load_dotenv()

mongo_uri = os.environ.get('MONGODB_URI')
connected_to_mongo = False
records = []

if mongo_uri:
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        # Test connection
        client.admin.command('ping')
        
        try:
            db = client.get_default_database()
            if db is None or db.name == 'admin':
                db = client['fake_news_db']
        except Exception:
            db = client['fake_news_db']
            
        collection = db['history']
        records = list(collection.find())
        connected_to_mongo = True
        print("Successfully fetched history from MongoDB Atlas.")
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Falling back to SQLite.")

if not connected_to_mongo:
    # SQLite Fallback
    try:
        import sqlite3
        # Look for predictions.db in the parent directory as templates/UserData.py is inside templates/
        db_path = '../predictions.db'
        if not os.path.exists(db_path):
            db_path = 'predictions.db' # Try local directory just in case
            
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
            conn.close()
            if not df.empty:
                print("Successfully fetched history from local SQLite.")
                print(df)
            else:
                print(f"No records found in SQLite {db_path}.")
        else:
            print("SQLite database predictions.db not found, and MongoDB Atlas connection failed.")
    except Exception as sqlite_err:
        print(f"Error querying local SQLite: {sqlite_err}")
else:
    if records:
        for r in records:
            if '_id' in r:
                r['_id'] = str(r['_id'])
        df = pd.DataFrame(records)
        print(df)
    else:
        print("No history records found in MongoDB.")