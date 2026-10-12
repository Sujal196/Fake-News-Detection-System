import os
import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

mongo_uri = os.environ.get(
    'MONGODB_URI',
    'mongodb+srv://sujalsahu196_db_user:<db_password>@drishtix.dzxbcvj.mongodb.net/truthscan?retryWrites=true&w=majority&appName=DrishtiX'
)

df = None

if MongoClient is None:
    print("pymongo is not installed.")
elif '<db_password>' in mongo_uri:
    print("Please set your database password in .env or MONGODB_URI.")
else:
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[os.environ.get('MONGODB_DB_NAME', 'truthscan')]
        collection = db[os.environ.get('MONGODB_COLLECTION', 'history')]
        records = list(collection.find({}, {'_id': 0}))
        if records:
            df = pd.DataFrame(records)
            print("Predictions DataFrame from MongoDB Atlas:")
            print(df)
        else:
            print("MongoDB Atlas collection is currently empty.")
    except Exception as e:
        print(f"MongoDB Atlas connection/read error: {e}")