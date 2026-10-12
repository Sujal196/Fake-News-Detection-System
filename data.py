import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import pymongo
    from pymongo import MongoClient
except ImportError:
    pymongo = None
    MongoClient = None

print("=" * 65)
print("     Fake News Detection System - MongoDB Atlas Data Viewer")
print("=" * 65)

mongo_uri = os.environ.get(
    'MONGODB_URI',
    'mongodb+srv://sujalsahu196_db_user:<db_password>@drishtix.dzxbcvj.mongodb.net/truthscan?retryWrites=true&w=majority&appName=DrishtiX'
)

if MongoClient is None:
    print("\n[ERR] pymongo is not installed. Please run: pip install pymongo dnspython")
elif '<db_password>' in mongo_uri:
    print("\n[INFO] Placeholder <db_password> detected in .env / MONGODB_URI.")
    print("       To connect to your cluster, replace <db_password> with your actual")
    print("       database password in the .env file.")
else:
    try:
        print("\n[CONNECTING] Connecting to MongoDB Atlas cluster...")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db_name = os.environ.get('MONGODB_DB_NAME', 'truthscan')
        coll_name = os.environ.get('MONGODB_COLLECTION', 'history')
        db = client[db_name]
        collection = db[coll_name]

        count = collection.count_documents({})
        print(f"[OK] Successfully connected to MongoDB Atlas! (Database: '{db_name}', Collection: '{coll_name}')")
        print(f"     Total predictions stored: {count}\n")

        cursor = collection.find().sort('_id', -1).limit(25)
        for i, doc in enumerate(cursor, 1):
            text = (doc.get('input_text') or doc.get('text', ''))[:60]
            pred = doc.get('prediction', 'Unknown')
            conf = float(doc.get('confidence', 0.0))
            ts = doc.get('timestamp', 'N/A')
            sess = doc.get('session_id', '')[:12]
            print(f"  [{i:02d}] {ts} | {pred:10s} | Conf: {conf:.2%} | Sess: {sess}... | Text: {text}...")

        if count == 0:
            print("  (No predictions stored in MongoDB Atlas yet. Submit an article in TruthScan to test!)")
    except Exception as me:
        print(f"\n[ERR] MongoDB Atlas connection error: {me}")

print("\n" + "=" * 65)
print("  Database Engine: MongoDB Atlas (Cloud NoSQL)")
print("  Vercel Persistence: Fully supported with real-time updates")
print("=" * 65)