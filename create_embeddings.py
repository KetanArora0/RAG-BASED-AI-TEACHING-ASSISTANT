import joblib
from sentence_transformers import SentenceTransformer

print("Loading data...")
df = joblib.load("embeddings.joblib")

print("Removing old embeddings...")
df = df.drop(columns=["embedding"])

print("Loading model... (first time download hoga ~1.3GB)")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Creating new embeddings...")
df["embedding"] = df["text"].apply(lambda x: model.encode(x).tolist())

print("Saving...")
joblib.dump(df, "embeddings.joblib")
print("Done!")