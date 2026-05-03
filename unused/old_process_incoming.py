import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from groq import Groq
import os

# ✅ Initialize Groq client (IMPORTANT FIX)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embeddings
df = joblib.load('embeddings.joblib')

# LLM function
def inference(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model = "llama-3.1-8b-instant"
    )
    return response.choices[0].message.content

# Take user input
incoming_query = input("Ask a Question: ")

# ✔ Temporary workaround (use existing embedding)
question_embedding = np.array(df['embedding'].iloc[0])

# Similarity search
similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [question_embedding]
).flatten()

top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]
new_df = df.loc[max_indx]

# Prompt
prompt = f"""
You are an AI teaching assistant.

Answer the user's question based on the provided lecture content.

If the answer is not clearly in the content, give a general explanation.

Content:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

Question:
{incoming_query}

Answer:
"""

# Save prompt (optional)
with open("prompt.txt", "w") as f:
    f.write(prompt)

# Get answer
response = inference(prompt)

# Print output
print("\nAnswer:\n")
print(response)

# Save response
with open("response.txt", "w") as f:
    f.write(response)