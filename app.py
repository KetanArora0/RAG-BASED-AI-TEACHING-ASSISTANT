import streamlit as st
import pandas as pd
import joblib
from groq import Groq
import os

# Load data (video transcripts + embeddings)
df = joblib.load("embeddings.joblib")

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.title("AI Teaching Assistant (Video Based)")

query = st.text_input("Ask your question")

if st.button("Submit"):
    if query:

        # 🔥 Step 1: smart keyword filtering
        words = query.lower().split()

        filtered = df[
            df["text"].str.lower().apply(
                lambda x: any(word in x for word in words)
            )
        ]

        # 🔥 Step 2: fallback (important)
        if len(filtered) == 0:
            filtered = df.sample(5)
        else:
            filtered = filtered.head(5)

        # 🔥 Step 3: build context
        context = "\n".join(filtered["text"].tolist())

        # 🔥 Step 4: prompt
        prompt = f"""
You are a teaching assistant.

Answer ONLY from the lecture content below.
Mention video number if possible.

Content:
{context}

Question:
{query}

Answer:
"""

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant"
            )

            answer = response.choices[0].message.content.strip()

            st.subheader("Answer:")
            st.write(answer)

        except Exception as e:
            st.error(str(e))