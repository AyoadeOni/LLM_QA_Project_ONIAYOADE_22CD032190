# LLM_QA_CLI.py - Command Line Question Answering System
import os
import re
import sys
from dotenv import load_dotenv
from groq import Groq

# Load the secret key
load_dotenv()

# Check if key exists
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found!")
    print("Please put your key in the .env file")
    sys.exit(1)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\?]', '', text)   # remove punctuation except ?
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_answer(question):
    prompt = f"""You are a helpful Q&A assistant. Answer clearly and naturally.

Question: {question}
Answer:"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-instant",
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🧠 LLM Question-Answering System (CLI)")
    print("Type 'exit' or 'quit' to stop\n")

    while True:
        try:
            q = input("Your question: ").strip()
            if q.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            if not q:
                print("Please type a question.\n")
                continue

            print(f"\nOriginal   : {q}")
            processed = preprocess(q)
            print(f"Processed  : {processed}")
            print("\nThinking...")

            answer = get_answer(processed)
            print(f"\nAnswer:\n{answer}\n")
            print("-" * 70)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()