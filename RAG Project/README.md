# PROJECT: ASK KRISHNA 🪈
## Everything You Need to Know (In Simple & Technical Terms)

---

### Part 1: Explaining to a 9-Year-Old 🧒
*"The Magic Virtual Librarian"*

Imagine you have a **Magic Library** filled with thousands of ancient, dusty books (the Puranas). You want to ask the librarian a question, but you don't want just any answer—you want it to sound exactly like **Lord Krishna** talking to his best friend.

Here is how we built it:

1. **The Super-Fast Book Reader (`ingest.py`):** We have a robot that read all the books, cut them into small pieces (chunks), and turned them into "Secret Number Codes" (Embeddings).
2. **The Brainy Box (ChromaDB):** We put those secret codes into a special box. When you ask a question, the box finds the 5 most similar pieces of paper almost instantly.
3. **The Wise Messenger (FastAPI):** We built a small telephone office that takes your question from the website and sends it to the AI.
4. **The Great Actor (OpenAI):** We told the smartest AI in the world: *"You are Krishna. Use these 5 pieces of paper we found in the library, and answer this child with love, a Sanskrit poem (Shloka), and a little Hinglish."*
5. **The Beautiful Mirror (`ask_krishna.html`):** We built a shiny website with stars and a flute so you can talk to him whenever you feel sad or confused.
