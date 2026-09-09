# Multi-Agent Weather, RAG & Memory Assistant

A simple AI assistant that tells you the current weather, recommends what to wear and eat, and remembers what you did before (like what you wore yesterday) — all by combining four small "agents" that each do one job.

---

## What This Project Does

You ask something like:

> "What is the weather in Bangalore right now? What should I wear and eat, and what did I wear yesterday?"

The assistant answers by pulling together three different sources of information instead of guessing:

1. **Live weather** — fetched from a real weather service (not made up).
2. **Clothing & food advice** — retrieved from a knowledge document using RAG (Retrieval-Augmented Generation), not hard-coded in the app.
3. **Your personal history** — retrieved from a memory store (e.g., what you said you wore yesterday).

A final "Chat" step combines all three into one clear answer.

---

## How It Works (Plain-English Architecture)

```
YOU ask a question
       │
       ▼
┌─────────────────┐
│  Supervisor /    │  Figures out what info is needed
│  Chat Agent      │
└────────┬─────────┘
         │
   ┌─────┼─────────────┬─────────────┐
   ▼                   ▼             ▼
┌──────────┐     ┌────────────┐  ┌───────────┐
│ RAG Agent│     │ Weather    │  │ Memory    │
│          │     │ Tool Agent │  │ Agent     │
└────┬─────┘     └─────┬──────┘  └─────┬─────┘
     │                 │               │
     ▼                 ▼               ▼
 Clothing/food     Real-time        Your past
 guidance from     weather from     history
 the document      an API           (past outfits, etc.)
     │                 │               │
     └────────┬────────┴───────────────┘
              ▼
     ┌──────────────────┐
     │  Chat/LLM Agent   │  Combines everything into
     │  + Validation     │  one grounded answer
     └────────┬──────────┘
              │
        (if it fails)
              ▼
     ┌──────────────────┐
     │  Fallback Model    │
     └────────┬──────────┘
              ▼
      FINAL ANSWER TO YOU
```

### The Four Agents, in Simple Terms

| Agent | Job | Where its info comes from |
|---|---|---|
| **RAG Agent** | Looks up clothing/food advice for the current weather | A document, broken into chunks, searched by meaning |
| **Weather Tool Agent** | Gets the actual current weather | A real weather API (never hard-coded) |
| **Memory Agent** | Remembers things you told it before | A separate memory store, not the document |
| **Chat/LLM Agent** | Combines everything into a final answer | Output of the other three agents |

**Important distinction:** RAG = shared general knowledge (the document). Memory = things specific to *you* (like yesterday's outfit). These are kept separate on purpose.

---

## Project Structure

```
project/
├── agents/
│   ├── rag_agent.py         # Searches the recommendation document
│   ├── weather_agent.py     # Calls the weather API/tool
│   ├── memory_agent.py      # Stores/retrieves user history
│   └── chat_agent.py        # Combines everything, validates, falls back if needed
├── data/
│   └── weather_clothing_food_recommendations.md   # The RAG source document
├── graph.py                 # LangGraph workflow wiring the agents together
├── schemas.py                # Input/output validation schemas
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your environment variables

Copy the example file and fill in your own keys (never commit real keys):

```bash
cp .env.example .env
```

You'll need, at minimum:
- An LLM API key (primary model)
- A fallback LLM API key/model name
- A weather API key

### 3. Ingest the document (build the RAG index)

Run the ingestion script once to load, chunk, embed, and store the recommendation document in the vector store:

```bash
python ingest.py
```

### 4. Run the assistant

```bash
python main.py
```

Then type a question, for example:

```
What is the weather in Bangalore right now? What should I wear and eat, and what did I wear yesterday?
```

---

## Example Questions to Try

| Type | Example |
|---|---|
| Weather only | "What's the weather in Chennai?" |
| RAG only | "What should I wear in cold weather?" |
| Memory only | "What did I wear yesterday?" |
| Combined | "It's raining in Mumbai — what should I wear and eat?" |
| Validation/fallback | Try an empty or invalid city name and see it get rejected gracefully |

---

## What Happens If Something Fails?

- If a tool gets **bad input** (e.g., an empty city name), it's rejected before it ever runs — you'll see a clear validation message instead of a crash.
- If the **main AI model fails** (timeout, API error, etc.), the system automatically switches to a backup model so you still get an answer.
- Every step logs what it's doing, so you can see exactly which agent ran and what it returned.

---

## Where Each Tool Is Used

- **LangChain** — defines the weather tool, its input schema, and validation.
- **LlamaIndex** — loads, chunks, embeds, and indexes the recommendation document for RAG.
- **LangGraph** — wires the four agents together as a graph with shared state, instead of one big function.

---

## Notes

- Weather data is always fetched live — never hard-coded.
- Clothing/food advice always comes from the retrieved document — never hard-coded in the app logic.
- The `.env` file is git-ignored; only `.env.example` (with placeholder values) is committed.