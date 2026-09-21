import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Updated to use Groq with high-parameter LLaMA model
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# 1. SHORT-TERM MEMORY (Current Turn Context)

# Temporary holding space for the immediate exchange.
short_term = [
    HumanMessage(content="My current task is reviewing pull request #42.")
]
print("--- 1. Short-Term Memory ---")
res = llm.invoke(short_term)
print(f"Response: {res.content}\n")

# 2. CONVERSATION MEMORY (Chat History Stream)

# Sequential log of past turns in an active session.
chat_history = [
    HumanMessage(content="Hi, my name is virendra."),
    HumanMessage(content="What is my name?")
]
print("--- 2. Conversation Memory ---")
res = llm.invoke(chat_history)
print(f"Response: {res.content}\n")


# 3. WORKING MEMORY (Active Constraints & System Instructions)

# Active rules or scratchpad governing how the model processes inputs.
working_memory_prompt = [
    SystemMessage(content="Rule: Respond in 3 words or fewer. Capitalize every word."),
    HumanMessage(content="Explain what Python is.")
]
print("--- 3. Working Memory ---")
res = llm.invoke(working_memory_prompt)
print(f"Response: {res.content}\n")

# 4. LONG-TERM MEMORY (External Persistent Store / Database)

# Information stored outside the immediate context window and retrieved.
user_db = {"user_id": "ram_123", "preferred_language": "Python"}

# Simulating retrieval from database
retrieved_fact = user_db.get("preferred_language")
long_term_prompt = [
    HumanMessage(content=f"User preference retrieved from database: {retrieved_fact}. Write a print statement in their favorite language.")
]
print("--- 4. Long-Term Memory ---")
res = llm.invoke(long_term_prompt)
print(f"Response: {res.content}\n")


# 5. EPISODIC MEMORY (Specific Past Events / Experiences)

# Memories of specific past interaction episodes or occurrences.
past_episodes = [
    {"event_id": "ep_001", "detail": "Yesterday, Ram faced a RecursionError in his script."}
]
episodic_prompt = [
    HumanMessage(content=f"Past Episode Context: {past_episodes[0]['detail']}\nQuestion: What issue did I encounter yesterday?")
]
print("--- 5. Episodic Memory ---")
res = llm.invoke(episodic_prompt)
print(f"Response: {res.content}\n")

# 6. SEMANTIC MEMORY (General Knowledge & Concepts)

# Fact-based world knowledge, independently stored or injected as reference context.
knowledge_base = {
    "concept": "Recursion",
    "definition": "A programming technique where a function calls itself until reaching a base condition."
}
semantic_prompt = [
    HumanMessage(content=f"Reference Knowledge: {knowledge_base['definition']}\nQuestion: Explain {knowledge_base['concept']} simply.")
]
print("--- 6. Semantic Memory ---")
res = llm.invoke(semantic_prompt)
print(f"Response: {res.content}\n")


# 7. PROCEDURAL MEMORY (Step-by-Step Instructions / Workflows)

# Knowledge of *how* to perform specific tasks or algorithmic execution steps.
workflow_procedure = """
To debug a Python script:
Step 1: Check the trace back for the error line.
Step 2: Print or log variables leading up to the error.
Step 3: Fix the root issue and test again.
"""
procedural_prompt = [
    SystemMessage(content=f"Execution Procedure:\n{workflow_procedure}"),
    HumanMessage(content="Summarize the first step to debug code based on your procedure.")
]
print("--- 7. Procedural Memory ---")
res = llm.invoke(procedural_prompt)
print(f"Response: {res.content}\n")
