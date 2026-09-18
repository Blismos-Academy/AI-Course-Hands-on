import sqlite3
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from fastmcp import FastMCP

# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()


# 2. DATABASE
def create_database():

    conn = sqlite3.connect("app_data.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    """)

    count = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    if count == 0:

        conn.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            [
                ("Alice", "alice@example.com"),
                ("Bob", "bob@example.com")
            ]
        )

    conn.commit()
    conn.close()


create_database()


# 3. MCP SERVER

mcp = FastMCP("SimpleAgent")


# Tool 1: Get User Information
@mcp.tool()
def get_user(user_id: int) -> str:
    """Get user information from the database."""

    conn = sqlite3.connect("app_data.db")

    user = conn.execute(
        "SELECT id, name, email FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    conn.close()

    if user:
        return (
            f"ID: {user[0]}, "
            f"Name: {user[1]}, "
            f"Email: {user[2]}"
        )

    return "User not found"


# Tool 2: Calculator
@mcp.tool()
def calculate(
    a: float,
    b: float,
    operation: str
) -> str:
    """Perform a simple calculation."""

    if operation == "add":

        result = a + b

    elif operation == "subtract":

        result = a - b

    elif operation == "multiply":

        result = a * b

    elif operation == "divide":

        if b == 0:
            return "Cannot divide by zero"

        result = a / b

    else:

        return "Invalid operation"

    return f"Result: {result}"



# 4. LLM

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# 5. AGENT LOGIC
def ask_agent(question: str, user_id: int = 1):

    # Decide which MCP tool to use

    if "user" in question.lower():

        print("Using MCP Tool: get_user")

        data = get_user(user_id)

    else:

        print("Using MCP Tool: calculate")

        # Example calculation
        data = calculate(
            10,
            20,
            "add"
        )

    # Send Tool Result to LLM
    prompt = f"""
    You are a helpful AI assistant.

    Answer the user's question using the data below.

    Data:
    {data}

    User question:
    {question}

    Give a simple and natural answer.
    """

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return response.content


# 6. FASTAPI SERVER
app = FastAPI(
    title="Simple MCP Agent",
    description="Beginner MCP + LLM + FastAPI example"
)


# 7. REQUEST MODEL

class Query(BaseModel):

    question: str

    user_id: int = 1


# 8. CALCULATOR REQUEST MODEL

class CalculatorRequest(BaseModel):

    a: float

    b: float

    operation: str


# 9. RESPONSE MODEL

class Response(BaseModel):

    answer: str


# 10. AGENT API ENDPOINT

@app.post(
    "/ask",
    response_model=Response
)
async def ask(request: Query):

    answer = ask_agent(
        request.question,
        request.user_id
    )

    return Response(
        answer=answer
    )


# 11. CALCULATOR API ENDPOINT

@app.post("/calculate")
async def calculator(request: CalculatorRequest):

    print("Using MCP Tool: calculate")

    result = calculate(
        request.a,
        request.b,
        request.operation
    )

    return {
        "result": result
    }

# 12. RUN SERVER
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
