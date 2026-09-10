from agents.chat_agent import LLMCore


def main():

    # Initialize the LLM
    llm = LLMCore()

    print("AI Q&A Assistant")
    print("Type 'exit' to stop.\n")

    while True:

        # Get user question
        user_question = input("You: ")

        # Exit condition
        if user_question.lower().strip() == "exit":
            print("AI: Goodbye!")
            break

        # Ignore empty input
        if not user_question.strip():
            continue

        # Send question to LLM
        answer = llm.ask(user_question)

        # Display answer
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()