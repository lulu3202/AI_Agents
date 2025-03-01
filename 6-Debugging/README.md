# README file for FOLDER 6 - DEBUGGING:

This openai_agent python script defines and compiles two different LangGraph workflows using LangChain and OpenAI's GPT model.

1. **Environment Setup:**
   - Loads .env variables for API keys (OPENAI_API_KEY and LANGCHAIN_API_KEY).
   - Initializes the ChatOpenAI model with zero temperature (deterministic responses).

2. **State Definition:**
   - Defines State, a dictionary that stores messages and uses add_messages to maintain conversation history.

3. **Graph 1 (Simple LLM Agent):**
   - The make_default_graph() function:
     - Creates a basic chatbot agent.
     - Calls the model with previous messages.
     - Defines a linear workflow where the model processes input and ends execution.

4. **Graph 2 (Tool-Calling LLM Agent):**
   - The make_alternative_graph() function:
     - Defines a tool (add()) to add two numbers.
     - Integrates the tool with the model.
     - Uses conditional logic to decide whether to call the tool ("tools") or finish execution (END).

5. **Execution:**
   - To track this in LangSmith, a json file is needed. Inside this dependencies are provided per LangSmith documentation.
   - The last line agent is also referenced there ../env means go to parent directory
   - For langgraph studio, you need to import langgraph-cli library
   - To run this, go to cd 6-Debugging folder and type "langgraph dev" to execute this directly in Langsmith - automatically langsmith opens up

```json
{
  "dependencies":["."],
  "graphs":{
    "openai_agent":"./openai_agent.py:agent"
  },
  "env":"../.env"
}
