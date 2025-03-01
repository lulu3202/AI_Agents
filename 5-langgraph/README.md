# README file for simplegraph.ipnyb

## Part 1: Building a Simple State Graph
This part showcases a basic state machine, or graph, with the following components:

1. **State Definition:**
   - It starts by defining a State using TypedDict.
   - The state is a dictionary with a key named graph_state.
   - The graph_state stores a string value.
   - This structure is used to track the progression of the state machine.

2. **Nodes:**
   - Three functions (first_node, second_node, third_node) are defined. These will be the nodes in our graph.
   - Each function takes the current state as input.
   - Each node function updates the graph_state by adding a different string to the existing value.
   - The functions print a message to know which one is being called.
   - The original state is not changed in place; instead, each function returns a new state with the modified graph_state.

3. **Decision Logic (decide_play):**
   - This function acts as a decision point.
   - It randomly chooses between second_node and third_node based on a 50/50 probability.
   - The returned value of this function determines the next node to be executed.

4. **Graph Construction:**
   - A StateGraph is created using the State definition.
   - Nodes are added to the graph using builder.add_node.
   - Edges (connections between nodes) are defined:
     - The graph always starts at first_node.
     - From first_node, a conditional edge is added, directing execution to either second_node or third_node based on the output of decide_play.
     - Both second_node and third_node then lead to the END of the graph.
   - builder.compile() builds the graph.
   - graph.get_graph().draw_mermaid_png() generates a visual representation of the graph.
   - graph.invoke() starts the execution of the graph.

5. **Summary:**
   - The graph starts at the first_node.
   - The first_node randomly goes to second_node or third_node.
   - second_node and third_node are the end of the execution.
   - The nodes add information to the graph_state attribute in each execution.

## Part 2: Building a Basic Chatbot State Graph
This part demonstrates a more complex use case: creating a basic chatbot using a state graph.

1. **New State Definition:**
   - A new State is defined, now containing a messages key.
   - This key will store a list of messages for the chatbot's conversation.

2. **Environment Variables:**
   - os.environ["GROQ_API_KEY"] sets the environment variable with the value of the GROQ_API_KEY. This value is taken from the .env file with the load_dotenv() method.
   - This is important to authenticate with the API.

3. **LLM model:**
   - llm=ChatGroq(model="gemma2-9b-it") instantiates a Large Language Model (LLM) called gemma2-9b-it using the Langchain_groq library.

4. **Chatbot Node:**
   - A chatbot function is defined.
   - It takes the current state (containing the list of messages) and sends the latest messages to the LLM.
   - The response from the LLM is added as a new message to the messages list.
   - A dictionary with the new message is returned.

5. **Graph Construction:**
   - A new StateGraph is created with the new State definition.
   - A single "chatbot" node is added.
   - Edges are added: the graph starts at START, goes to the chatbot node, and then to END.

6. **Graph Visualization:**
   - graph.get_graph().draw_mermaid_png() displays a diagram of the simple chatbot graph.

7. **Streaming Updates:**
   - stream_graph_updates function: This handles taking user input and managing the conversation. It is responsible for the execution of the graph.
   - It passes user input to the graph in the messages key.
   - for event in graph.stream() returns all the intermediate steps of the graph execution.
   - The output is printed.

8. **Conversation Loop:**
   - The while True loop continuously prompts the user for input.
   - If the user types "quit," "exit," or "q," the loop breaks.
   - Otherwise, it calls stream_graph_updates to handle the conversation.

## In Summary
The notebook shows how to:
- Define states for state machines using TypedDict.
- Create nodes (functions) that modify the state.
- Add decision logic to conditionally direct the flow of the graph.
- Build and compile a StateGraph.
- Visualize the graph.
- Run the graph with an initial state.
- Integrate an LLM to build a simple conversational agent.
- Run the chat bot in a loop to maintain a conversation.

This is a comprehensive example of how to use langgraph to implement both simple and slightly more complex state transition systems.

# README file for lang_Graph_chains.ipnyb

Notebook demonstrating the use of LangChain and LangGraph for building conversational agents, particularly focusing on tool-calling.

## Notebook Setup
1. **Import Libraries:**
   - The notebook starts by importing various libraries, including:
     - IPython for interactive features in the notebook.
     - pprint for pretty-printing.
     - langchain_core for basic Langchain functionality.
     - langchain_groq for using Groq's LLM.
     - langchain_openai for using OpenAI's LLM.
     - dotenv and os for environment variable management.
     - langgraph for building graphs.
     - typing_extensions and typing for type hinting.

2. **Environment Variables:**
   - It loads environment variables from a .env file, which is common practice for storing sensitive information like API keys.
   - It then sets the GROQ_API_KEY and OPENAI_API_KEY environment variables, which are needed to interact with the Groq and OpenAI APIs, respectively. Remember, to use this code you need to have your own API keys and a .env file setup.

## Conversational Examples
1. **Basic Message Exchange:**
   - A sequence of AIMessage and HumanMessage objects are created to represent a simple conversation about ocean mammals.
   - The pretty_print() method is used to display the messages in a readable format.

2. **Groq LLM Interaction:**
   - A ChatGroq instance is created, specifying the qwen-2.5-32b model.
   - The LLM is invoked with the list of messages, and the result is stored. This demonstrates a basic interaction with the Groq LLM.

## Tool-Calling Examples
1. **Simple Addition Function:**
   - A Python function add(a, b) is defined to perform addition.

2. **Binding a Tool to an LLM:**
   - The llm_with_tools object is created by binding the add function as a tool to the ChatGroq LLM. This allows the LLM to call the add function if it determines it's necessary to answer a user's query.

3. **Invoking the LLM with a Tool:**
   - The llm_with_tools is invoked with a user query, "What is 2 plus 3".
   - The tool_calls attribute of the result is accessed to show the LLM's tool call. To see the output, run the code.

## LangGraph Examples
1. **Defining a Message State:**
   - A MessageState is defined using TypedDict and Annotated. This will be used to represent the state of the conversation in the graph.
   - This is used to represent a dynamic conversation in a graph.

2. **Adding Messages to a List:**
   - An example shows how to add a new AIMessage to a list of existing messages using add_messages.

3. **Building a LangGraph:**
   - A StateGraph is built using MessageState.
   - A node tool_calling_llm is added, which invokes the llm_with_tools object, returning the output of the LLM call.
   - Edges are added to connect the nodes. In this case, it forms a simple loop that goes from START to tool_calling_llm and then to END.
   - The graph is compiled.
   - The graph is then visualized as a Mermaid diagram. To see the diagram, run the code.

4. **Invoking the LangGraph:**
   - The graph is invoked with a question. The response will then include the conversational history and the LLMs response. To see the output, run the code.

## OpenAI LLM Example
1. **OpenAI Setup:**
   - The notebook then switches to using OpenAI's models.
   - The OPENAI_API_KEY environment variable is used to authenticate with OpenAI.

2. **Multiply Function:**
   - A multiply(a, b) function is defined.

3. **OpenAI LLM Interaction:**
   - A ChatOpenAI instance is created, specifying the gpt-4o model.
   - A simple invocation with "Hello" is done, and the result is saved.

4. **Binding Tools to OpenAI LLM:**
   - The multiply and add functions are bound as tools to the ChatOpenAI LLM, creating llm_with_tools.

5. **Building a More Complex Graph:**
   - A more complex LangGraph is built, again using StateGraph, but this time using MessagesState.
   - A tool_calling_llm node is defined, similar to the previous graph.
   - A tools node is added, which uses ToolNode to represent the available tools.
   - add_conditional_edges is used to make the graph dynamic:
     - If tool_calling_llm results in a tool call, the graph routes to the tools node.
     - If tool_calling_llm does not result in a tool call, the graph routes to the END.
     - tools then routes to END.
   - The graph is visualized as a Mermaid diagram. To see the diagram, run the code.

6. **Invoking the Complex Graph:**
   - The graph is invoked with a complex query involving multiple operations, "Add 3 and 4. Multiply the output by 2 and add 5."
   - The conversation history is returned along with the final answer. To see the output, run the code.

# README file for lang_Graph_agents.ipnyb

## Overall Goal:
The primary objective of this notebook is to build an intelligent agent that can perform a series of arithmetic calculations based on user input and then maintain memory between different user interactions.

## Key Concepts:
1. **ReAct Agent:**
   - ReAct stands for "Reasoning and Acting". It's an agent architecture where the agent can:
     - Act: Call specific tools to perform tasks.
     - Observe: Process the results from the tool.
     - Reason: Decide the next action based on the tool's output.

2. **LangGraph:**
   - A library that helps create graphs of nodes, where each node is a function/action.
   - It provides a way to define how these nodes connect, creating complex workflows for agents.

3. **Tools:**
   - Functions that the agent can call to perform actions. In this notebook, the tools are add, multiply, and divide.

4. **Large Language Model (LLM):**
   - The "brain" of the agent. It's responsible for reasoning about user input and deciding which tools to call and in what order.
   - Here, gpt-4o from OpenAI is used as the LLM.

5. **MessagesState:**
   - This is a custom data structure that is used to pass messages between nodes in the LangGraph.
   - It is used to store the conversation history (user messages and assistant responses).

6. **Memory:**
   - The agent needs to remember previous interactions to answer follow-up questions.
   - The MemorySaver class is used for this, allowing the agent to remember interactions in a specific "thread".

## Code Breakdown:
1. **Setup:**
   - The code first imports necessary libraries (os, dotenv, langchain_openai, langgraph, etc.).
   - It then loads API keys for Groq and OpenAI from a .env file.

2. **Tool Definition:**
   - Three tools are defined: add, multiply, and divide.
   - Each tool is a simple Python function with type hints and docstrings.

3. **LLM Setup:**
   - A ChatOpenAI instance is created using the gpt-4o model.
   - The bind_tools method connects these tools to the LLM, creating llm_with_tools.

4. **MessagesState and System Message:**
   - The MessagesState class is defined as a state to manage messages in a conversation.
   - A system message is defined that tells the LLM it is a math assistant.

5. **Assistant Node:**
   - The assistant function is a node in the graph.
   - It is the core logic, where the LLM will process the current messages in MessagesState and either return an answer or call a tool.

6. **LangGraph Creation:**
   - A StateGraph is created using MessagesState.
   - The assistant and tools nodes are added to the graph.
   - Edges are created to define the flow of control:
     - START to assistant: The first node that is visited.
     - assistant to tools_condition:
       - If the latest message is a tool call, the next node is the tools node.
       - Otherwise, it goes to the end of the graph.
     - tools to assistant: The output of a tool is passed back to the assistant.
   - The react_graph is compiled.
   - The output is displayed as a mermaid diagram.

7. **Basic ReAct Example:**
   - The code creates a message, HumanMessage(content="Add 10 and 14. Multiply the output by 2. Divide the output by 5"), which will be passed as input to the graph.
   - The graph is invoked, the assistant node is executed and the tools are called if necessary.
   - The output is printed, showing how the graph responded to the messages.

8. **Agent Memory:**
   - The next section shows that the agent does not have memory between interactions.
   - A first message is sent: HumanMessage(content="Add 14 and 15.")
   - A follow-up message is sent: HumanMessage(content="Multiply that by 2."). It will not be able to respond correctly.

9. **Memory with MemorySaver:**
   - A MemorySaver instance is created.
   - The react_graph is compiled with checkpointer=memory.
   - Thread 1:
     - The code sends HumanMessage(content="Add 3 and 4."), using config={"configurable":{"thread_id":"1"}}.
     - A follow-up message HumanMessage(content="Multiply that by 2.") is sent, using the same config.
     - The agent should now know that "that" refers to 7 from the previous message.
   - Thread 2:
     - The same two messages are sent but with config1={"configurable":{"thread_id":"2"}}.
     - The agent now has memory for two different conversations/threads.

## In essence:
This notebook builds a ReAct agent, capable of performing calculations, by setting up a graph of nodes that communicate and pass messages between each other. It then introduces memory so that the agent can remember previous interactions.
