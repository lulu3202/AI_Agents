# README file for pydantic.ipnyb:

## Overall Purpose
The primary goal of this notebook is to showcase how to build sophisticated applications with LLMs beyond simple prompt-response interactions. It introduces LangGraph, a library that helps you structure and orchestrate complex workflows involving LLMs.

## Key Concepts
- **LangGraph**: A framework for building complex, stateful applications with LLMs. It allows you to define workflows as directed graphs, where nodes represent LLM calls or other operations, and edges define the flow of information between them.
- **State**: A central object that stores information shared between the nodes in a LangGraph.
- **Nodes**: Functions or operations performed within the graph, such as making a call to an LLM, evaluating a result, or transforming data.
- **Edges**: Connections between nodes that define the order of execution and flow of data.
- **Workflows**: Different patterns of using LLMs, such as chaining, parallelization, routing, orchestrator-worker, and evaluator-optimizer.
- **Groq**: This notebook uses the Groq API for the LLM.

## Sections and Their Explanations
1. **Setup and Imports:**
   - The notebook starts by importing necessary libraries like langgraph, pydantic, and langchain_groq.
   - It defines type hints (TypedDict) for creating structured data.
   - It also imports some utility functions from IPython to allow displaying images.
   - It loads the .env file in order to get the GROQ_API_KEY and OPEN_API_KEY variables, this is necessary to be able to use the API.

2. **Basic State Graph Example:**
   - This section demonstrates the simplest use case of LangGraph.
   - It defines an OverallState (using pydantic) to hold a string value a.
   - It defines a single node function that returns a dictionary, which will be used to update the state.
   - It builds a StateGraph using builder, adds the node, and defines edges to indicate that execution starts with node and ends after it.
   - The code show that the state graph checks that the value of key a is a string and not an integer, and will throw an error if it is not a string.
   - It runs the graph with valid and invalid inputs, with the second input raising an error.

3. **Prompt Chaining:**
   - This section introduces a more advanced workflow called "prompt chaining."
   - Concept: A task is broken into a sequence of steps. The output of one LLM call becomes the input of the next one.
   - Example: The example creates a workflow to generate, check, improve, and polish a joke.
     - **generate_joke**: Generates an initial joke about a given topic.
     - **check_punchline**: Checks if the joke has a question mark or exclamation mark, which is needed for a good punchline.
     - **improve_joke**: Tries to make the joke funnier using wordplay.
     - **polish_joke**: Adds a surprising twist to the joke.
   - Conditional Edges: The check_punchline function uses conditional edges to control the flow of execution. If the joke has a punchline, it goes to END; otherwise, it goes to improve_joke.
   - The function check_punchline is the one that will use the conditions, and will route the workflow to the appropriate node.

4. **Parallelization:**
   - This section illustrates how to run multiple LLM calls concurrently.
   - Concept: Multiple independent tasks are executed at the same time.
   - Variations:
     - **Sectioning**: Dividing a task into subtasks.
     - **Voting**: Running the same task multiple times for diverse results.
   - Example: The example generates a joke, a story, and a poem about a given topic in parallel.
     - **call_llm_1**: Generates a joke.
     - **call_llm_2**: Generates a story.
     - **call_llm_3**: Generates a poem.
     - **aggregator**: Combines the joke, story, and poem into a single output.
   - Edges: All three LLM calls start at the same time (START). They all then feed into the aggregator, which then ends the graph.
   - The use of several LLM that run in parallel and that have their output concatenated into one single string.

5. **Routing:**
   - This section demonstrates how to use an LLM to classify an input and route it to a specialized task.
   - Concept: Directs an input to a specific node based on its characteristics.
   - Example: The example has an LLM router that decides whether the input should be used to generate a story, a joke, or a poem.
     - **llm_call_router**: Classifies the input.
     - **llm_call_1**: Writes a story.
     - **llm_call_2**: Writes a joke.
     - **llm_call_3**: Writes a poem.
   - Conditional Edges: The route_decision function determines the next node based on the output of llm_call_router.
   - The input is classified and the workflow is routed to the appropriate LLM based on this classification.
        
6. **Orchestrator-Worker:**
   - This section shows a more complex workflow where a central LLM (the orchestrator) breaks down a task, delegates it to worker LLMs, and then combines their results.
   - Concept: An orchestrator assigns subtasks to workers, and then collects and synthesizes their outputs.
   - Example: The example creates a report on a given topic.
     - **orchestrator**: Generates a plan for the report.
     - **llm_call**: A worker that writes a section of the report based on the plan.
     - **synthesizer**: Combines the sections into a final report.
   - Send API: The assign_workers function uses the Send() API to dynamically create and assign workers.
   - The orchestrator calls several workers in order to work in parallel to generate the report.
        
7. **Evaluator-Optimizer:**
   - This section shows a workflow where one LLM generates a response, and another evaluates it and provides feedback.
   - Concept: One LLM refines a response based on feedback from another LLM.
   - Example: The example refines a joke based on feedback from an evaluator.
     - **llm_call_generator**: Generates a joke or refines it based on feedback.
     - **llm_call_evaluator**: Evaluates the joke and provides feedback.
   - Conditional Edges: The route_joke function determines whether to go back to llm_call_generator for improvement or to END if the joke is funny.
   - An evaluator will decide if a result of an LLM is good enough, if not, it will be improved until it is accepted.

# README file for human_in_the_loop.ipnyb:

This notebook demonstrates how to use LangGraph, a library likely for building and executing state-based workflows, with LangChain, a framework for building language model applications. The notebook focuses on incorporating human-in-the-loop interactions within these workflows, allowing users to interrupt, debug, edit, and approve agent actions.

## Here's a step-by-step explanation:

1. **Setup and Initialization:**
   - Imports necessary libraries like langchain_groq, langgraph, and langchain_core.
   - Sets up environment variables for API keys (like GROQ_API_KEY).
   - Initializes a language model (LLM) using ChatGroq.
   - Defines basic arithmetic functions (multiply, add, divide) to be used as tools by the LLM.

2. **Building the Workflow Graph:**
   - Creates a StateGraph to represent the workflow.
   - Defines nodes within the graph, such as assistant for LLM interaction and tools for executing tool functions.
   - Establishes edges and conditional edges to control the flow between nodes.
   - Compiles the graph, enabling interruption before the assistant node and using MemorySaver for state persistence.

3. **Human-in-the-Loop Interactions:**
   - **Streaming and Interruption:**
     - Starts the workflow with an initial input and runs until interruption.
     - Prints the LLM's response at each step.
     - Accesses the current state of the graph.
     - Continues execution, potentially allowing user input.
   - **Editing State:**
     - Resets the workflow and starts again.
     - Updates the graph's state with new user instructions.
     - Prints the updated state and continues execution.
   - **Waiting for User Input:**
     - Modifies the workflow to include a human_feedback node that pauses execution.
     - Interrupts the workflow before the human_feedback node.
     - Prompts the user for input to update the state.
     - Continues the workflow based on user feedback.

## Essentially, the notebook showcases how LangGraph facilitates building workflows with points for human intervention. Users can:
- Interrupt the workflow to review intermediate results.
- Edit the state to modify the behavior.
- Provide feedback to guide the process.
- Approve actions or rewind to previous steps for debugging.

# README file for personal_assistant.ipnyb:

## Overall Goal: 
The notebook demonstrates the use of LangGraph, an agent framework, to create a system where AI analysts provide different perspectives on a given topic. It incorporates a human feedback loop to refine the generated analysts.

## Steps:
1. **Setup & Imports:**
   - Loads environment variables, likely for API keys.
   - Imports necessary libraries such as langchain_groq, pydantic, and langgraph.

2. **Defining Data Structures:**
   - Defines Pydantic models (Analyst, Perspectives) to structure the data for analysts and their perspectives.
   - These models help in organizing the information about each analyst, such as their name, role, affiliation, and description.

3. **Analyst Creation Logic (create_analysts function):**
   - This function takes a GenerateAnalystsState dictionary as input.
   - It uses ChatGroq as the language model.
   - It formats a system message using analyst_instructions and the provided topic and human feedback.
   - It then calls the language model to generate a list of analysts based on the system message and human feedback.
   - Finally, it returns a dictionary containing the generated list of analysts.

4. **Human Feedback Logic (human_feedback and should_continue functions):**
   - human_feedback is a placeholder function designed for human intervention in the process.
   - should_continue determines the flow of the program based on human feedback. If feedback is provided, it directs the program to create_analysts to generate new analysts. If no feedback is given, the program ends.

5. **Workflow Definition (LangGraph):**
   - It builds a state graph using langgraph.
