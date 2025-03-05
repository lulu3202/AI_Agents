# AGENTIC_RAG.ipynb

This notebook builds a system for question answering using LangChain and LangGraph, leveraging a vector database and large language models (LLMs). Here's a step-by-step explanation:

## 1. Setting up the Environment and Data
- Loads environment variables, likely for API keys for services like Groq and OpenAI.
- Loads web pages from specified URLs using `WebBaseLoader` as the knowledge base.
- Splits the loaded documents into smaller chunks using `RecursiveCharacterTextSplitter`.
- Embeds these chunks using `OpenAIEmbeddings` and stores them in a FAISS vector database for efficient similarity search.

## 2. Creating a Retriever Tool
- A retriever tool is created using the FAISS vector database to retrieve relevant information based on user queries.

## 3. Defining the Agent and Workflow
- Defines an agent function using a Groq LLM to process user queries, deciding whether to retrieve information or generate a response directly.
- Uses `LangGraph` to create a workflow that includes steps for retrieving information, rephrasing queries if needed, and generating responses.

## 4. Defining Edges and Conditions
- Defines edges in the workflow to control the execution flow between nodes.
- Uses conditional edges to determine the next step based on the relevance of retrieved documents to the query.

## 5. Compiling and Running the Graph
- Compiles the workflow into a runnable graph.
- Invokes the graph with a user query, initiating the question-answering process.
- Prints the output, providing the answer to the user's question.

## System Capabilities
This notebook creates a system that can:
1. Understand user questions.
2. Retrieve relevant information from a knowledge base.
3. Potentially rephrase questions for better results.
4. Generate answers based on the retrieved information.

### Summary of the Process
1. Receives the user's question.
2. Retrieves relevant information from the knowledge base.
3. Assesses the relevance of the information.
4. Generates an answer using the relevant information.
5. Returns the answer to the user.

---

# ADAPTIVE_RAG.ipynb

This notebook builds a more adaptive system for question answering, incorporating question routing, retrieval grading, and adaptive generation strategies.

## 1. Document Loading and Vectorization
- Loads documents from three URLs related to agents, prompt engineering, and adversarial attacks.
- Splits the documents into smaller chunks.
- Embeds these chunks using `OpenAIEmbeddings` and stores them in a FAISS vectorstore for efficient similarity search.

## 2. Question Routing
- A router decides whether a user's question should be answered using the vectorstore (for topics covered in the loaded documents) or web search (for other topics).
- The decision is made by a language model (`ChatOpenAI`) based on the question.

## 3. Retrieval and Grading
- If routed to the vectorstore, the system retrieves relevant documents based on the question.
- A retrieval grader (another language model) assesses the relevance of the retrieved documents, filtering out irrelevant ones.

## 4. Answer Generation and Grading
- A RAG (Retrieval-Augmented Generation) chain generates an answer based on the retrieved documents and the question.
- The answer is graded by two more language models:
  - **Hallucination grader**: Checks if the answer is grounded in the retrieved documents.
  - **Answer grader**: Checks if the answer addresses the question.

## 5. Question Rewriting
- If the initial answer is unsatisfactory, the question is rewritten to improve retrieval and generation.

## 6. Web Search
- If routed to web search, the system uses `Tavily Search` to find relevant web pages.

## 7. Workflow Orchestration
- Uses a `StateGraph` from `LangGraph` to define the flow of the question-answering process.
- The graph includes nodes for each step (retrieval, grading, generation, etc.) and edges that determine the order of execution.

## 8. Execution
- Runs the workflow with an example question, displaying intermediate outputs and the final generated response.

---

Both of these notebooks demonstrate different approaches to building question-answering systems with LLMs, vector databases, and workflow automation.
