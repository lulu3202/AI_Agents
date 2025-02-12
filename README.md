# 1. Getting Started - AI, ML, GenAI, and Agentic AI Overview

## Artificial Intelligence (AI)
- AI performs tasks independently without human intervention.  
  _Example: Self-driving cars_

## Machine Learning (ML)
- ML provides statistical tools to analyze, visualize, and predict data.  
  _Example:_  
  - In ML, accuracy improves with more training data, but only up to a certain point. Beyond this, accuracy saturates as dimensionality increases.

## Multi-Neural Networks (Multi-NN)
- As data increases, accuracy improves.
- Backpropagation plays a key role in enhancing performance.

## Deep Learning (DL)
- DL emerged with the availability of massive datasets.
- Key architectures include:
  - RNN
  - LSTM RNN
  - Encoder-Decoder
  - Transformer
  - BERT
- The secret behind the success of Transformers is **Self-Attention**.
- Unlike traditional ML, DL continues to improve with more data.

## Generative AI (Gen AI)
- A subset of Deep Learning (DL) focused on generating new content.

## Agentic AI
- A subset of Generative AI.
- Unlike Gen AI, its primary goal is to solve complex workflows rather than generate new content.
- Agentic AI leverages multiple AI agents or tools.

### AI Agent
- AI agents are specialized tools designed to perform specific tasks within an AI system.
- Agentic AI is the broader system that utilizes these AI agents to achieve its objectives.

# 2. RAG and LangChain Overview 

## RAG Fundamentals
RAG enhances LLM responses by retrieving relevant external knowledge before generating an answer.

Basic RAG Pipeline:
1. Data Source – Identify relevant data (e.g., documents, articles, PDFs).
2. Data Loader – Load the data into a structured format.
3. Chunking – Split data into smaller segments (e.g., using text splitters like RecursiveCharacterTextSplitter).
4. Embedding – Convert text chunks into vector representations (e.g., OpenAI embeddings).
5. Vector Store – Store embeddings in a vector database (e.g., FAISS, Pinecone).
6. Retrieval via Similarity Search – When a query is made, the most relevant chunks are retrieved as context.
This process ensures LLMs generate responses based on up-to-date and relevant information rather than relying solely on pre-trained knowledge.

Simple RAG Workflow:
1. Data ingestion (libraries like PYPDF Loader, WebBaseLoader, Wikipedia Loader)
2. Data Transformation (recursive character text splitter, character text splitter, HTML text Splitter, Recursive JSON Splitter)
3. Embedding (Open ai embedding - text embeddng 3 large, Ollama embeddings - llama 2, HF Embeddings - all-miniLM-L6 )
4. Vector Store (Faiss and Chroma)
   #### Faiss
Facebook AI Similarity Search (Faiss) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.
docs=db.similarity_search(query)
docs_and_score=db.similarity_search_with_score(query)

We can also convert the vectorstore into a Retriever class.

## LangChain Fundamentals
LangChain is a powerful framework that integrates with any Large Language Model (LLM) to build advanced AI applications efficiently.

Key LangChain Components:
- LangGraph – Enables building more complex, structured AI workflows.
- LangServe – Simplifies deploying generative AI applications as APIs.
- LangSmith – Helps debug, monitor, and analyze interactions with LLMs.
- Integrations – Supports third-party integrations to extend functionality.

### Chains
Chains are easily reusable components linked together.

Chains encode a sequence of calls to components like models, document retrievers, other Chains, etc., and provide a simple interface to this sequence.

The Chain interface makes it easy to create apps that are:

- Stateful: add Memory to any Chain to give it state,

- Observable: pass Callbacks to a Chain to execute additional functionality, like logging, outside the main sequence of component calls,

- Composable: combine Chains with other components, including other Chains.

### Runnables vs LCEL Chains 
Runnables:
- Fundamental Building Blocks: Runnables are the core components in LangChain. They represent any object that can be "run" or executed, such as LLMs, prompts, retrievers, tools, etc.
- Standardized Interface: They implement a standardized interface (the Runnable interface) which defines methods like invoke, batch, stream, etc., ensuring consistency in how they are used.
- Individual Units: Runnables are individual units of functionality. They can be used on their own, but their real power comes when combined into chains.

LCEL Chains:
- Orchestration and Composition: LCEL chains are a way to combine and orchestrate multiple Runnables to create complex workflows.
- Declarative Syntax: They use a declarative syntax (| operator) to specify the flow of data and execution between Runnables. This makes the chain logic more readable and easier to understand.

# 3. Chatbots 
## Chatbots notebook
chatbots notebook is using the langchain library to build a conversational chatbot powered by Groq (i.e AI inference Engine).It aims to showcase the capabilities of langchain for building conversational chatbots. It demonstrates basic message exchange, session management, prompt template usage, language customization, and conversation history management to create an interactive and efficient chatbot experience.

Steps:
1. Setup:
Imports necessary libraries like langchain, langchain_groq, etc.
Loads environment variables, potentially containing API keys.
Initializes a ChatGroq instance with the Gemma2-9b-It model.

2. Basic Interaction:
Sends a simple message to the model and prints its response using model.invoke.
Demonstrates basic message exchange using HumanMessage and AIMessage.

3. Session History:
Introduces the concept of session history using ChatMessageHistory to maintain context across interactions within a session.
Defines a get_session_history function to retrieve or create a session history.
Uses RunnableWithMessageHistory to wrap the model and manage sessions.

4. Prompt Templates:
Defines a ChatPromptTemplate to provide a system message and placeholder for user messages.
Creates a chain combining the prompt template and the model.

Conversation with Prompt Templates:
Demonstrates using the chain to interact with the model, passing user messages and receiving responses.

5. Language Customization:
Modifies the prompt template to accept a language parameter.
Shows how to pass the language parameter in the chain's invocation.

6. Conversation History Management and Trimming:
Explains the importance of managing conversation history and limiting its size.
Introduces trim_messages to trim excess messages from the history.
Demonstrates using the trimmer to reduce the context sent to the model.
Integrates the trimmer into the chain using RunnablePassthrough to manage history automatically.
Shows how the chatbot remembers recent information even after trimming.

## tools_agents
The tools_agents notebook aims to build a simple search engine using LangChain's tools and agents. It leverages large language models (LLMs) and external APIs to answer user queries.

The notebook demonstrates how to use LangChain's framework to build a basic question-answering system that can access multiple information sources.

Steps:
1. Import Libraries and Set Up: Imports necessary libraries like langchain, OpenAI, etc., and sets up API keys for Groq and OpenAI.
2. Define Tools:
Creates tools for querying Wikipedia and Arxiv using their respective APIs.
Builds a custom tool for searching the LangChain documentation using FAISS for vector search.
3. Initialize Language Model and Prompt:
Initializes a ChatGroq language model from Groq.
Uses a pre-built prompt template for OpenAI Functions Agent.
4. Create and Execute Agent:
Creates an agent that combines the language model and tools using create_openai_tools_agent.
Initializes an AgentExecutor to run the agent with the defined tools.
5. Test the Search Engine:
Runs the agent executor with sample user queries related to Langsmith, machine learning, and a specific research paper on Arxiv.

## vector_retriever notebook 
This notebook demonstrates how to build a system that can answer questions by:
- Storing information in a structured way (documents).
- Converting text to a format that can be searched efficiently (embeddings and vector store).
- Finding relevant information based on a question (retriever).
- Combining the relevant information with a language model to generate an answer (RAG).

### RAG vs Retriever
- A Retriever can be used as a standalone component to simply fetch information, but it's often integrated into larger workflows like RAG.

- RAG is a technique that combines the power of information retrieval (using a Retriever) with the text generation capabilities of a language model.

Purpose of Retreiver: A Retriever's primary job is to fetch relevant information from a knowledge base or data source based on a given query. It's like a specialized search engine within your application.

RAG Workflow
- Retrieval: A Retriever is used to fetch relevant documents or information based on the user's query.
- Augmentation: The retrieved information is used to "augment" the prompt that is given to the language model. This provides the language model with context and knowledge relevant to the query.
- Generation: The language model then generates a response based on both the original query and the retrieved context

### RAG vs AI Agents
RAG: Emphasizes enhancing language model outputs with external knowledge.
Agents: Focuses on enabling language models to perform actions and interact with their environment.

### Vector Store vs Reteriver
The key difference is that a Vector Store is a data storage and retrieval mechanism, while a Retriever is a more abstract component that uses a strategy (potentially involving a Vector Store) to fetch relevant information.