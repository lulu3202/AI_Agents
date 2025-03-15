import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display, Markdown
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import Send
import operator

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatGroq(model="qwen-2.5-32b")


# Schema for structured output to use in planning
class Topic(BaseModel):
    name: str = Field(
        description="Name for this topic in the learning path.",
    )
    description: str = Field(
        description="Brief overview of the main concepts to be covered in this topic.",
    )


class Topics(BaseModel):
    topics: List[Topic] = Field(
        description="List of learning topics.",
    )


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Topics)


# Graph state
class State(TypedDict):
    user_skills: str  # User's existing skills
    user_goals: str  # User's learning goals
    topics: List[Topic]  # List of topics to learn
    completed_topics: Annotated[List[str], operator.add]  # Summaries of each topic
    learning_roadmap: str  # Final learning roadmap


# Worker state
class WorkerState(TypedDict):
    topic: Topic
    completed_topics: List[str]


# Nodes
@traceable
def orchestrator(state: State):
    """Orchestrator that creates a study plan based on user skills and goals."""
    study_plan = planner.invoke(
        [
            SystemMessage(
                content="Create a detailed study plan based on the provided user skills and goals. The study plan should be structured as a list of topics, each with a brief description of what it covers."
            ),
            HumanMessage(
                content=f"User skills: {state['user_skills']}\nUser goals: {state['user_goals']}"
            ),
        ]
    )

    print("Study Plan:", study_plan)

    return {"topics": study_plan.topics}


@traceable
def llm_call(state: WorkerState):
    """Worker generates a content summary for a topic."""
    topic_summary = llm.invoke(
        [
            SystemMessage(
                content="Generate a content summary for the provided topic. The summary should briefly explain the topic and list key resources for further study. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the topic to summarize: {state['topic'].name}\nDescription:{state['topic'].description}"
            ),
        ]
    )

    return {"completed_topics": [topic_summary.content]}


@traceable
def synthesizer(state: State):
    """Synthesizer that organizes summaries into a structured learning roadmap."""
    topic_summaries = state["completed_topics"]
    learning_roadmap = "\n\n---\n\n".join(topic_summaries)

    return {"learning_roadmap": learning_roadmap}


# Conditional edge function
def assign_workers(state: State):
    """Assign a worker to each topic in the plan."""
    return [Send("llm_call", {"topic": t}) for t in state["topics"]]


# Build workflow
learning_path_builder = StateGraph(State)

# Add the nodes to workflow
learning_path_builder.add_node("orchestrator", orchestrator)
learning_path_builder.add_node("llm_call", llm_call)
learning_path_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
learning_path_builder.add_edge(START, "orchestrator")
learning_path_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
learning_path_builder.add_edge("llm_call", "synthesizer")
learning_path_builder.add_edge("synthesizer", END)

# Compile the workflow
learning_path_workflow = learning_path_builder.compile()

# Show the workflow
try:
    display(Image(learning_path_workflow.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Error generating Mermaid diagram: {e}")

# Invoke
user_skills = "Python programming, basic machine learning concepts"
user_goals = "Learn advanced AI, master prompt engineering, and build AI applications"

state = learning_path_workflow.invoke(
    {"user_skills": user_skills, "user_goals": user_goals}
)

# Display the final roadmap
Markdown(state["learning_roadmap"])
