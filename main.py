import os
import random
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import  StateGraph, START, END
from langgraph.graph.message import add_messages
# from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import  BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# llm = init_chat_model(
#     #""
# )
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state["messages"][-1].content
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """ 
            Classify the users message as either:
            - 'emotional': If it asks for emotional support therapy deals with feelings or personal problems.
            - 'logical': If It asks for facts, information, logical analysis, or practical solution.
            """
        },
        {
            "role": "user",
            "content": last_message
        }
    ])

    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state["messages"][-1].content

    messages = [
        {
            "role": "system",
            "content": """You are a compassionate and empathetic therapist. 
            Your goal is to provide emotional support, understanding, and encouragement. 
            Focus on validating the user’s feelings, offering comfort, and guiding them gently 
            toward self-reflection and healing. 

            Avoid giving purely factual or logical answers. 
            Instead, respond with warmth, empathy, and thoughtful care, 
            as if you are a trusted listener who genuinely wants to help the user feel understood and supported.
            """
        },
        {
            "role": "user",
            "content": last_message
        }
    ]

    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1].content

    messages = [
        {
            "role": "system",
            "content": """You are a pure logical assistant.  
                Your role is to provide clear, concise, and factual answers.  
                Base all responses strictly on logic, reasoning, and verifiable information.  
                
                Do not address emotions, provide comfort, or offer emotional support.  
                Avoid empathetic or therapeutic language.  
                
                Be direct, precise, and evidence-driven in your responses, 
                focusing only on analysis, problem-solving, and practical solutions.  
            """
        },
        {
            "role": "user",
            "content": last_message
        }
    ]

    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    path_map={"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {
        "messages": [], "message_type": None
    }

    while True:
        user_input = input("Message: ")
        if user_input == 'exit':
            print(random.choice(["Bye...", "See ya...", "Nice talking to you..."]))
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()


# graph_builder = StateGraph(State)

# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}
#
# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)
#
# graph = graph_builder.compile()



# user_input = input("Enter a message: ")
# state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
#
# print(state["messages"][-1].content)
# print(state["messages"][-1])
