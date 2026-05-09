import os
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

load_dotenv()

# utilizing groq with openai/gpt-4o-20b for high-performance content generation;
llm = ChatGroq(model="openai/gpt-4o-20b")

class ContentState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    post: dict   # final structured output;
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def mock_searxng_search(query: str) -> str:
    """Provides simulated news headlines based on query keywords; to be replaced with a live SearXNG endpoint."""
    q = query.lower()
    if "crypto" in q or "bitcoin" in q:
        return "BREAKING: Bitcoin hits all-time high as institutional adoption surges."
    if "ai" in q or "model" in q or "developer" in q:
        return "OpenAI releases GPT-5 preview; developers debate the future of coding jobs."
    if "market" in q or "stock" in q or "interest rate" in q:
        return "Fed maintains interest rates; markets react with mixed signals."
    
    return "Tech industry faces new regulatory scrutiny amid global shifts."

tools = [mock_searxng_search]
llm_with_tools = llm.bind_tools(tools)

def decide_search_node(state: ContentState):
    """Determines an appropriate search query based on the bot's persona."""
    persona = state["persona"]
    prompt = f"You are a bot with this persona: {persona}. Based on your interests, what's a single search query to find news to comment on? Return just the query string."
    
    response = llm.invoke([HumanMessage(content=prompt)])
    query = response.content.strip().replace('"', '')
    
    return {"search_query": query}

def web_search_node(state: ContentState):
    """Executes the search query using the simulated search tool."""
    query = state["search_query"]
    results = mock_searxng_search.invoke(query)
    
    return {"search_results": results}

def draft_post_node(state: ContentState):
    """Generates a structured social media post based on the search results and bot persona."""
    persona = state["persona"]
    results = state["search_results"]
    bot_id = state["bot_id"]
    
    # explicit json instructions are provided to ensure structured output from the llm;
    prompt = f"""
    You are a bot with this persona: {persona}
    
    Search results for current events: {results}
    
    Write a 280-character opinionated post about this.
    
    YOU MUST RETURN ONLY A VALID JSON OBJECT WITH THESE KEYS:
    "bot_id": "{bot_id}",
    "topic": "the topic you found",
    "post_content": "the actual 280-char post"
    
    DO NOT INCLUDE ANY OTHER TEXT OR MARKDOWN.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    raw_content = response.content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:-3].strip()
    
    try:
        post_json = json.loads(raw_content)
    except Exception as e:
        # fallback mechanism in case of json parsing errors;
        post_json = {
            "bot_id": bot_id,
            "topic": "error",
            "post_content": f"Failed to parse JSON: {str(e)}. Raw: {raw_content[:100]}"
        }
    
    return {"post": post_json}

# graph workflow definition: start → decide_search → web_search → draft_post → end;
workflow = StateGraph(ContentState)
workflow.add_node("decide_search", decide_search_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("draft_post", draft_post_node)

workflow.add_edge(START, "decide_search")
workflow.add_edge("decide_search", "web_search")
workflow.add_edge("web_search", "draft_post")
workflow.add_edge("draft_post", END)

# persistent state management using sqlite;
memory = SqliteSaver.from_conn_string("chatbot.db")
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    bot_a_persona = "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
    
    initial_state = {
        "bot_id": "bot_a",
        "persona": bot_a_persona,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": "test_thread_1"}}
    
    print("Running Content Engine for Bot A...\n")
    final_state = graph.invoke(initial_state, config=config)
    
    print("Generated Post (JSON):")
    print(json.dumps(final_state["post"], indent=2))
