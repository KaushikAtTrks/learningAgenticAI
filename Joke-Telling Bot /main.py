from pydantic import BaseModel
from typing import List, Literal, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# JSON file path for storing jokes
JOKES_FILE = "jokes_history.json"

# Initialize LLM for joke generation
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,  # Higher temperature for more creative jokes
    api_key=os.getenv("GROQ_API_KEY")
)


class Joke(BaseModel):
    text: str
    category: str
    language: str = "en"
    timestamp: str = ""
	
class JokeState(BaseModel):
    jokes: Annotated[List[Joke], add] = []
    jokes_choice: Literal["n", "c", "l", "h", "q"] = "n"  # next, category, language, history, quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False

def load_jokes_from_json() -> List[Joke]:
    """Load jokes from JSON file"""
    if os.path.exists(JOKES_FILE):
        try:
            with open(JOKES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Joke(**joke) for joke in data]
        except Exception as e:
            print(f"Warning: Could not load jokes history: {str(e)}")
            return []
    return []

def save_jokes_to_json(jokes: List[Joke]):
    """Save jokes to JSON file"""
    try:
        with open(JOKES_FILE, 'w', encoding='utf-8') as f:
            json.dump([joke.model_dump() for joke in jokes], f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save jokes history: {str(e)}")

def get_joke(language: str = "en", category: str = "neutral") -> str:
    """Generate a joke using LLM based on category and language"""
    
    # Define category-specific prompts
    category_prompts = {
        "neutral": "Tell me a clean, family-friendly, funny joke. Just the joke, nothing else.",
        "chuck": "Tell me a Chuck Norris joke. Make it funny and legendary. Just the joke, nothing else.",
        "all": "Tell me any random funny joke from any category. Just the joke, nothing else."
    }
    
    # Language instructions
    language_instructions = {
        "en": "",
        "hi": " Tell the joke in Hindi.",
        "gj": " Tell the joke in Gujarati."
    }
    
    # Build the prompt
    base_prompt = category_prompts.get(category, category_prompts["neutral"])
    lang_instruction = language_instructions.get(language, "")
    full_prompt = base_prompt + lang_instruction
    
    try:
        response = llm.invoke(full_prompt)
        joke = response.content.strip()
        return joke
    except Exception as e:
        return f"Error generating joke: {str(e)}"

def show_menu(state: JokeState) -> dict:
    user_input = input("[n] Next  [c] Category  [l] Language  [h] History  [q] Quit\n> ").strip().lower()
    return {"jokes_choice": user_input}

def route_choice(state: JokeState) -> str:
    if state.jokes_choice == "n":
        return "fetch_joke"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "l":
        return "language_choice"
    elif state.jokes_choice == "h":
        return "show_history"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"

def fetch_joke(state: JokeState) -> dict:
    print("\n🎭 Generating joke...")
    joke_text = get_joke(language=state.language, category=state.category)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_joke = Joke(
        text=joke_text, 
        category=state.category,
        language=state.language,
        timestamp=timestamp
    )
    print(f"\n{'='*60}")
    print(f"Category: {state.category} | Language: {state.language}")
    print(f"{'='*60}")
    print(f"{joke_text}")
    print(f"{'='*60}\n")
    
    # Save to JSON after generating
    all_jokes = state.jokes + [new_joke]
    save_jokes_to_json(all_jokes)
    
    return {"jokes": [new_joke]}

def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    print("\nAvailable categories:")
    print("  [0] Neutral - Family-friendly jokes")
    print("  [1] Chuck - Chuck Norris jokes")
    print("  [2] All - Random jokes from any category")
    selection = int(input("Select category [0/1/2]: ").strip())
    selected_category = categories[selection] if 0 <= selection < len(categories) else "neutral"
    print(f"✓ Category changed to: {selected_category}\n")
    return {"category": selected_category}

def exit_bot(state: JokeState) -> dict:
    print("\n👋 Thanks for laughing with me! Goodbye!\n")
    return {"quit": True}

def language_choice(state: JokeState) -> dict:
    """Allow user to select language preference"""
    languages = {"en": "English", "hi": "Hindi", "gj": "Gujarati"}
    print("\nAvailable languages:")
    print("  [0] English")
    print("  [1] Hindi")
    print("  [2] Gujarati")
    selection = int(input("Select language [0/1/2]: ").strip())
    lang_codes = ["en", "hi", "gj"]
    selected_language = lang_codes[selection] if 0 <= selection < len(lang_codes) else "en"
    print(f"✓ Language changed to: {languages[selected_language]}\n")
    return {"language": selected_language}

def show_history(state: JokeState) -> dict:
    """Display all previously generated jokes"""
    if not state.jokes:
        print("\n📭 No jokes in history yet! Generate some jokes first.\n")
    else:
        print("\n" + "="*60)
        print("📚 JOKE HISTORY")
        print("="*60)
        for idx, joke in enumerate(state.jokes, 1):
            print(f"\n[{idx}] Category: {joke.category} | Language: {joke.language}")
            if joke.timestamp:
                print(f"    Time: {joke.timestamp}")
            print(f"{'-'*60}")
            print(f"{joke.text}")
            print(f"{'-'*60}")
        print(f"\nTotal jokes: {len(state.jokes)}")
        print("="*60 + "\n")
    return {}

def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("fetch_joke", fetch_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("language_choice", language_choice)
    workflow.add_node("show_history", show_history)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "fetch_joke",
            "update_category": "update_category",
            "language_choice": "language_choice",
            "show_history": "show_history",
            "exit_bot": "exit_bot",
        }
    )

    workflow.add_edge("fetch_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("language_choice", "show_menu")
    workflow.add_edge("show_history", "show_menu")
    workflow.add_edge("exit_bot", END)
    
    return workflow.compile()

def main():
    print("="*60)
    print("🎭 Welcome to the AI-Powered Joke-Telling Bot! 🎭")
    print("="*60)
    print("Powered by LangGraph and ChatGroq LLM\n")
    
    # Load previous jokes from JSON
    previous_jokes = load_jokes_from_json()
    if previous_jokes:
        print(f"📚 Loaded {len(previous_jokes)} jokes from previous sessions\n")
    
    graph = build_joke_graph()
    initial_state = JokeState(jokes=previous_jokes)
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

if __name__ == "__main__":
    main()
