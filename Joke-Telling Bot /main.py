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
    current_joke: str = ""  # Temporary storage for joke being reviewed
    critic_approved: bool = False
    revision_count: int = 0  # Track how many times Writer revised

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

def writer_agent(state: JokeState) -> dict:
    """Writer agent that generates jokes"""
    if state.revision_count > 0:
        print(f"\n✍️  Writer: Okay, let me try again (Attempt #{state.revision_count + 1})...")
    else:
        print("\n✍️  Writer: Crafting a joke for you...")
    
    joke_text = get_joke(language=state.language, category=state.category)
    
    return {
        "current_joke": joke_text,
        "revision_count": state.revision_count + 1,
        "critic_approved": False
    }

def critic_agent(state: JokeState) -> dict:
    """Critic agent that evaluates jokes"""
    print("\n🎭 Critic: Let me evaluate this joke...")
    
    # Build critic prompt
    critic_prompt = f"""You are a professional comedy critic. Evaluate this joke:

"{state.current_joke}"

Rate it on:
1. Humor (is it funny?)
2. Appropriateness (is it clean and suitable for all audiences?)
3. Structure (does it have good setup and punchline?)

Respond with ONLY one of these:
- "APPROVED" if the joke is good enough
- "REJECTED: [brief reason]" if it needs improvement

Be strict but fair. Only approve genuinely funny jokes."""
    
    try:
        response = llm.invoke(critic_prompt)
        evaluation = response.content.strip()
        
        if "APPROVED" in evaluation.upper():
            print("✅ Critic: This joke passes! It's ready to share.")
            return {"critic_approved": True}
        else:
            print(f"❌ Critic: {evaluation}")
            print("   Sending back to Writer for revision...")
            return {"critic_approved": False}
    except Exception as e:
        print(f"⚠️  Critic error: {str(e)}. Approving by default.")
        return {"critic_approved": True}

def route_critic_decision(state: JokeState) -> str:
    """Route based on critic's decision"""
    max_attempts = 3
    
    if state.critic_approved:
        return "show_final_joke"
    elif state.revision_count >= max_attempts:
        print(f"\n⚠️  Maximum attempts ({max_attempts}) reached. Using current joke anyway.")
        return "show_final_joke"
    else:
        return "writer"

def show_final_joke(state: JokeState) -> dict:
    """Display the approved joke and save it"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_joke = Joke(
        text=state.current_joke, 
        category=state.category,
        language=state.language,
        timestamp=timestamp
    )
    
    print(f"\n{'='*60}")
    print(f"✨ APPROVED JOKE ✨")
    print(f"Category: {state.category} | Language: {state.language}")
    print(f"Attempts: {state.revision_count}")
    print(f"{'='*60}")
    print(f"{state.current_joke}")
    print(f"{'='*60}\n")
    
    # Save to JSON after generating
    all_jokes = state.jokes + [new_joke]
    save_jokes_to_json(all_jokes)
    
    # Reset state for next joke
    return {
        "jokes": [new_joke],
        "current_joke": "",
        "revision_count": 0,
        "critic_approved": False
    }

def fetch_joke(state: JokeState) -> dict:
    """Legacy function - kept for compatibility"""
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

def show_menu(state: JokeState) -> dict:
    user_input = input("[n] Next  [c] Category  [l] Language  [h] History  [q] Quit\n> ").strip().lower()
    return {"jokes_choice": user_input}

def route_choice(state: JokeState) -> str:
    if state.jokes_choice == "n":
        return "writer"  # Now routes to Writer instead of fetch_joke
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "l":
        return "language_choice"
    elif state.jokes_choice == "h":
        return "show_history"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"

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

    # Add all nodes
    workflow.add_node("show_menu", show_menu)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("show_final_joke", show_final_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("language_choice", language_choice)
    workflow.add_node("show_history", show_history)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    # Menu routing
    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "writer": "writer",
            "update_category": "update_category",
            "language_choice": "language_choice",
            "show_history": "show_history",
            "exit_bot": "exit_bot",
        }
    )

    # Writer-Critic workflow
    workflow.add_edge("writer", "critic")  # Writer sends to Critic
    
    # Critic decision routing
    workflow.add_conditional_edges(
        "critic",
        route_critic_decision,
        {
            "writer": "writer",  # Rejected: back to Writer
            "show_final_joke": "show_final_joke",  # Approved: show joke
        }
    )

    # After showing joke, return to menu
    workflow.add_edge("show_final_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("language_choice", "show_menu")
    workflow.add_edge("show_history", "show_menu")
    workflow.add_edge("exit_bot", END)
    
    return workflow.compile()

def main():
    print("="*60)
    print("🎭 Welcome to the AI-Powered Joke-Telling Bot! 🎭")
    print("="*60)
    print("Powered by LangGraph and ChatGroq LLM")
    print("✨ Now featuring Writer-Critic AI Collaboration! ✨\n")
    
    # Load previous jokes from JSON
    previous_jokes = load_jokes_from_json()
    if previous_jokes:
        print(f"📚 Loaded {len(previous_jokes)} jokes from previous sessions\n")
    
    graph = build_joke_graph()
    initial_state = JokeState(jokes=previous_jokes)
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

if __name__ == "__main__":
    main()
