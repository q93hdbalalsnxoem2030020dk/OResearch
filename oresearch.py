import os
import json
import time
import socket
import requests
import wikipedia
import spacy
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Back, Style, init

# Initialize colorama with autoreset to ensure proper color handling
init(autoreset=True)

# Global variables for better organization
HAS_GPU = torch.cuda.is_available()
EMBED_MODEL = None

# Load NLP pipeline with better error handling
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print(f"[ore] Installing required language model...")
        import subprocess
        subprocess.run([
            "python", "-m", "spacy", "download", "en_core_web_sm"
        ], check=True)
        return spacy.load("en_core_web_sm")

# Initialize NLP pipeline
nlp = load_nlp()

class OResearch:
    def __init__(self):
        # Setup directories and paths
        self.data_dir = os.path.join(os.getcwd(), ".data")
        self.kb_path = os.path.join(self.data_dir, "knowledge_base.json")
        self.user_path = os.path.join(self.data_dir, "user_data.json")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize vectorizer for text similarity
        self.vectorizer = TfidfVectorizer()
        
        # Load data files
        self.kb = self.load_kb()
        self.user = self.load_user()
        
        # Check connection status
        self.offline = not self.is_net()
        
        # Initialize templates
        self.init_templates()
        
        # Initialize session stats
        self.stats = {
            "queries": 0,
            "online": 0,
            "offline": 0,
            "start_time": datetime.now()
        }
        
        # Show startup message
        self.hello()
        
        # Load embeddings model if available
        self.load_embed_model()

    def init_templates(self):
        """Initialize response templates"""
        # About me templates
        self.info_templates = [
            "I'm OResearch — a lightweight, offline-capable research assistant. I combine NLP, search algorithms, and custom prediction models.",
            "My name is OResearch, designed to be your research companion with or without internet access. I use advanced language processing.",
            "OResearch here! I'm a research assistant built to extract and summarize information using AI techniques. I work both online and offline.",
            "I'm an AI research tool called OResearch. I can search online sources when connected, or use my built-in knowledge base when offline.",
            "OResearch at your service! I'm a dual-mode research assistant that uses NLP and machine learning to answer your questions."
        ]
        
        # Tutorial templates
        self.tutorial_templates = [
            "Here's a step-by-step guide on {}:",
            "Let me walk you through the process of {}:",
            "Follow these steps to {}:",
            "Here's my detailed tutorial on {}:",
            "A comprehensive guide to {}:"
        ]

    def load_embed_model(self):
        """Load embeddings model if available"""
        global EMBED_MODEL
        try:
            if HAS_GPU:
                from sentence_transformers import SentenceTransformer
                EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"[ore] Enhanced embeddings model loaded (GPU accelerated)")
            else:
                print(f"[ore] Running in basic mode (No GPU detected)")
        except ImportError:
            print(f"[ore] Running in basic mode (Missing sentence-transformers package)")

    def hello(self):
        """Display startup message with improved formatting"""
        fish = r"""
               ,-,
             .(((^v
        ,---'\   - 
       /    ~ \___/
      /|      /|~|\
     | |     | |~||
     | |     | |~|'
     | |     | |
     | |     | |
    """
        print("\n" + "~" * 50)
        print(fish)
        print("🔍 OResearch v2.1 - Advanced Research Assistant")
        print("~" * 50)
        print("[ore] Starting up... I'm not perfect, be patient with me")
        if self.offline:
            print("[ore] ⚠️  Running in OFFLINE mode - using local knowledge only")
        print("~" * 50 + "\n")

    def is_net(self) -> bool:
        """Check internet connectivity more efficiently"""
        try:
            # Try multiple DNS servers for reliability
            for dns in ["8.8.8.8", "1.1.1.1"]:
                socket.create_connection((dns, 53), timeout=1)
                return True
        except OSError:
            return False
        return False

    def type(self, text: str, speed: float = 0.01):
        """Print text with typewriter effect"""
        # Remove color codes for length calculation
        clean_text = text
        for color in vars(Fore).values():
            if isinstance(color, str):
                clean_text = clean_text.replace(color, '')
        
        for char in text:
            print(char, end='', flush=True)
            time.sleep(speed)
        print()

    def gen_ctx(self, text: str) -> str:
        """Extract meaningful context from user input - IMPROVED ALGORITHM"""
        doc = nlp(text)
        
        # IMPROVED: Weighted token extraction for better relevance
        context_elements = []
        
        # Check for tutorial intent with broader matching
        tutorial_phrases = ["how to", "tutorial", "guide", "steps", "learn", 
                          "teach me", "explain how", "instructions for", "procedure"]
        is_tutorial = any(phrase in text.lower() for phrase in tutorial_phrases)
        
        # Process named entities with higher weight
        entities = [(ent.text.lower(), 3.0) for ent in doc.ents]
        entity_texts = [e[0] for e in entities]
        
        # Extract topical nouns (subjects and objects)
        nouns = []
        for token in doc:
            # Get important noun phrases with syntactic role
            if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 2:
                weight = 1.0
                
                # Increase weight for subjects and objects
                if token.dep_ in ('nsubj', 'dobj', 'pobj'):
                    weight = 2.0
                    
                # Only add if not already covered by entities
                if token.text.lower() not in entity_texts:
                    nouns.append((token.text.lower(), weight))
        
        # Extract main verbs and relevant adjectives
        verbs_and_adjs = []
        for token in doc:
            # Main verbs (not auxiliaries)
            if token.pos_ == 'VERB' and token.dep_ in ('ROOT', 'xcomp') and not token.is_stop:
                verbs_and_adjs.append((token.lemma_, 1.5))
            
            # Descriptive adjectives
            elif token.pos_ == 'ADJ' and len(token.text) > 2:
                # Check if it modifies an important noun
                if token.head.pos_ in ('NOUN', 'PROPN'):
                    verbs_and_adjs.append((token.text.lower(), 1.0))
        
        # Combine all elements with weights
        all_elements = entities + nouns + verbs_and_adjs
        
        # Sort by weight (descending)
        all_elements.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the text values
        context_elements = [item[0] for item in all_elements]
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_context = []
        for item in context_elements:
            if item not in seen:
                filtered_context.append(item)
                seen.add(item)
        
        # Add tutorial marker if needed
        if is_tutorial and filtered_context:
            return "TUTORIAL " + " ".join(filtered_context)
        
        return " ".join(filtered_context)

    def load_kb(self) -> Dict:
        """Load knowledge base from file with error handling"""
        if not os.path.exists(self.kb_path):
            default_kb = {
                "oresearch": {
                    "question": ["who are you", "what are you", "what is oresearch"],
                    "responses": [
                        "I'm OResearch, a lightweight research assistant designed to provide information both online and offline."
                    ]
                }
            }
            with open(self.kb_path, "w") as file:
                json.dump(default_kb, file, indent=2)
            return default_kb
        
        try:
            with open(self.kb_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("[ore] Error: Knowledge base file corrupted. Creating new one.")
            os.rename(self.kb_path, f"{self.kb_path}.bak")
            return {}

    def load_user(self) -> Dict:
        """Load user data and preferences with error handling"""
        if not os.path.exists(self.user_path):
            default_user = {
                "history": [],
                "prefs": {
                    "type_speed": 0.01,
                    "max_history": 100
                }
            }
            with open(self.user_path, "w") as file:
                json.dump(default_user, file, indent=2)
            return default_user
        
        try:
            with open(self.user_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("[ore] Error: User data file corrupted. Creating new one.")
            os.rename(self.user_path, f"{self.user_path}.bak")
            return {
                "history": [],
                "prefs": {
                    "type_speed": 0.01,
                    "max_history": 100
                }
            }

    def save_kb(self):
        """Save knowledge base to file"""
        with open(self.kb_path, "w") as file:
            json.dump(self.kb, file, indent=2)

    def save_user(self):
        """Save user data to file with history management"""
        # Limit history size
        max_history = self.user["prefs"]["max_history"]
        if len(self.user["history"]) > max_history:
            self.user["history"] = self.user["history"][-max_history:]
        
        with open(self.user_path, "w") as file:
            json.dump(self.user, file, indent=2)

    def add_hist(self, query: str, response: str, source: str):
        """Update search history"""
        self.user["history"].append({
            "time": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "source": source
        })
        
        # Save periodically (every 5 queries)
        if len(self.user["history"]) % 5 == 0:
            self.save_user()

    def kb_search(self, context: str) -> str:
        """Improved search for answers in offline knowledge base"""
        is_tutorial = context.startswith("TUTORIAL")
        search_context = context.replace("TUTORIAL", "", 1).strip() if is_tutorial else context
        
        # Try direct key lookup first (case insensitive)
        for key, entry in self.kb.items():
            if key.lower() == search_context.lower():
                response = random.choice(entry.get("responses", ["No answer available."]))
                return self.format_tutorial(search_context, response) if is_tutorial else response
        
        # Search by similarity
        best_match = None
        best_score = 0
        
        # Prepare context tokens for better matching
        context_tokens = set(search_context.lower().split())
        
        # Search through all questions
        for topic, entry in self.kb.items():
            questions = entry.get("question", [])
            
            # Calculate similarity scores for each question
            for q in questions:
                q_tokens = set(q.lower().split())
                
                # Calculate overlap score
                if context_tokens and q_tokens:
                    # Jaccard similarity
                    overlap = len(context_tokens & q_tokens)
                    union = len(context_tokens | q_tokens)
                    score = overlap / union if union > 0 else 0
                    
                    # Boost score for substring matches
                    if search_context.lower() in q.lower() or q.lower() in search_context.lower():
                        score += 0.2
                        
                    if score > best_score:
                        best_score = score
                        best_match = entry
        
        # Return best match if score is above threshold
        if best_match and best_score > 0.25:
            response = random.choice(best_match.get("responses", ["No specific information available."]))
            return self.format_tutorial(search_context, response) if is_tutorial else response
        
        # No good match found
        if is_tutorial:
            return f"Sorry, I couldn't find a tutorial on {search_context} in my offline knowledge base."
        return "Sorry, I couldn't find anything in my offline knowledge base."

    def format_tutorial(self, topic: str, content: str) -> str:
        """Format content as a step-by-step tutorial"""
        # Select a tutorial template
        template = random.choice(self.tutorial_templates).format(topic)
        
        # Split content into sentences for steps
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Build tutorial
        tutorial = f"{template}\n\n"
        
        # If we have enough content, format as steps
        if len(sentences) >= 3:
            for i, sentence in enumerate(sentences, 1):
                tutorial += f"{i}. {sentence}.\n"
        else:
            # Not enough content to make good steps, just use the original
            tutorial += f"{content}\n"
            
        return tutorial

    def is_about_me(self, text: str) -> bool:
        """Improved check if the query is about the bot itself"""
        # Normalized input for better matching
        norm_text = text.lower()
        
        # Bot-related keywords - expanded list
        bot_keywords = [
            "who are you", "what are you", "your name", "what is oresearch", 
            "about oresearch", "tell me about yourself", "what can you do",
            "how do you work", "your capabilities", "what's oresearch",
            "who made you", "your creator", "your purpose", "introduce yourself",
            "tell me about oresearch", "what do you do", "your function", 
            "help me understand what you are", "describe yourself"
        ]
        
        # Check for direct matches
        for keyword in bot_keywords:
            if keyword in norm_text:
                return True
                
        # NLP-based detection
        doc = nlp(norm_text)
        
        # Check for second-person pronouns combined with question words
        has_you = any(token.text.lower() == "you" for token in doc)
        has_question = any(token.tag_ in ["WP", "WRB"] for token in doc)
        
        if has_you and has_question:
            return True
            
        return False

    def get_about(self) -> str:
        """Generate a varied response about the bot itself"""
        # Starting with a template
        base = random.choice(self.info_templates)
        
        # Add dynamic capabilities information
        caps = []
        if not self.offline:
            caps.append("search the internet for up-to-date information")
        caps.append("access my local knowledge base")
        caps.append("learn from our interactions to improve future responses")
        
        if EMBED_MODEL:
            caps.append("use neural embeddings for semantic understanding")
        
        # Format response with capabilities
        caps_text = ", ".join(caps[:-1]) + f" and {caps[-1]}" if len(caps) > 1 else caps[0]
        
        response = f"{base}\n\n"
        response += f"I can {caps_text}."
        
        # Add session stats for a more dynamic feel
        session_time = datetime.now() - self.stats["start_time"]
        minutes = int(session_time.total_seconds() / 60)
        
        if self.stats["queries"] > 0:
            response += f"\n\nIn our current session ({minutes} minutes), I've answered {self.stats['queries']} queries."
        
        return response

    def ddg_search(self, query: str) -> Optional[str]:
        """Search using DuckDuckGo"""
        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
                timeout=5
            )
            
            data = response.json()
            
            # Try to get abstract first
            if data.get("AbstractText"):
                return data.get("AbstractText")
                
            # Fall back to related topics
            if data.get("RelatedTopics"):
                topics = data.get("RelatedTopics")
                content = []
                
                # Extract text from topics
                for topic in topics[:3]:
                    if "Text" in topic:
                        content.append(topic["Text"])
                
                if content:
                    return "\n".join(content)
            
            return None
        except Exception as e:
            print(f"[ore] DuckDuckGo search error: {e}")
            return None

    def wiki_search(self, query: str) -> str:
        """Search using Wikipedia with better error handling"""
        try:
            # First try to get direct page
            try:
                return wikipedia.summary(query, sentences=3)
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation by picking the first option
                return wikipedia.summary(e.options[0], sentences=3)
            except wikipedia.exceptions.PageError:
                # Try search
                results = wikipedia.search(query)
                if results:
                    return wikipedia.summary(results[0], sentences=3)
                return "Wikipedia has no results for this query."
        except Exception as e:
            print(f"[ore] Wikipedia search error: {e}")
            return "Error retrieving information from Wikipedia."

    def analyze(self, articles: List[str], context: str) -> str:
        """Analyze and find most relevant article with improved semantic matching"""
        if not articles:
            return "No valid articles to analyze."
            
        # Use embeddings model if available
        if EMBED_MODEL:
            try:
                # Get embeddings
                query_embed = EMBED_MODEL.encode([context])[0]
                article_embeds = EMBED_MODEL.encode(articles)
                
                # Calculate cosine similarity
                similarities = cosine_similarity([query_embed], article_embeds)[0]
                
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.3:
                    return articles[best_idx]
                return "No highly relevant content found."
            except Exception as e:
                print(f"[ore] Embeddings error: {e}")
                # Fall back to TF-IDF
        
        # TF-IDF fallback with error handling
        try:
            vectors = self.vectorizer.fit_transform([context] + articles)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            best_idx = np.argmax(sims)
            return articles[best_idx] if sims[best_idx] > 0.2 else "No relevant content found."
        except Exception as e:
            print(f"[ore] TF-IDF error: {e}")
            # Return first article if all else fails
            return articles[0] if articles else "Error analyzing the content."

    def learn(self, query: str, response: str):
        """Improved learning algorithm for better knowledge acquisition"""
        # Only learn meaningful content
        if not query or not response or len(response) < 20:
            return
            
        # Extract keywords for better categorization
        keywords = self.gen_ctx(query)
        
        if not keywords:
            return
            
        # Create a better topic key - use first two keywords for more specificity
        key_words = keywords.split()
        if len(key_words) >= 2:
            topic = f"{key_words[0]}_{key_words[1]}"
        else:
            topic = key_words[0]
            
        # Create new entry or update existing one
        if topic not in self.kb:
            self.kb[topic] = {
                "question": [query],
                "responses": [response]
            }
        else:
            # Check similarity before adding
            questions = self.kb[topic]["question"]
            responses = self.kb[topic]["responses"]
            
            # Only add if not too similar to existing entries
            if not any(self.text_similarity(query, q) > 0.8 for q in questions):
                self.kb[topic]["question"].append(query)
                
            if not any(self.text_similarity(response, r) > 0.8 for r in responses):
                self.kb[topic]["responses"].append(response)
        
        # Save periodically with reduced frequency
        if random.random() < 0.2:  # 20% chance to save
            self.save_kb()

    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity for deduplication"""
        # Simple word overlap for speed
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def net_search(self, query: str, context: str) -> Dict:
        """Search online using multiple sources in parallel with improved coordination"""
        results = {"content": None, "source": None}
        
        def duck_search():
            return {"content": self.ddg_search(query), "source": "DuckDuckGo"}
            
        def wiki_search():
            # Use the most relevant part of the context for wiki search
            search_terms = context.split()[:3]  # First three keywords
            wiki_query = " ".join(search_terms) if search_terms else context
            return {"content": self.wiki_search(wiki_query), "source": "Wikipedia"}
        
        # Execute searches in parallel for efficiency
        with ThreadPoolExecutor(max_workers=2) as executor:
            duck_future = executor.submit(duck_search)
            wiki_future = executor.submit(wiki_search)
            
            # Get results and select best one
            duck_result = duck_future.result()
            wiki_result = wiki_future.result()
            
            # Prioritize DuckDuckGo if it has content
            if duck_result["content"]:
                results = duck_result
            elif wiki_result["content"]:
                results = wiki_result
            else:
                results = {"content": "I couldn't find relevant information from my sources.", "source": "None"}
                
        # Format tutorial if needed
        if context.startswith("TUTORIAL") and results["content"]:
            tutorial_topic = context.replace("TUTORIAL", "", 1).strip()
            results["content"] = self.format_tutorial(tutorial_topic, results["content"])
                
        return results

    def run(self):
        """Main interaction loop with improved UX"""
        try:
            while True:
                print()  # Add space for readability
                user_input = input("You: ")
                
                # Skip empty inputs
                if not user_input.strip():
                    continue
                
                # Update stats
                self.stats["queries"] += 1
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    self.type("Thank you for using OResearch. Goodbye!")
                    self.save_user()  # Save data before exiting
                    break
                
                # Generate context for processing
                context = self.gen_ctx(user_input)
                
                # Check if query is about the bot
                if self.is_about_me(user_input):
                    response = self.get_about()
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "bot-info")
                    continue
                
                # Check connection status
                if self.offline != (not self.is_net()):
                    self.offline = not self.is_net()
                    if self.offline:
                        print("[ore] ⚠️ Connection lost. Switching to offline mode.")
                    else:
                        print("[ore] ✓ Connection restored. Online search available.")
                
                # Handle offline mode
                if self.offline:
                    print("[ore] Searching local knowledge base...")
                    self.stats["offline"] += 1
                    response = self.kb_search(context)
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "offline")
                    continue
                
                # Online search
                print("[ore] Searching online sources...")
                self.stats["online"] += 1
                
                # Show spinner on a separate thread
                stop_spinner = threading.Event()
                def spin():
                    spinner = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
                    i = 0
                    while not stop_spinner.is_set():
                        print(f"\r[ore] Searching {spinner[i % len(spinner)]}", end='')
                        i += 1
                        time.sleep(0.1)
                    print("\r", end='')  # Clear spinner line
                
                spinner_thread = threading.Thread(target=spin)
                spinner_thread.start()
                
                try:
                    # Search online
                    search_result = self.net_search(user_input, context)
                    
                    # Stop spinner
                    stop_spinner.set()
                    spinner_thread.join()
                    
                    if search_result["content"]:
                        print(f"[ore] Found information from {search_result['source']}:")
                        self.type(search_result["content"], speed=self.user["prefs"]["type_speed"])
                        
                        # Learn from this interaction
                        self.learn(user_input, search_result["content"])
                        
                        # Update history
                        self.add_hist(user_input, search_result["content"], search_result["source"])
                    else:
                        print("[ore] No relevant information found online.")
                        self.type("I couldn't find relevant information. Try rephrasing your question.")
                except Exception as e:
                    # Stop spinner in case of error
                    stop_spinner.set()
                    if spinner_thread.is_alive():
                        spinner_thread.join()
                    
                    print(f"[ore] Error during search: {e}")
                    self.type("Sorry, I encountered an error while searching. Please try again.")
        
        except KeyboardInterrupt:
            print("\n[ore] Session interrupted. Saving data...")
            self.save_user()
            self.save_kb()
            print("[ore] Thank you for using OResearch. Goodbye!")

if __name__ == '__main__':
    bot = OResearch()
    bot.run()
