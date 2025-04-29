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

# Initialize colorama
init(autoreset=True)

# Check if GPU is available for potential acceleration
has_gpu = torch.cuda.is_available()

# Initialize NLP pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(f"{Fore.YELLOW}[ore] Installing required language model...")
    import subprocess
    subprocess.run([
        "python", "-m", "spacy", "download", "en_core_web_sm"
    ], check=True)
    nlp = spacy.load("en_core_web_sm")

class OResearch:
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), ".data")
        self.kb_path = os.path.join(self.data_dir, "knowledge_base.json")
        self.user_path = os.path.join(self.data_dir, "user_data.json")
        self.vectorizer = TfidfVectorizer()
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load knowledge base and user data
        self.kb = self.load_kb()
        self.user = self.load_user()
        
        # Check connection status
        self.offline = not self.is_net()
        
        # Info templates
        self.info_templates = [
            "I'm OResearch — a lightweight, offline-capable research assistant. I combine NLP, search algorithms, and custom prediction models.",
            "My name is OResearch, designed to be your research companion with or without internet access. I use advanced language processing.",
            "OResearch here! I'm a research assistant built to extract and summarize information using AI techniques. I work both online and offline.",
            "I'm an AI research tool called OResearch. I can search online sources when connected, or use my built-in knowledge base when offline.",
            "OResearch at your service! I'm a dual-mode research assistant that uses NLP and machine learning to answer your questions."
        ]
        
        # Tutorial templates for step-by-step guides
        self.tutorial_templates = [
            "Here's a step-by-step guide on {}:",
            "Let me walk you through the process of {}:",
            "Follow these steps to {}:",
            "Here's my detailed tutorial on {}:",
            "A comprehensive guide to {}:"
        ]
        
        # Show startup message
        self.hello()
        
        # Initialize session stats
        self.stats = {
            "queries": 0,
            "online": 0,
            "offline": 0,
            "start_time": datetime.now()
        }
        
        # Load embeddings model if available
        self.embed_model = None
        try:
            if has_gpu:
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"{Fore.GREEN}[ore] Enhanced embeddings model loaded {Fore.CYAN}(GPU accelerated)")
            else:
                print(f"{Fore.YELLOW}[ore] Running in basic mode {Fore.RED}(No GPU detected)")
        except ImportError:
            print(f"{Fore.YELLOW}[ore] Running in basic mode {Fore.RED}(Missing sentence-transformers package)")

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
        print(f"\n{Style.BRIGHT}{Fore.CYAN}" + "~" * 50)
        print(f"{Fore.CYAN}{fish}")
        print(f"{Style.BRIGHT}{Fore.WHITE}🔍 {Fore.CYAN}OResearch v2.1{Fore.WHITE} - Advanced Research Assistant")
        print(f"{Fore.CYAN}" + "~" * 50)
        print(f"{Fore.GREEN}[ore] Starting up... {Fore.YELLOW}I'm not perfect, be patient with me")
        if self.offline:
            print(f"{Fore.RED}[ore] ⚠️  Running in OFFLINE mode - using local knowledge only")
        print(f"{Fore.CYAN}" + "~" * 50 + "\n")

    def is_net(self) -> bool:
        """Check internet connectivity"""
        try:
            # Try multiple DNS servers for reliability
            for dns in ["8.8.8.8", "1.1.1.1"]:
                socket.create_connection((dns, 53), timeout=1)
                return True
        except OSError:
            return False

    def type(self, text: str, speed: float = 0.01):
        """Print text with typewriter effect"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(speed)
        print()

    def gen_ctx(self, text: str) -> str:
        """Extract meaningful context from user input using advanced NLP"""
        doc = nlp(text)
        
        # Extract entities first
        entities = [ent.text for ent in doc.ents]
        
        # Check for tutorial keywords
        is_tutorial = any(word in text.lower() for word in ["how to", "tutorial", "guide", "steps", "learn", "teach me"])
        
        # Advanced keyword extraction based on part-of-speech and dependencies
        keywords = []
        
        # Extract subject-verb-object triplets for richer context
        for token in doc:
            # Get main verbs and their direct objects
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                keywords.append(token.lemma_)
                # Look for the object of this verb
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"] and not child.is_stop:
                        keywords.append(child.text)
        
        # Extract important nouns and adjectives
        important_tokens = [token.lemma_ for token in doc if (
            token.pos_ in ('NOUN', 'PROPN', 'ADJ') and 
            not token.is_stop and
            len(token.text) > 2
        )]
        
        # Add key nouns and adjectives
        keywords.extend(important_tokens)
        
        # Combine entities and keywords, prioritizing entities
        all_context = entities + [k for k in keywords if k not in entities]
        
        # For tutorials, add special marker to signal tutorial format is needed
        if is_tutorial:
            all_context.insert(0, "TUTORIAL")
            
        # Filter duplicates while preserving order
        seen = set()
        filtered_context = []
        for item in all_context:
            if item.lower() not in seen:
                filtered_context.append(item)
                seen.add(item.lower())
        
        return " ".join(filtered_context)

    def load_kb(self) -> Dict:
        """Load knowledge base from file"""
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
            print(f"{Fore.RED}[ore] Error: Knowledge base file corrupted. Creating new one.")
            os.rename(self.kb_path, f"{self.kb_path}.bak")
            return {}

    def load_user(self) -> Dict:
        """Load user data and preferences"""
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
            print(f"{Fore.RED}[ore] Error: User data file corrupted. Creating new one.")
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
        """Save user data to file"""
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
        """Search for answers in offline knowledge base"""
        best_match = None
        best_score = 0
        
        # Check if this is a tutorial request
        is_tutorial = context.startswith("TUTORIAL")
        if is_tutorial:
            # Remove the TUTORIAL marker for matching
            context = context.replace("TUTORIAL", "", 1).strip()
        
        # Try direct key lookup first
        if context.lower() in self.kb:
            entry = self.kb[context.lower()]
            response = random.choice(entry.get("responses", ["No answer available."]))
            
            # Format as tutorial if needed
            if is_tutorial:
                return self.format_tutorial(context, response)
            return response
            
        # Otherwise, search through all questions
        for topic, entry in self.kb.items():
            questions = entry.get("question", [])
            
            # Calculate similarity scores
            for q in questions:
                # Simple substring match
                if context.lower() in q.lower() or q.lower() in context.lower():
                    score = len(set(context.lower().split()) & set(q.lower().split())) / max(len(context.split()), len(q.split()))
                    
                    if score > best_score:
                        best_score = score
                        best_match = entry
        
        # Return best match if score is above threshold
        if best_match and best_score > 0.3:
            response = random.choice(best_match.get("responses", ["No specific information available."]))
            
            # Format as tutorial if needed
            if is_tutorial:
                return self.format_tutorial(context, response)
            return response
        
        if is_tutorial:
            return f"Sorry, I couldn't find a tutorial on {context} in my offline knowledge base."
        return f"{Fore.YELLOW}Sorry, I couldn't find anything in my offline knowledge base."

    def format_tutorial(self, topic: str, content: str) -> str:
        """Format content as a step-by-step tutorial"""
        # Select a tutorial template
        template = random.choice(self.tutorial_templates).format(topic)
        
        # Split content into sentences for steps
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Build tutorial
        tutorial = f"{Fore.CYAN}{template}\n\n"
        
        # If we have enough content, format as steps
        if len(sentences) >= 3:
            for i, sentence in enumerate(sentences, 1):
                tutorial += f"{Fore.GREEN}{i}. {Fore.WHITE}{sentence}.\n"
        else:
            # Not enough content to make good steps, just use the original
            tutorial += f"{Fore.WHITE}{content}\n"
            
        return tutorial

    def is_about_me(self, text: str) -> bool:
        """Check if the query is about the bot itself"""
        # Normalized input for better matching
        norm_text = text.lower()
        
        # Bot-related keywords
        bot_keywords = [
            "who are you", "what are you", "your name", "what is oresearch", 
            "about oresearch", "tell me about yourself", "what can you do",
            "how do you work", "your capabilities", "what's oresearch",
            "who made you", "your creator", "your purpose", "introduce yourself"
        ]
        
        # Check for direct matches
        for keyword in bot_keywords:
            if keyword in norm_text:
                return True
                
        # More sophisticated check using NLP
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
        
        if self.embed_model:
            caps.append("use neural embeddings for semantic understanding")
        
        # Format response with capabilities
        caps_text = ", ".join(caps[:-1]) + f" and {caps[-1]}" if len(caps) > 1 else caps[0]
        
        response = f"{Fore.CYAN}{base}\n\n{Fore.GREEN}I can {caps_text}."
        
        # Add session stats for a more dynamic feel
        session_time = datetime.now() - self.stats["start_time"]
        minutes = int(session_time.total_seconds() / 60)
        
        if self.stats["queries"] > 0:
            response += f"\n\n{Fore.YELLOW}In our current session ({minutes} minutes), I've answered {self.stats['queries']} queries."
        
        return response

    def ddg_search(self, query: str) -> Optional[str]:
        """Search using DuckDuckGo"""
        try:
            response = requests.get(
                f"https://api.duckduckgo.com/",
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
            print(f"{Fore.RED}[ore] DuckDuckGo search error: {e}")
            return None

    def wiki_search(self, query: str) -> str:
        """Search using Wikipedia"""
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
            print(f"{Fore.RED}[ore] Wikipedia search error: {e}")
            return "Error retrieving information from Wikipedia."

    def analyze(self, articles: List[str], context: str) -> str:
        """Analyze and find most relevant article"""
        if not articles:
            return "No valid articles to analyze."
            
        # Use embeddings model if available for better semantic matching
        if self.embed_model:
            try:
                # Get embeddings
                query_embed = self.embed_model.encode([context])[0]
                article_embeds = self.embed_model.encode(articles)
                
                # Calculate cosine similarity
                similarities = cosine_similarity([query_embed], article_embeds)[0]
                
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.3:
                    return articles[best_idx]
                return "No highly relevant content found."
            except Exception as e:
                print(f"{Fore.RED}[ore] Embeddings error: {e}")
                # Fall back to TF-IDF if embeddings fail
        
        # TF-IDF fallback
        try:
            vectors = self.vectorizer.fit_transform([context] + articles)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            best_idx = np.argmax(sims)
            return articles[best_idx] if sims[best_idx] > 0.2 else "No relevant content found."
        except Exception as e:
            print(f"{Fore.RED}[ore] TF-IDF error: {e}")
            if articles:
                # Return first article if all else fails
                return articles[0]
            return "Error analyzing the content."

    def learn(self, query: str, response: str):
        """Learn from successful interactions to improve future responses"""
        # Extract keywords for categorization
        keywords = self.gen_ctx(query)
        
        # Check if this is a new knowledge entry
        if keywords and response and len(response) > 20:
            # Use the first keyword as a simple topic identifier
            topic = keywords.split()[0] if keywords.split() else keywords
            
            if topic not in self.kb:
                self.kb[topic] = {
                    "question": [query],
                    "responses": [response]
                }
            else:
                # Update existing knowledge
                if query not in self.kb[topic]["question"]:
                    self.kb[topic]["question"].append(query)
                if response not in self.kb[topic]["responses"]:
                    self.kb[topic]["responses"].append(response)
            
            # Save periodically
            if random.random() < 0.2:  # 20% chance to save
                self.save_kb()

    def net_search(self, query: str, context: str) -> Dict:
        """Search online using multiple sources in parallel"""
        results = {"content": None, "source": None}
        
        def duck_search():
            return {"content": self.ddg_search(query), "source": "DuckDuckGo"}
            
        def wiki_search():
            return {"content": self.wiki_search(context), "source": "Wikipedia"}
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            duck_future = executor.submit(duck_search)
            wiki_future = executor.submit(wiki_search)
            
            # Get DuckDuckGo results
            duck_result = duck_future.result()
            if duck_result["content"]:
                results = duck_result
            else:
                # Fall back to Wikipedia
                wiki_result = wiki_future.result()
                results = wiki_result
                
        # Check if result is for a tutorial request
        if context.startswith("TUTORIAL") and results["content"]:
            tutorial_topic = context.replace("TUTORIAL", "", 1).strip()
            results["content"] = self.format_tutorial(tutorial_topic, results["content"])
                
        return results

    def run(self):
        """Main interaction loop"""
        try:
            while True:
                user_input = input(f"\n{Fore.CYAN}You: {Fore.WHITE}")
                
                # Update stats
                self.stats["queries"] += 1
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.type(f"{Fore.GREEN}Thank you for using OResearch. Goodbye!")
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
                        print(f"{Fore.RED}[ore] ⚠️ Connection lost. Switching to offline mode.")
                    else:
                        print(f"{Fore.GREEN}[ore] ✓ Connection restored. Online search available.")
                
                # Handle offline mode
                if self.offline:
                    print(f"{Fore.YELLOW}[ore] Searching local knowledge base...")
                    self.stats["offline"] += 1
                    response = self.kb_search(context)
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "offline")
                    continue
                
                # Online search
                print(f"{Fore.GREEN}[ore] Searching online sources...")
                self.stats["online"] += 1
                
                # Show spinner on a separate thread
                stop_spinner = threading.Event()
                def spin():
                    spinner = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
                    i = 0
                    while not stop_spinner.is_set():
                        print(f"\r{Fore.CYAN}[ore] Searching {spinner[i % len(spinner)]}", end='')
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
                        print(f"{Fore.GREEN}[ore] Found information from {search_result['source']}:")
                        self.type(search_result["content"], speed=self.user["prefs"]["type_speed"])
                        
                        # Learn from this interaction
                        self.learn(user_input, search_result["content"])
                        
                        # Update history
                        self.add_hist(user_input, search_result["content"], search_result["source"])
                    else:
                        print(f"{Fore.YELLOW}[ore] No relevant information found online.")
                        self.type(f"{Fore.YELLOW}I couldn't find relevant information. Try rephrasing your question.")
                except Exception as e:
                    # Stop spinner in case of error
                    stop_spinner.set()
                    spinner_thread.join()
                    
                    print(f"{Fore.RED}[ore] Error during search: {e}")
                    self.type(f"{Fore.RED}Sorry, I encountered an error while searching. Please try again.")
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}[ore] Session interrupted. Saving data...")
            self.save_user()
            self.save_kb()
            print(f"{Fore.GREEN}[ore] Thank you for using OResearch. Goodbye!")

if __name__ == '__main__':
    bot = OResearch()
    bot.run()
     
