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

# Check if GPU is available for potential acceleration
has_gpu = torch.cuda.is_available()

# Initialize NLP pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[oresearch] Installing required language model...")
    import subprocess
    subprocess.run([
        "python", "-m", "spacy", "download", "en_core_web_sm"
    ], check=True)
    nlp = spacy.load("en_core_web_sm")

class OResearch:
    def __init__(self):
        self.knowledge_dir = os.path.join(os.getcwd(), ".data")
        self.knowledge_path = os.path.join(self.knowledge_dir, "knowledge_base.json")
        self.user_data_path = os.path.join(self.knowledge_dir, "user_data.json")
        self.vectorizer = TfidfVectorizer()
        
        # Ensure data directory exists
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        # Load knowledge base and user data
        self.knowledge = self.load_knowledge()
        self.user_data = self.load_user_data()
        
        # Check connection status
        self.offline = not self.is_connected()
        
        # Auto-generator templates for bot information
        self.bot_info_templates = [
            "I'm OResearch — a lightweight, offline-capable research assistant. I combine NLP, search algorithms, and custom prediction models to provide information.",
            "My name is OResearch, designed to be your research companion with or without internet access. I use advanced language processing to understand your queries.",
            "OResearch here! I'm a research assistant built to extract and summarize information using AI techniques. I work both online and offline.",
            "I'm an AI research tool called OResearch. I can search online sources when connected, or use my built-in knowledge base when offline.",
            "OResearch at your service! I'm a dual-mode research assistant that uses NLP and machine learning to answer your questions with relevant information."
        ]
        
        # Show startup message
        self.say_startup_message()
        
        # Initialize session stats
        self.session_stats = {
            "queries": 0,
            "online_searches": 0,
            "offline_searches": 0,
            "start_time": datetime.now()
        }
        
        # Load embeddings model if available
        self.embeddings_model = None
        try:
            if has_gpu:
                from sentence_transformers import SentenceTransformer
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[oresearch] Enhanced embeddings model loaded (GPU accelerated)")
            else:
                print("[oresearch] Running in basic mode (No GPU detected)")
        except ImportError:
            print("[oresearch] Running in basic mode (Missing sentence-transformers package)")

    def say_startup_message(self):
        """Display startup message with improved formatting"""
        print("\n" + "=" * 60)
        print("🔍 OResearch v2.0 - Advanced Research Assistant")
        print("=" * 60)
        print("[oresearch] Starting up... I'm not perfect, so it's understandable")
        print("[oresearch] if the output isn't exactly what you expected.")
        if self.offline:
            print("[oresearch] ⚠️ Running in OFFLINE mode - using local knowledge only")
        print("=" * 60 + "\n")

    def is_connected(self) -> bool:
        """Check internet connectivity"""
        try:
            # Try multiple DNS servers for reliability
            for dns in ["8.8.8.8", "1.1.1.1"]:
                socket.create_connection((dns, 53), timeout=1)
                return True
        except OSError:
            return False

    def typewriter(self, text: str, speed: float = 0.01):
        """Print text with typewriter effect"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(speed)
        print()

    def generate_context(self, user_input: str) -> str:
        """Extract meaningful context from user input"""
        doc = nlp(user_input)
        
        # Extract entities first
        entities = [ent.text for ent in doc.ents]
        
        # Extract keywords from tokens
        keywords = [token.lemma_ for token in doc if (
            token.pos_ in ('NOUN', 'VERB', 'PROPN', 'ADJ') and 
            not token.is_stop and
            len(token.text) > 2
        )]
        
        # Combine entities and keywords
        context_parts = entities + keywords
        return " ".join(context_parts)

    def load_knowledge(self) -> Dict:
        """Load knowledge base from file"""
        if not os.path.exists(self.knowledge_path):
            default_knowledge = {
                "oresearch": {
                    "question": ["who are you", "what are you", "what is oresearch"],
                    "responses": [
                        "I'm OResearch, a lightweight research assistant designed to provide information both online and offline."
                    ]
                }
            }
            with open(self.knowledge_path, "w") as file:
                json.dump(default_knowledge, file, indent=2)
            return default_knowledge
        
        try:
            with open(self.knowledge_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("[oresearch] Error: Knowledge base file corrupted. Creating new one.")
            os.rename(self.knowledge_path, f"{self.knowledge_path}.bak")
            return {}

    def load_user_data(self) -> Dict:
        """Load user data and preferences"""
        if not os.path.exists(self.user_data_path):
            default_user_data = {
                "search_history": [],
                "preferences": {
                    "typewriter_speed": 0.01,
                    "max_history": 100
                }
            }
            with open(self.user_data_path, "w") as file:
                json.dump(default_user_data, file, indent=2)
            return default_user_data
        
        try:
            with open(self.user_data_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("[oresearch] Error: User data file corrupted. Creating new one.")
            os.rename(self.user_data_path, f"{self.user_data_path}.bak")
            return {
                "search_history": [],
                "preferences": {
                    "typewriter_speed": 0.01,
                    "max_history": 100
                }
            }

    def save_knowledge(self):
        """Save knowledge base to file"""
        with open(self.knowledge_path, "w") as file:
            json.dump(self.knowledge, file, indent=2)

    def save_user_data(self):
        """Save user data to file"""
        # Limit history size
        max_history = self.user_data["preferences"]["max_history"]
        if len(self.user_data["search_history"]) > max_history:
            self.user_data["search_history"] = self.user_data["search_history"][-max_history:]
        
        with open(self.user_data_path, "w") as file:
            json.dump(self.user_data, file, indent=2)

    def update_history(self, query: str, response: str, source: str):
        """Update search history"""
        self.user_data["search_history"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "source": source
        })
        
        # Save periodically (every 5 queries)
        if len(self.user_data["search_history"]) % 5 == 0:
            self.save_user_data()

    def offline_lookup(self, context: str) -> str:
        """Search for answers in offline knowledge base"""
        best_match = None
        best_score = 0
        
        # Try direct key lookup first
        if context.lower() in self.knowledge:
            entry = self.knowledge[context.lower()]
            return random.choice(entry.get("responses", ["No answer available."]))
            
        # Otherwise, search through all questions
        for topic, entry in self.knowledge.items():
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
            return random.choice(best_match.get("responses", ["No specific information available."]))
        
        return "Sorry, I couldn't find anything in my offline knowledge base."

    def is_about_bot(self, user_input: str) -> bool:
        """Check if the query is about the bot itself"""
        # Normalized input for better matching
        normalized_input = user_input.lower()
        
        # More comprehensive bot-related keywords
        bot_keywords = [
            "who are you", "what are you", "your name", "what is oresearch", 
            "about oresearch", "tell me about yourself", "what can you do",
            "how do you work", "your capabilities", "what's oresearch",
            "who made you", "your creator", "your purpose", "introduce yourself"
        ]
        
        # Check for direct matches
        for keyword in bot_keywords:
            if keyword in normalized_input:
                return True
                
        # More sophisticated check using NLP
        doc = nlp(normalized_input)
        
        # Check for second-person pronouns combined with question words
        has_you = any(token.text.lower() == "you" for token in doc)
        has_question = any(token.tag_ in ["WP", "WRB"] for token in doc)
        
        if has_you and has_question:
            return True
            
        return False

    def generate_about_response(self) -> str:
        """Generate a varied response about the bot itself"""
        # Starting with a template
        base_response = random.choice(self.bot_info_templates)
        
        # Add dynamic capabilities information
        capabilities = []
        if not self.offline:
            capabilities.append("search the internet for up-to-date information")
        capabilities.append("access my local knowledge base")
        capabilities.append("learn from our interactions to improve future responses")
        
        if self.embeddings_model:
            capabilities.append("use advanced semantic matching for better understanding")
        
        # Format response with capabilities
        capabilities_text = ", ".join(capabilities[:-1]) + f" and {capabilities[-1]}" if len(capabilities) > 1 else capabilities[0]
        
        response = f"{base_response}\n\nI can {capabilities_text}."
        
        # Add session stats for a more dynamic feel
        session_duration = datetime.now() - self.session_stats["start_time"]
        minutes = int(session_duration.total_seconds() / 60)
        
        if self.session_stats["queries"] > 0:
            response += f"\n\nIn our current session ({minutes} minutes), I've answered {self.session_stats['queries']} queries."
        
        return response

    def search_duckduckgo(self, query: str) -> Optional[str]:
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
            print(f"[oresearch] DuckDuckGo search error: {e}")
            return None

    def fallback_wikipedia(self, query: str) -> str:
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
            print(f"[oresearch] Wikipedia search error: {e}")
            return "Error retrieving information from Wikipedia."

    def analyze_articles(self, articles: List[str], context: str) -> str:
        """Analyze and find most relevant article"""
        if not articles:
            return "No valid articles to analyze."
            
        # Use embeddings model if available for better semantic matching
        if self.embeddings_model:
            try:
                # Get embeddings
                query_embedding = self.embeddings_model.encode([context])[0]
                article_embeddings = self.embeddings_model.encode(articles)
                
                # Calculate cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity([query_embedding], article_embeddings)[0]
                
                best_index = np.argmax(similarities)
                if similarities[best_index] > 0.3:
                    return articles[best_index]
                return "No highly relevant content found."
            except Exception as e:
                print(f"[oresearch] Embeddings error: {e}")
                # Fall back to TF-IDF if embeddings fail
        
        # TF-IDF fallback
        try:
            vectors = self.vectorizer.fit_transform([context] + articles)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            best_index = np.argmax(sims)
            return articles[best_index] if sims[best_index] > 0.2 else "No relevant content found."
        except Exception as e:
            print(f"[oresearch] TF-IDF error: {e}")
            if articles:
                # Return first article if all else fails
                return articles[0]
            return "Error analyzing the content."

    def learn_from_interaction(self, query: str, response: str):
        """Learn from successful interactions to improve future responses"""
        # Extract keywords for categorization
        keywords = self.generate_context(query)
        
        # Check if this is a new knowledge entry
        if keywords and response and len(response) > 20:
            # Use the first keyword as a simple topic identifier
            topic = keywords.split()[0] if keywords.split() else keywords
            
            if topic not in self.knowledge:
                self.knowledge[topic] = {
                    "question": [query],
                    "responses": [response]
                }
            else:
                # Update existing knowledge
                if query not in self.knowledge[topic]["question"]:
                    self.knowledge[topic]["question"].append(query)
                if response not in self.knowledge[topic]["responses"]:
                    self.knowledge[topic]["responses"].append(response)
            
            # Save periodically
            if random.random() < 0.2:  # 20% chance to save
                self.save_knowledge()

    def search_online(self, query: str, context: str) -> Dict:
        """Search online using multiple sources in parallel"""
        results = {"content": None, "source": None}
        
        def duck_search():
            return {"content": self.search_duckduckgo(query), "source": "DuckDuckGo"}
            
        def wiki_search():
            return {"content": self.fallback_wikipedia(context), "source": "Wikipedia"}
        
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
                
        return results

    def run(self):
        """Main interaction loop"""
        try:
            while True:
                user_input = input("\nYou: ")
                
                # Update stats
                self.session_stats["queries"] += 1
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.typewriter("Thank you for using OResearch. Goodbye!")
                    self.save_user_data()  # Save data before exiting
                    break
                
                # Generate context for processing
                context = self.generate_context(user_input)
                
                # Check if query is about the bot
                if self.is_about_bot(user_input):
                    response = self.generate_about_response()
                    self.typewriter(response, speed=self.user_data["preferences"]["typewriter_speed"])
                    self.update_history(user_input, response, "bot-info")
                    continue
                
                # Check connection status
                if self.offline != (not self.is_connected()):
                    self.offline = not self.is_connected()
                    if self.offline:
                        print("[oresearch] ⚠️ Connection lost. Switching to offline mode.")
                    else:
                        print("[oresearch] ✓ Connection restored. Online search available.")
                
                # Handle offline mode
                if self.offline:
                    print("[oresearch] Searching local knowledge base...")
                    self.session_stats["offline_searches"] += 1
                    response = self.offline_lookup(context)
                    self.typewriter(response, speed=self.user_data["preferences"]["typewriter_speed"])
                    self.update_history(user_input, response, "offline")
                    continue
                
                # Online search
                print("[oresearch] Searching online sources...")
                self.session_stats["online_searches"] += 1
                
                # Show spinner on a separate thread
                stop_spinner = threading.Event()
                def spin():
                    spinner = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
                    i = 0
                    while not stop_spinner.is_set():
                        print(f"\r[oresearch] Searching {spinner[i % len(spinner)]}", end='')
                        i += 1
                        time.sleep(0.1)
                    print("\r", end='')  # Clear spinner line
                
                spinner_thread = threading.Thread(target=spin)
                spinner_thread.start()
                
                try:
                    # Search online
                    search_result = self.search_online(user_input, context)
                    
                    # Stop spinner
                    stop_spinner.set()
                    spinner_thread.join()
                    
                    if search_result["content"]:
                        print(f"[oresearch] Found information from {search_result['source']}:")
                        self.typewriter(search_result["content"], speed=self.user_data["preferences"]["typewriter_speed"])
                        
                        # Learn from this interaction
                        self.learn_from_interaction(user_input, search_result["content"])
                        
                        # Update history
                        self.update_history(user_input, search_result["content"], search_result["source"])
                    else:
                        print("[oresearch] No relevant information found online.")
                        self.typewriter("I couldn't find relevant information. Try rephrasing your question.")
                except Exception as e:
                    # Stop spinner in case of error
                    stop_spinner.set()
                    spinner_thread.join()
                    
                    print(f"[oresearch] Error during search: {e}")
                    self.typewriter("Sorry, I encountered an error while searching. Please try again.")
        
        except KeyboardInterrupt:
            print("\n[oresearch] Session interrupted. Saving data...")
            self.save_user_data()
            self.save_knowledge()
            print("[oresearch] Thank you for using OResearch. Goodbye!")

if __name__ == '__main__':
    bot = OResearch()
    bot.run()
