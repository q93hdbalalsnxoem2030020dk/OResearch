import os
import json
import time
import socket
import requests
import wikipedia
import spacy
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Back, Style, init

"""
Copyright (c) 2025 sxc_qq1 (ore)

This software (oresearch) is licensed for **personal, non-commercial use only**.
You are **not permitted** to:
- Modify, edit, or reverse-engineer any part of the code.
- Redistribute or publish any portion of the source code or its derivatives.
- Use this code for training, commercial deployment, or integration into any other software.

By using this script, you agree to these terms.

Violation of this license may result in permanent revocation of use, and legal actions may be taken where applicable.
For inquiries or special permission, contact: sunshinexjuhari@protonmail.com
"""

init(autoreset=True, convert=True)
has_gpu = torch.cuda.is_available()

# Enhanced color palette
C = {
    'TITLE': Fore.MAGENTA + Style.BRIGHT,
    'HEADER': Fore.CYAN + Style.BRIGHT,
    'TEXT': Fore.WHITE,
    'SUCCESS': Fore.GREEN + Style.BRIGHT,
    'INFO': Fore.CYAN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'ACCENT1': Fore.MAGENTA,
    'ACCENT2': Fore.BLUE + Style.BRIGHT,
    'ACCENT3': Fore.YELLOW + Style.BRIGHT,
    'RESET': Style.RESET_ALL,
    'STEP': Fore.GREEN + Style.BRIGHT
}

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(f"{C['WARNING']}[ore] Installing required language model...")
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
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.kb = self.load_kb()
        self.user = self.load_user()
        
        self.offline = not self.is_net()
        
        # Colorful templates for responses
        self.info_templates = [
            f"{C['ACCENT2']}I'm ore (Open Research Explorer) — a lightweight, AI-powered research assistant created by sxc_qq1 from Yx.GG Discord. I combine NLP, neural networks, and custom prediction models for optimal results.",
            f"{C['ACCENT1']}My name is ore (Open Research Explorer), designed by sxc_qq1 from Yx.GG Discord server to be your research companion with or without internet access. I use advanced language processing and PyTorch-powered neural networks.",
            f"{C['ACCENT3']}ore here! Made by sxc_qq1 from YxGG Discord, I'm a research assistant built with PyTorch to extract and summarize information using AI techniques. I work both online and offline.",
            f"{C['HEADER']}I'm an AI research tool called ore (Open Research Explorer), developed by sxc_qq1 from YxGG Discord. I can search online sources when connected, or use my PyTorch-powered knowledge base when offline.",
            f"{C['TITLE']}ore (Open Research Explorer) at your service! Created by sxc_qq1 from YxGG Discord, I'm a dual-mode research assistant using PyTorch and NLP to answer your questions efficiently."
        ]
        
        self.tutorial_templates = [
            f"{C['HEADER']}Here's a colorful step-by-step guide on {{}}:",
            f"{C['ACCENT1']}Let me walk you through the process of {{}}:",
            f"{C['ACCENT2']}Follow these vibrant steps to {{}}:",
            f"{C['ACCENT3']}Here's my detailed and colorful tutorial on {{}}:",
            f"{C['TITLE']}A rainbow-colored guide to {{}}:"
        ]
        
        self.hello()
        
        self.stats = {
            "queries": 0,
            "online": 0,
            "offline": 0,
            "start_time": datetime.now()
        }
        
        # PyTorch-based embedding model
        self.embed_model = None
        try:
            if has_gpu:
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"{C['SUCCESS']}[ore] Enhanced PyTorch embeddings model loaded {C['ACCENT3']}(GPU accelerated)")
            else:
                print(f"{C['WARNING']}[ore] Running in basic mode {C['ERROR']}(No GPU detected)")
        except ImportError:
            print(f"{C['WARNING']}[ore] Running in basic mode {C['ERROR']}(Missing sentence-transformers package)")

    def hello(self):
        print(f"\n{C['HEADER']}" + "~" * 50)
        print(f"{C['ACCENT3']}               ,-,")
        print(f"{C['ACCENT3']}             .(((^v")
        print(f"{C['ACCENT3']}        ,---'\\   - ")
        print(f"{C['ACCENT2']}       /    ~ \\___/")
        print(f"{C['ACCENT2']}      /|      /|~|\\")
        print(f"{C['ACCENT1']}     | |     | |~||")
        print(f"{C['ACCENT1']}     | |     | |~|'")
        print(f"{C['INFO']}     | |     | |")
        print(f"{C['INFO']}     | |     | |")
        print(f"{C['TITLE']}🔍 {C['ACCENT3']}ORE v2.2{C['TEXT']} - Open Research Explorer (ore){C['RESET']}")
        print(f"{C['HEADER']}" + "~" * 50)
        print(f"{C['SUCCESS']}[ore] Starting up... {C['WARNING']}I'm not perfect, be patient with me")
        if self.offline:
            print(f"{C['ERROR']}[ore] ⚠️ Running in OFFLINE mode - using local knowledge only")
            print(f"{C['HEADER']}" + "~" * 50 + "\n")
    
    def is_net(self) -> bool:
        """Check internet connectivity"""
        try:
            for dns in ["8.8.8.8", "1.1.1.1"]:
                socket.create_connection((dns, 53), timeout=1)
                return True            
        except OSError:
            return False

    def type(self, text: str, speed: float = 0.01):
        text_for_timing = text
        for color_code in list(C.values()):
            text_for_timing = text_for_timing.replace(color_code, "")
            
        for char in text:
            print(char, end='', flush=True)
            if char not in "\033[":
                time.sleep(speed)                
        print()

    def gen_ctx(self, text: str) -> str:
        if not text or not text.strip():
            return ""
            
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.text.strip()]
        
        is_tutorial = any(word in text.lower() for word in ["how to", "tutorial", "guide", "steps", "learn", "teach me"])
        keywords = []
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                keywords.append(token.lemma_)
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"] and not child.is_stop:
                        keywords.append(child.text)
        
        important_tokens = [token.lemma_ for token in doc if (
            token.pos_ in ('NOUN', 'PROPN', 'ADJ') and 
            not token.is_stop and
            len(token.text) > 2
        )]
        
        keywords.extend(important_tokens)
        all_context = entities + [k for k in keywords if k not in entities]

        if is_tutorial:
            all_context.insert(0, "TUTORIAL")
            
        seen = set()
        filtered_context = []
        for item in all_context:
            if item and item.lower() not in seen:
                filtered_context.append(item)
                seen.add(item.lower())
        
        return " ".join(filtered_context)

    def load_kb(self) -> Dict:
        if not os.path.exists(self.kb_path):
            default_kb = {
                "oresearch": {
                    "question": ["who are you", "what are you", "what is oresearch"],
                    "responses": [
                        "I'm OResearch also known as ore (Open Research Explorer) , a lightweight research assistant designed to provide information both online and offline."
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
            print(f"{C['ERROR']}[ore] Error: Knowledge base file corrupted. Creating new one.")
            os.rename(self.kb_path, f"{self.kb_path}.bak")
            return {}

    def load_user(self) -> Dict:
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
            print(f"{C['ERROR']}[ore] Error: User data file corrupted. Creating new one.")
            os.rename(self.user_path, f"{self.user_path}.bak")
            return {
                "history": [],
                "prefs": {
                    "type_speed": 0.01,
                    "max_history": 100
                }
            }

    def save_kb(self):
        with open(self.kb_path, "w") as file:
            json.dump(self.kb, file, indent=2)

    def save_user(self):
        max_history = self.user["prefs"]["max_history"]
        if len(self.user["history"]) > max_history:
            self.user["history"] = self.user["history"][-max_history:]
        
        with open(self.user_path, "w") as file:
            json.dump(self.user, file, indent=2)

    def add_hist(self, query: str, response: str, source: str):
        clean_response = response
        for color_code in list(C.values()):
            clean_response = clean_response.replace(color_code, "")
            
        self.user["history"].append({
            "time": datetime.now().isoformat(),
            "query": query,
            "response": clean_response,
            "source": source
        })

        if len(self.user["history"]) % 5 == 0:
            self.save_user()

    def kb_search(self, context: str) -> str:
        if not context:
            return f"{C['WARNING']}I need more specific information to search my knowledge base."
            
        best_match = None
        best_score = 0
        
        is_tutorial = context.startswith("TUTORIAL")
        if is_tutorial:
            context = context.replace("TUTORIAL", "", 1).strip()
        
        if context.lower() in self.kb:
            entry = self.kb[context.lower()]
            response = random.choice(entry.get("responses", ["No answer available."]))
            
            if is_tutorial:
                return self.format_tutorial(context, response)
            return response
            
        for topic, entry in self.kb.items():
            questions = entry.get("question", [])
            
            for q in questions:
                # substring match
                if context.lower() in q.lower() or q.lower() in context.lower():
                    score = len(set(context.lower().split()) & set(q.lower().split())) / max(len(context.split()), len(q.split()))
                    
                    if score > best_score:
                        best_score = score
                        best_match = entry
        
        if best_match and best_score > 0.3:
            response = random.choice(best_match.get("responses", ["No specific information available."]))
            
            if is_tutorial:
                return self.format_tutorial(context, response)
            return response
        
        if is_tutorial:
            return f"{C['INFO']}Sorry, I couldn't find a tutorial on {C['ACCENT2']}{context}{C['INFO']} in my offline knowledge base."
        return f"{C['WARNING']}Sorry, I couldn't find anything in my offline knowledge base."

    def format_tutorial(self, topic: str, content: str) -> str:
        if not topic or not content:
            return f"{C['WARNING']}I don't have enough information to create a tutorial."
            
        template = random.choice(self.tutorial_templates).format(topic)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        tutorial = f"{template}\n\n"
        if len(sentences) >= 3:
            for i, sentence in enumerate(sentences, 1):
                step_color = random.choice([C['STEP'], C['ACCENT1'], C['ACCENT2'], C['ACCENT3']])
                tutorial += f"{step_color}{i}. {C['TEXT']}{sentence}.\n"
        else:
            tutorial += f"{C['TEXT']}{content}\n"
            
        return tutorial

    def is_about_me(self, text: str) -> bool:
        """Improved detection of questions about ore and its capabilities"""
        if not text:
            return False
            
        # Normalize text
        norm_text = text.lower()
        
        # Direct identity questions
        bot_keywords = [
            "who are you", "what are you", "your name", "what is oresearch", 
            "about oresearch", "tell me about yourself", "what can you do",
            "how do you work", "your capabilities", "what's oresearch",
            "who made you", "your creator", "your purpose", "introduce yourself"
        ]
        
        for keyword in bot_keywords:
            if keyword in norm_text:
                return True
        
        capability_patterns = [
            "can you", "do you have", "are you able to", "could you", 
            "do you support", "are you capable of", "do you know how to",
            "are you equipped with", "do you use", "are you using"
        ]
        
        for pattern in capability_patterns:
            if pattern in norm_text:
                return True
                
        doc = nlp(norm_text)
        
        # Check for second-person pronouns in questions
        has_you = any(token.text.lower() in ["you", "your"] for token in doc)
        has_question = any(token.tag_ in ["WP", "WRB"] for token in doc) or norm_text.endswith("?")
        
        if has_you and has_question:
            return True
            
        return False

    def get_about(self) -> str:
        base = random.choice(self.info_templates)
        caps = []
        
        if not self.offline:
            caps.append(f"search the internet for up-to-date information")
        caps.append(f"access my local knowledge base")
        caps.append(f"learn from our interactions to improve future responses")
        
        if self.embed_model:
            caps.append(f"use PyTorch neural embeddings for semantic understanding")
            caps.append(f"process language with advanced NLP models")
        
        colored_caps = []
        colors = [C['SUCCESS'], C['ACCENT1'], C['ACCENT2'], C['ACCENT3']]
        for i, cap in enumerate(caps):
            colored_caps.append(f"{colors[i % len(colors)]}{cap}{C['TEXT']}")
            
        caps_text = ", ".join(colored_caps[:-1]) + f" and {colored_caps[-1]}" if len(colored_caps) > 1 else colored_caps[0]
        
        response = f"{base}\n\n{C['INFO']}I can {caps_text}."
        session_time = datetime.now() - self.stats["start_time"]
        minutes = int(session_time.total_seconds() / 60)
        
        if self.stats["queries"] > 0:
            response += f"\n\n{C['TITLE']}In our current session ({minutes} minutes), I've answered {C['ACCENT3']}{self.stats['queries']}{C['TITLE']} queries."
        
        return response

    def ddg_search(self, query: str) -> Optional[str]:
        if not query:
            return None
            
        try:
            response = requests.get(
                f"https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
                timeout=5
            )
            
            data = response.json()
            
            if data.get("AbstractText"):
                return data.get("AbstractText")
                
            if data.get("RelatedTopics"):
                topics = data.get("RelatedTopics")
                content = []
                
                for topic in topics[:3]:
                    if "Text" in topic:
                        content.append(topic["Text"])
                
                if content:
                    return "\n".join(content)
            
            return None
        except Exception as e:
            print(f"{C['ERROR']}[ore] DuckDuckGo search error: {e}")
            return None

    def wiki_search(self, query: str) -> str:
        if not query:
            return "Please provide a search term for Wikipedia."
            
        try:
            try:
                return wikipedia.summary(query, sentences=3)
            except wikipedia.exceptions.DisambiguationError as e:
                return wikipedia.summary(e.options[0], sentences=3)
                
            except wikipedia.exceptions.PageError:
                results = wikipedia.search(query)
                if results:
                    return wikipedia.summary(results[0], sentences=3)
                return "Wikipedia has no results for this query."
        except Exception as e:
            print(f"{C['ERROR']}[ore] Wikipedia search error: {e}")
            return "Error retrieving information from Wikipedia."

    def analyze(self, articles: List[str], context: str) -> str:
        """PyTorch-powered semantic analysis"""
        if not articles or not context:
            return "No valid articles to analyze."
            
        if self.embed_model:
            try:
                query_embed = self.embed_model.encode([context])[0]
                article_embeds = self.embed_model.encode(articles)
                
                # Using PyTorch F.cosine_similarity for better performance
                query_tensor = torch.tensor(query_embed).unsqueeze(0)
                articles_tensor = torch.tensor(article_embeds)
                
                similarities = F.cosine_similarity(query_tensor, articles_tensor)
                best_idx = torch.argmax(similarities).item()
                
                if similarities[best_idx] > 0.3:
                    return articles[best_idx]
                return "No highly relevant content found."
            except Exception as e:
                print(f"{C['ERROR']}[ore] PyTorch embeddings error: {e}")
                # Fall back to TF-IDF if embeddings fail
        
        # TF-IDF fallback
        try:
            vectors = self.vectorizer.fit_transform([context] + articles)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            best_idx = np.argmax(sims)
            return articles[best_idx] if sims[best_idx] > 0.2 else "No relevant content found."
        except Exception as e:
            print(f"{C['ERROR']}[ore] TF-IDF error: {e}")
            if articles:
                return articles[0]
            return "Error analyzing the content."

    def learn(self, query: str, response: str):
        if not query or not response:
            return
            
        keywords = self.gen_ctx(query)

        if keywords and response and len(response) > 20:
            clean_response = response
            for color_code in list(C.values()):
                clean_response = clean_response.replace(color_code, "")
            topic = keywords.split()[0] if keywords.split() else keywords
            
            if topic not in self.kb:
                self.kb[topic] = {
                    "question": [query],
                    "responses": [clean_response]
                }
            else:
                if query not in self.kb[topic]["question"]:
                    self.kb[topic]["question"].append(query)
                    
                if clean_response not in self.kb[topic]["responses"]:
                    self.kb[topic]["responses"].append(clean_response)
            
            # Save periodically
            if random.random() < 0.2:
                self.save_kb()

    def net_search(self, query: str, context: str) -> Dict:
        if not query:
            return {"content": "Please provide a search query.", "source": None}
            
        results = {"content": None, "source": None}
        
        def duck_search():
            return {"content": self.ddg_search(query), "source": f"{C['ACCENT2']}DuckDuckGo"}
            
        def wiki_search():
            wiki_query = context.replace("TUTORIAL", "", 1).strip() if context.startswith("TUTORIAL") else context
            return {"content": self.wiki_search(wiki_query), "source": f"{C['ACCENT3']}Wikipedia"}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            duck_future = executor.submit(duck_search)
            wiki_future = executor.submit(wiki_search)
            
            duck_result = duck_future.result()
            if duck_result["content"]:
                results = duck_result
            else:
                wiki_result = wiki_future.result()
                results = wiki_result
                
        if context.startswith("TUTORIAL") and results["content"]:
            tutorial_topic = context.replace("TUTORIAL", "", 1).strip()
            results["content"] = self.format_tutorial(tutorial_topic, results["content"])
                
        return results

    def run(self):
        try:
            while True:
                user_input = input(f"\n{C['ACCENT2']}You: {C['TEXT']}")
                self.stats["queries"] += 1
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.type(f"{C['SUCCESS']}Thank you for using {C['ACCENT3']}ore{C['SUCCESS']}, Goodbye!")
                    self.save_user()
                    break
                
                context = self.gen_ctx(user_input)
                
                if self.is_about_me(user_input):
                    response = self.get_about()
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "bot-info")
                    continue
                
                if self.offline != (not self.is_net()):
                    self.offline = not self.is_net()
                    if self.offline:
                        print(f"{C['ERROR']}[ore] ⚠️ Connection lost. Switching to offline mode.")
                    else:
                        print(f"{C['SUCCESS']}[ore] ✓ Connection restored. Online search available.")
                
                print(f"{C['WARNING']}[ore] Processing query...")
                
                if self.offline:
                    self.stats["offline"] += 1
                    response = self.kb_search(context)
                    print(f"{C['TITLE']}OResearch: ", end="")
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "kb")                    
                    self.learn(user_input, response)
                    
                else:
                    self.stats["online"] += 1
                    results = self.net_search(user_input, context)
                    
                    if results["content"]:
                        print(f"{C['TITLE']}ore: ", end="")
                        self.type(f"{C['TEXT']}{results['content']}\n{C['INFO']}Source: {results['source']}", 
                               speed=self.user["prefs"]["type_speed"])
                        
                        # Store in history and learn
                        self.add_hist(user_input, results["content"], results["source"])
                        self.learn(user_input, results["content"])
                    else:
                        # Fall back to offline search if online search fails
                        print(f"{C['WARNING']}[ore] Online search failed. Trying knowledge base...")
                        response = self.kb_search(context)
                        print(f"{C['TITLE']}OResearch: ", end="")
                        self.type(response, speed=self.user["prefs"]["type_speed"])
                        self.add_hist(user_input, response, "kb-fallback")
        
        except KeyboardInterrupt:
            print(f"\n{C['SUCCESS']}[ore] ore session interrupted. Saving data...")
            self.save_user()
            print(f"{C['SUCCESS']}[ore] Thank you for using ore! Goodbye!")
        
        except Exception as e:
            print(f"\n{C['ERROR']}[ore] Error: {str(e)}")
            print(f"{C['WARNING']}[ore] Please report this issue. Attempting to save data...")
            self.save_user()

if __name__ == "__main__":
    ore = OResearch()
    ore.run()
