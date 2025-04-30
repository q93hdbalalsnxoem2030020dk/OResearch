import os
import json
import time
import socket
import requests
import wikipedia
import spacy
import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Back, Style, init
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from collections import Counter

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

try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print(f"[ore] NLTK not found, installing...")
    import subprocess
    subprocess.run(["pip", "install", "nltk"], check=True)
    import nltk
    nltk.download('punkt', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(f"{Fore.YELLOW}[ore] Installing required language model...")
    import subprocess
    subprocess.run([
        "python", "-m", "spacy", "download", "en_core_web_sm"
    ], check=True)
    nlp = spacy.load("en_core_web_sm")

# Intent classification patterns
INTENT_PATTERNS = {
    "tutorial": [
        r"how\s+to\s+.+",
        r"steps\s+to\s+.+",
        r"guide\s+for\s+.+",
        r"tutorial\s+on\s+.+",
        r"teach\s+me\s+.+",
        r"explain\s+how\s+to\s+.+",
        r"instructions\s+for\s+.+",
        r"process\s+of\s+.+ing",
        r"way\s+to\s+.+",
        r"method\s+for\s+.+"
    ],
    "about_ore": [
        r"who\s+are\s+you",
        r"what\s+are\s+you",
        r"what\s+is\s+oresearch",
        r"about\s+yourself",
        r"tell\s+me\s+about\s+you",
        r"your\s+capabilities",
        r"who\s+made\s+you",
        r"your\s+creator",
        r"what\s+can\s+you\s+do",
        r"your\s+purpose"
    ],
    "question": [
        r"what\s+is\s+.+",
        r"who\s+is\s+.+",
        r"where\s+is\s+.+",
        r"when\s+was\s+.+",
        r"why\s+does\s+.+",
        r"how\s+does\s+.+"
    ]
}


class OResearch:
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), ".data")
        self.kb_path = os.path.join(self.data_dir, "knowledge_base.json")
        self.user_path = os.path.join(self.data_dir, "user_data.json")
        self.vectorizer = TfidfVectorizer()

        os.makedirs(self.data_dir, exist_ok=True)

        self.kb = self.load_kb()
        self.user = self.load_user()

        self.offline = not self.is_net()
        self.info_templates = [
            "I'm ore also known as Open Research Explorer — a lightweight, offline-capable research assistant created by sxc_qq1 from Yx.GG Discord server. I combine NLP, search algorithms, and custom prediction models.",
            "My name is ore also known as Open Research Explorer, designed by sxc_qq1 from Yx.GG Discord server to be your research companion with or without internet access. I use advanced language processing.",
            "ore here! Made by sxc_qq1 from YxGG Discord, I'm a research assistant built to extract and summarize information using AI techniques. I work both online and offline.",
            "I'm an AI research tool called ore or Open Research Explorer, developed by sxc_qq1 from YxGG Discord. I can search online sources when connected, or use my built-in knowledge base when offline.",
            "ore (Open Research Explorer) at your service! Created by sxc_qq1 from YxGG Discord, I'm a dual-mode research assistant that uses NLP and machine learning to answer your questions."]

        self.tutorial_templates = [
            "Here's a step-by-step guide on {}:",
            "Let me walk you through the process of {}:",
            "Follow these steps to {}:",
            "Here's my detailed tutorial on {}:",
            "A comprehensive guide to {}:"
        ]

        self.hello()

        self.stats = {
            "queries": 0,
            "online": 0,
            "offline": 0,
            "start_time": datetime.now()
        }

        # Load advanced NLP models
        self.setup_advanced_models()

    def setup_advanced_models(self):
        """Initialize advanced NLP models for better understanding"""
        self.embed_model = None
        self.intent_classifier = None

        try:
            if has_gpu:
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(
                    f"{Fore.GREEN}[ore] Enhanced embeddings model loaded {Fore.CYAN}(GPU accelerated)")

                # Load intent classifier if GPU available
                try:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model="distilbert-base-uncased",
                        tokenizer="distilbert-base-uncased",
                        device=0
                    )
                    print(
                        f"{Fore.GREEN}[ore] Intent classifier loaded {Fore.CYAN}(GPU accelerated)")
                except Exception as e:
                    print(
                        f"{Fore.YELLOW}[ore] Intent classifier not loaded: {e}")
            else:
                print(
                    f"{Fore.YELLOW}[ore] Running in basic mode {Fore.RED}(No GPU detected)")
        except ImportError:
            print(
                f"{
                    Fore.YELLOW}[ore] Running in basic mode {
                    Fore.RED}(Missing sentence-transformers package)")

    def hello(self):
        print(f"\n{Style.BRIGHT}{Fore.CYAN}" + "~" * 50)
        print(f"{Fore.CYAN}               ,-,")
        print(f"{Fore.CYAN}             .(((^v")
        print(f"{Fore.CYAN}        ,---'\\   - ")
        print(f"{Fore.CYAN}       /    ~ \\___/")
        print(f"{Fore.CYAN}      /|      /|~|\\")
        print(f"{Fore.CYAN}     | |     | |~||")
        print(f"{Fore.CYAN}     | |     | |~|'")
        print(f"{Fore.CYAN}     | |     | |")
        print(f"{Fore.CYAN}     | |     | |")
        print(
            f"{
                Style.BRIGHT}{
                Fore.WHITE}🔍 {
                Fore.CYAN}OResearch v3.0{
                    Fore.WHITE} - Enhanced AI Research Assistant")
        print(f"{Fore.CYAN}" + "~" * 50)
        print(
            f"{Fore.GREEN}[ore] Starting up... {Fore.YELLOW}I'm not perfect, be patient with me")
        if self.offline:
            print(
                f"{Fore.RED}[ore] ⚠️ Running in OFFLINE mode - using local knowledge only")
            print(f"{Fore.CYAN}" + "~" * 50 + "\n")

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
        for color_code in [
            f"{
                Fore.RED}", f"{
                Fore.GREEN}", f"{
                Fore.YELLOW}", f"{
                    Fore.BLUE}", f"{
                        Fore.MAGENTA}", f"{
                            Fore.CYAN}", f"{
                                Fore.WHITE}", f"{
                                    Style.BRIGHT}", f"{
                                        Style.RESET_ALL}"]:
            text_for_timing = text_for_timing.replace(color_code, "")

        for char in text:
            print(char, end='', flush=True)
            if char not in "\033[":
                time.sleep(speed)

        print()

    def detect_intent(self, text: str) -> Dict[str, float]:
        """Advanced intent detection using regex patterns and NLP"""
        if not text:
            return {"unknown": 1.0}

        text_lower = text.lower()
        intents = {}

        # Check for each intent pattern
        for intent_name, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    intents[intent_name] = intents.get(intent_name, 0) + 1

        # Check for question-specific intents
        doc = nlp(text_lower)
        if any(token.tag_ in ["WP", "WRB", "WDT"]
               for token in doc):  # WH-question words
            intents["question"] = intents.get("question", 0) + 2

        # Normalize scores
        total = sum(intents.values()) if intents else 1
        normalized = {k: v/total for k, v in intents.items()}

        # Default to "unknown" if no patterns matched
        if not normalized:
            normalized["unknown"] = 1.0

        return normalized

    def extract_tutorial_topic(self, text: str) -> str:
        """Extract the specific topic for a tutorial request"""
        if not text:
            return ""

        # Common tutorial request patterns
        patterns = [
            r"how\s+to\s+(.+?)(?:\?|$|\.)",
            r"steps\s+to\s+(.+?)(?:\?|$|\.)",
            r"guide\s+for\s+(.+?)(?:\?|$|\.)",
            r"tutorial\s+on\s+(.+?)(?:\?|$|\.)",
            r"teach\s+me\s+(?:how\s+to\s+)?(.+?)(?:\?|$|\.)",
            r"explain\s+how\s+to\s+(.+?)(?:\?|$|\.)",
            r"(?:can|could)\s+you\s+(?:tell|show)\s+me\s+how\s+to\s+(.+?)(?:\?|$|\.)",
            r"(?:can|could)\s+you\s+(?:give|provide)\s+(?:me\s+)(?:a\s+)?(?:tutorial|guide|instructions)\s+(?:on|for)\s+(.+?)(?:\?|$|\.)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()

        # Fallback to NLP extraction if regex fails
        doc = nlp(text)

        # Look for a verb followed by its object
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                # Find objects connected to this verb
                objects = [child.text for child in token.children
                           if child.dep_ in ["dobj", "pobj", "compound"]]
                if objects:
                    return " ".join(objects)

        # Last resort: extract noun phrases after "how to"
        if "how to" in text.lower():
            parts = text.lower().split("how to", 1)
            if len(parts) > 1:
                return parts[1].strip()

        return ""

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities and important concepts"""
        if not text:
            return []

        doc = nlp(text)

        # Extract named entities
        entities = [ent.text for ent in doc.ents if ent.text.strip()]

        # Extract important nouns and concepts
        important_nouns = [chunk.text for chunk in doc.noun_chunks
                           if not all(token.is_stop for token in chunk)]

        # Extract verbs that might be important
        action_verbs = [token.lemma_ for token in doc
                        if token.pos_ == "VERB" and not token.is_stop]

        # Combine all features
        all_features = entities + important_nouns + action_verbs

        # Remove duplicates while preserving order
        seen = set()
        return [
            x for x in all_features if not (
                x.lower() in seen or seen.add(
                    x.lower()))]

    def gen_ctx(self, text: str) -> Dict[str,
                                         Union[str, List[str], Dict[str, float]]]:
        """Generate structured context from user input"""
        if not text or not text.strip():
            return {
                "raw": "",
                "entities": [],
                "keywords": [],
                "intent": {"unknown": 1.0},
                "is_tutorial": False,
                "tutorial_topic": "",
                "query_type": "unknown"
            }

        # Basic context
        doc = nlp(text)

        # Detect intent
        intent_scores = self.detect_intent(text)
        is_tutorial = "tutorial" in intent_scores and intent_scores["tutorial"] > 0.3
        is_about_ore = "about_ore" in intent_scores and intent_scores["about_ore"] > 0.3

        # Extract entities and important concepts
        entities = self.extract_entities(text)

        # Extract keywords
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and
                    not token.is_stop and len(token.text) > 2):
                keywords.append(token.lemma_)

        # Determine query type
        query_type = "unknown"
        if is_about_ore:
            query_type = "about_ore"
        elif is_tutorial:
            query_type = "tutorial"

        # Extract tutorial topic if needed
        tutorial_topic = ""
        if is_tutorial:
            tutorial_topic = self.extract_tutorial_topic(text)

        # Create structured context
        context = {
            "raw": text,
            "entities": entities,
            "keywords": keywords,
            "intent": intent_scores,
            "is_tutorial": is_tutorial,
            "tutorial_topic": tutorial_topic,
            "query_type": query_type
        }

        return context

    def load_kb(self) -> Dict:
        if not os.path.exists(self.kb_path):
            default_kb = {"oresearch": {"question": ["who are you", "what are you", "what is oresearch"], "responses": [
                "I'm OResearch or ore, a lightweight research assistant designed to provide information both online and offline."]}}
            with open(self.kb_path, "w") as file:
                json.dump(default_kb, file, indent=2)
            return default_kb

        try:
            with open(self.kb_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(
                f"{Fore.RED}[ore] Error: Knowledge base file corrupted. Creating new one.")
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
            print(
                f"{Fore.RED}[ore] Error: User data file corrupted. Creating new one.")
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
        for color_code in [
            f"{
                Fore.RED}", f"{
                Fore.GREEN}", f"{
                Fore.YELLOW}", f"{
                    Fore.BLUE}", f"{
                        Fore.MAGENTA}", f"{
                            Fore.CYAN}", f"{
                                Fore.WHITE}", f"{
                                    Style.BRIGHT}", f"{
                                        Style.RESET_ALL}"]:
            clean_response = clean_response.replace(color_code, "")

        self.user["history"].append({
            "time": datetime.now().isoformat(),
            "query": query,
            "response": clean_response,
            "source": source
        })

        if len(self.user["history"]) % 5 == 0:
            self.save_user()

    def kb_search(self, context: Dict) -> str:
        """Advanced knowledge base search using context"""
        if not context:
            return f"{
                Fore.YELLOW}I need more specific information to search my knowledge base."

        best_match = None
        best_score = 0

        # Handle tutorial requests
        if context["is_tutorial"]:
            tutorial_topic = context["tutorial_topic"]
            if tutorial_topic:
                # Search for tutorial content
                for topic, entry in self.kb.items():
                    if tutorial_topic.lower() in topic.lower() or any(tutorial_topic.lower() in q.lower()
                                                                      for q in entry.get("question", [])):
                        response = random.choice(
                            entry.get(
                                "responses",
                                ["No tutorial available."]))
                        return self.format_tutorial(tutorial_topic, response)

                # If no direct match, try to find something related
                for topic, entry in self.kb.items():
                    for q in entry.get("question", []):
                        common_words = set(
                            tutorial_topic.lower().split()) & set(
                            q.lower().split())
                        if common_words and len(common_words) >= 1:
                            response = random.choice(entry.get(
                                "responses", ["No tutorial available."]))
                            return self.format_tutorial(
                                tutorial_topic, response)

                return f"{
                    Fore.YELLOW}I don't have a tutorial on '{tutorial_topic}' in my knowledge base."

        # Handle direct matches
        for keyword in context["keywords"]:
            if keyword.lower() in self.kb:
                entry = self.kb[keyword.lower()]
                return random.choice(
                    entry.get(
                        "responses",
                        ["No information available."]))

        # Try entity matches
        for entity in context["entities"]:
            if entity.lower() in self.kb:
                entry = self.kb[entity.lower()]
                return random.choice(
                    entry.get(
                        "responses",
                        ["No information available."]))

        # Try semantic search
        query_text = " ".join(context["entities"] + context["keywords"])

        for topic, entry in self.kb.items():
            questions = entry.get("question", [])

            for q in questions:
                # Calculate similarity score
                query_tokens = set(query_text.lower().split())
                q_tokens = set(q.lower().split())

                overlap = len(query_tokens & q_tokens)
                score = overlap / max(len(query_tokens),
                                      len(q_tokens)) if max(len(query_tokens),
                                                            len(q_tokens)) > 0 else 0

                if score > best_score:
                    best_score = score
                    best_match = entry

        if best_match and best_score > 0.3:
            response = random.choice(
                best_match.get(
                    "responses",
                    ["No specific information available."]))

            if context["is_tutorial"]:
                return self.format_tutorial(
                    context["tutorial_topic"], response)
            return response

        if context["is_tutorial"]:
            return f"{
                Fore.YELLOW}Sorry, I don't have a tutorial on that topic in my knowledge base."
        return f"{
            Fore.YELLOW}Sorry, I couldn't find relevant information in my knowledge base."

    def format_tutorial(self, topic: str, content: str) -> str:
        """Format content as a step-by-step tutorial"""
        if not topic or not content:
            return f"{
                Fore.YELLOW}I don't have enough information to create a tutorial."

        template = random.choice(self.tutorial_templates).format(topic)

        # Split content into meaningful steps
        steps = []
        sentences = sent_tokenize(content)

        # If we have enough sentences, use them as steps
        if len(sentences) >= 3:
            steps = sentences
        else:
            # Try to identify sub-steps by looking for markers
            step_markers = [
                "first",
                "then",
                "next",
                "after",
                "finally",
                "lastly",
                "begin",
                "start"]

            current_step = ""
            for sentence in sentences:
                if any(marker in sentence.lower()
                       for marker in step_markers) and current_step:
                    steps.append(current_step)
                    current_step = sentence
                else:
                    if current_step:
                        current_step += " " + sentence
                    else:
                        current_step = sentence

            if current_step:
                steps.append(current_step)

            # If we still don't have enough steps, just use the sentences
            if len(steps) < 2:
                steps = sentences

        tutorial = f"{Fore.CYAN}{template}\n\n"

        for i, step in enumerate(steps, 1):
            tutorial += f"{Fore.GREEN}{i}. {Fore.WHITE}{step.strip()}\n"

        return tutorial

    def get_about(self) -> str:
        base = random.choice(self.info_templates)
        caps = []

        if not self.offline:
            caps.append("search the internet for up-to-date information")
        caps.append("access my local knowledge base")
        caps.append("learn from our interactions to improve future responses")

        if self.embed_model:
            caps.append("use neural embeddings for semantic understanding")

        caps_text = ", ".join(
            caps[:-1]) + f" and {caps[-1]}" if len(caps) > 1 else caps[0]

        response = f"{Fore.CYAN}{base}\n\n{Fore.GREEN}I can {caps_text}."
        session_time = datetime.now() - self.stats["start_time"]
        minutes = int(session_time.total_seconds() / 60)

        if self.stats["queries"] > 0:
            response += f"\n\n{
                Fore.YELLOW}In our current session ({minutes} minutes), I've answered {
                self.stats['queries']} queries."

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
            print(f"{Fore.RED}[ore] DuckDuckGo search error: {e}")
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
            print(f"{Fore.RED}[ore] Wikipedia search error: {e}")
            return "Error retrieving information from Wikipedia."

    def analyze(self, articles: List[str], context: Dict) -> str:
        if not articles:
            return "No valid articles to analyze."

        # Extract search query from context
        search_query = context["raw"]
        if context["is_tutorial"]:
            search_query = context["tutorial_topic"]

        if not search_query:
            search_query = " ".join(context["entities"] + context["keywords"])

        if self.embed_model:
            try:
                query_embed = self.embed_model.encode([search_query])[0]
                article_embeds = self.embed_model.encode(articles)

                similarities = cosine_similarity(
                    [query_embed], article_embeds)[0]

                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.3:
                    return articles[best_idx]
                return "No highly relevant content found."
            except Exception as e:
                print(f"{Fore.RED}[ore] Embeddings error: {e}")
                # Fall back to TF-IDF

        # TF-IDF fallback
        try:
            vectors = self.vectorizer.fit_transform([search_query] + articles)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            best_idx = np.argmax(sims)
            return articles[best_idx] if sims[best_idx] > 0.2 else "No relevant content found."
        except Exception as e:
            print(f"{Fore.RED}[ore] TF-IDF error: {e}")
            if articles:
                return articles[0]
            return "Error analyzing the content."

    def learn(self, query: str, response: str, context: Dict):
        if not query or not response or len(response) < 20:
            return

        clean_response = response
        for color_code in [
            f"{
                Fore.RED}", f"{
                Fore.GREEN}", f"{
                Fore.YELLOW}", f"{
                    Fore.BLUE}", f"{
                        Fore.MAGENTA}", f"{
                            Fore.CYAN}", f"{
                                Fore.WHITE}", f"{
                                    Style.BRIGHT}", f"{
                                        Style.RESET_ALL}"]:
            clean_response = clean_response.replace(color_code, "")

        # Extract key topic from context
        topic = None

        if context["is_tutorial"] and context["tutorial_topic"]:
            topic = context["tutorial_topic"]
        elif context["entities"]:
            topic = context["entities"][0]
        elif context["keywords"]:
            topic = context["keywords"][0]

        if not topic:
            return

        # Add to knowledge base
        if topic.lower() not in self.kb:
            self.kb[topic.lower()] = {
                "question": [query],
                "responses": [clean_response]
            }
        else:
            if query not in self.kb[topic.lower()]["question"]:
                self.kb[topic.lower()]["question"].append(query)

            if clean_response not in self.kb[topic.lower()]["responses"]:
                self.kb[topic.lower()]["responses"].append(clean_response)

        # Save periodically
        if random.random() < 0.2:
            self.save_kb()

    def net_search(self, query: str, context: Dict) -> Dict:
        if not query:
            return {
                "content": "Please provide a search query.",
                "source": None}

        results = {"content": None, "source": None}

        # Determine search query
        search_query = query
        if context["is_tutorial"]:
            if context["tutorial_topic"]:
                search_query = f"how to {context['tutorial_topic']}"

        def duck_search():
            return {
                "content": self.ddg_search(search_query),
                "source": "DuckDuckGo"}

        def wiki_search():
            wiki_query = context["tutorial_topic"] if context["is_tutorial"] and context["tutorial_topic"] else search_query
            return {
                "content": self.wiki_search(wiki_query),
                "source": "Wikipedia"}

        with ThreadPoolExecutor(max_workers=2) as executor:
            duck_future = executor.submit(duck_search)
            wiki_future = executor.submit(wiki_search)

            duck_result = duck_future.result()
            if duck_result["content"]:
                results = duck_result
            else:
                wiki_result = wiki_future.result()
                results = wiki_result

        if context["is_tutorial"] and results["content"]:
            tutorial_topic = context["tutorial_topic"] or search_query
            results["content"] = self.format_tutorial(
                tutorial_topic, results["content"])

        return results

        def analyze(self, articles: List[str], context: Dict) -> str:
        if not articles:
            return "No valid articles to analyze."

        # Extract search query from context
        search_query = context["raw"]
        if context["is_tutorial"]:
            search_query = context["tutorial_topic"]

        if not search_query:
            search_query = " ".join(context["entities"] + context["keywords"])

        if self.embed_model:
            try:
                query_embed = self.embed_model.encode([search_query])[0]
                article_embeds = self.embed_model.encode(articles)

                similarities = cosine_similarity(
                    [query_embed], article_embeds)[0]

                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.3:
                    return articles[best_idx]
                return "No highly relevant content found."
            except Exception as e:
                print(f"{Fore.RED}[ore] Embeddings error: {e}")
                # Fall back to TF-IDF

        # TF-IDF fallback
        try:
            vectors = self.vectorizer.fit_transform([search_query] + articles)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            best_idx = np.argmax(sims)
            return articles[best_idx] if sims[best_idx] > 0.2 else "No relevant content found."
        except Exception as e:
            print(f"{Fore.RED}[ore] TF-IDF error: {e}")
            if articles:
                return articles[0]
            return "Error analyzing the content."

    def learn(self, query: str, response: str, context: Dict):
        if not query or not response or len(response) < 20:
            return

        clean_response = response
        for color_code in [
            f"{
                Fore.RED}", f"{
                Fore.GREEN}", f"{
                Fore.YELLOW}", f"{
                    Fore.BLUE}", f"{
                        Fore.MAGENTA}", f"{
                            Fore.CYAN}", f"{
                                Fore.WHITE}", f"{
                                    Style.BRIGHT}", f"{
                                        Style.RESET_ALL}"]:
            clean_response = clean_response.replace(color_code, "")

        # Extract key topic from context
        topic = None

        if context["is_tutorial"] and context["tutorial_topic"]:
            topic = context["tutorial_topic"]
        elif context["entities"]:
            topic = context["entities"][0]
        elif context["keywords"]:
            topic = context["keywords"][0]

        if not topic:
            return

        # Add to knowledge base
        if topic.lower() not in self.kb:
            self.kb[topic.lower()] = {
                "question": [query],
                "responses": [clean_response]
            }
        else:
            if query not in self.kb[topic.lower()]["question"]:
                self.kb[topic.lower()]["question"].append(query)

            if clean_response not in self.kb[topic.lower()]["responses"]:
                self.kb[topic.lower()]["responses"].append(clean_response)

        # Save periodically
        if random.random() < 0.2:
            self.save_kb()

    def run(self):
        try:
            while True:
                user_input = input(f"\n{Fore.CYAN}You: {Fore.WHITE}")
                self.stats["queries"] += 1

                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.type(f"{Fore.GREEN}Thank you for using ore, Goodbye!")
                    self.save_user()
                    break

                context = self.gen_ctx(user_input)

                if context["query_type"] == "about_ore":
                    response = self.get_about()
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "bot-info")
                    continue

                # Check network status changes
                current_net_status = self.is_net()
                if self.offline != (not current_net_status):
                    self.offline = not current_net_status
                    if self.offline:
                        print(
                            f"{Fore.RED}[ore] ⚠️ Connection lost. Switching to offline mode.")
                    else:
                        print(
                            f"{Fore.GREEN}[ore] ✓ Connection restored. Online search available.")

                print(f"{Fore.YELLOW}[ore] Processing query...")

                if self.offline:
                    self.stats["offline"] += 1
                    response = self.kb_search(context)
                    print(f"{Fore.CYAN}OResearch: ", end="")
                    self.type(response, speed=self.user["prefs"]["type_speed"])
                    self.add_hist(user_input, response, "kb")
                    self.learn(user_input, response, context)

                else:
                    self.stats["online"] += 1
                    results = self.net_search(user_input, context)

                    if results["content"]:
                        print(f"{Fore.CYAN}ore: ", end="")
                        self.type(
                            f"{
                                Fore.WHITE}{
                                results['content']}\n{
                                Fore.GREEN}Source: {
                                results['source']}",
                            speed=self.user["prefs"]["type_speed"])

                        # Store in history and learn
                        self.add_hist(
                            user_input, results["content"], results["source"])
                        self.learn(user_input, results["content"], context)
                    else:
                        # Fall back to offline search
                        print(
                            f"{Fore.YELLOW}[ore] Online search failed. Trying knowledge base...")
                        response = self.kb_search(context)
                        print(f"{Fore.CYAN}OResearch: ", end="")
                        self.type(
                            response, speed=self.user["prefs"]["type_speed"])
                        self.add_hist(user_input, response, "kb-fallback")
                        self.learn(user_input, response, context)

        except KeyboardInterrupt:
            print(f"\n{Fore.GREEN}[ore] Session interrupted. Saving data...")
            self.save_user()
            print(f"{Fore.GREEN}[ore] Goodbye!")
            exit(0)

        except Exception as e:
            print(f"\n{Fore.RED}[ore] Critical error: {str(e)}")
            print(f"{Fore.YELLOW}[ore] Attempting emergency save...")
            self.save_user()
            print(f"{Fore.RED}[ore] Shutting down. Please report this error.")
            exit(1)


if __name__ == "__main__":
    ore = OResearch()
    ore.run()
