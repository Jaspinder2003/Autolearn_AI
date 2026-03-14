import requests
import torch
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from collections import Counter, defaultdict
import re
import os
import nltk
from nltk.data import find
from mtranslate import translate
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Added imports for Hugging Face NLP
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    CLIPProcessor,
    CLIPModel
)


def translate_text(text, target_language='en'):
    """Translating text to english using mtranslate
    takes input as a text, outputs english text,used for Yandex"""
    try:
        return translate(text, target_language)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails


# Download all necessary NLTK data at the beginning
def ensure_nltk_resources():
    """this function ensures that all the nltk tokenizers and resources are downloaded
    only to ensure if the ntlk resources are already downloaded
    if not it downloads the resources"""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']

    for resource in resources:
        try:
            # Try to find the resource - this raises an exception if not found
            find(f'tokenizers/{resource}') if resource == 'punkt' else find(f'corpora/{resource}')
            print(f"Resource {resource} is already available.")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)


ensure_nltk_resources()

try:
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Couldn't load stopwords: {e}")
    # Fallback list of common stop words
    STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                  'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                  'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                  'to', 'from', 'in', 'on', 'at', 'by', 'with', 'without'}

# Add domain-specific stop words
DOMAIN_STOP_WORDS = {
    'general': {'image', 'picture', 'photo', 'jpg', 'jpeg', 'png', 'download', 'sale'}
}


class ReverseImageSearch:
    def __init__(self, debug=True, use_clip=True):
        self.debug = debug
        self.use_clip = use_clip
        # Set up headless browser
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        self.options.add_argument(f'user-agent={self.user_agent}')

        # Initialize NLP models
        self.setup_nlp_models()

        # Category mapping for general objects
        self.category_mapping = {
            # General categories
            "person": ["person", "people", "human", "man", "woman", "child"],
            "animal": ["animal", "wildlife", "pet", "dog", "cat", "bird", "fish"],
            "vehicle": ["vehicle", "car", "truck", "motorcycle", "bicycle", "bus", "train"],
            "building": ["building", "architecture", "house", "skyscraper", "tower", "castle"],
            "food": ["food", "meal", "dish", "cuisine", "fruit", "vegetable", "dessert"],
            "electronics": ["electronics", "device", "gadget", "phone", "computer", "laptop"],
            "nature": ["nature", "landscape", "mountain", "ocean", "forest", "river", "beach"],
            "sports": ["sports", "game", "athlete", "ball", "team", "competition"],
            "artwork": ["art", "painting", "sculpture", "drawing", "illustration"],
            "furniture": ["furniture", "chair", "table", "bed", "sofa", "desk"]
        }

        # Category confidence thresholds
        self.confidence_thresholds = {
            "default": 0.5,
            "person": 0.4,
            "animal": 0.4
        }

    def log(self, message):
        """Print debug messages if debug is enabled"""
        if self.debug:
            print(message)

    def setup_nlp_models(self):
        """Initialize NLP models for text processing"""
        try:
            # Initialize zero-shot classification for category classification
            self.zero_shot_model = pipeline("zero-shot-classification",
                                            model="facebook/bart-large-mnli")

            # Initialize NER model (more specialized than the original)
            self.ner_model = pipeline("token-classification",
                                      model="Jean-Baptiste/roberta-large-ner-english")

            # Initialize BERT model for better text classification
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

            # Initialize CLIP model for image-text similarity if requested
            if self.use_clip:
                try:
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    self.log("Successfully loaded CLIP model")
                except Exception as e:
                    self.log(f"Error loading CLIP model: {e}")
                    self.use_clip = False

            self.log("Successfully loaded NLP models")
        except Exception as e:
            self.log(f"Error loading NLP models: {e}")
            self.zero_shot_model = None
            self.ner_model = None
            self.bert_model = None
            self.tokenizer = None

    def setup_driver(self):
        return webdriver.Chrome(options=self.options)

    def search_with_yandex(self, image_path, max_results=10):
        """
        Perform reverse image search using Yandex
        """
        try:
            self.log(f"Starting Yandex search for: {image_path}")
            driver = self.setup_driver()
            driver.get('https://yandex.com/images/')

            # Find and click the camera icon to upload an image
            wait = WebDriverWait(driver, 10)
            camera_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.input__cbir-button')))
            camera_button.click()

            # Upload the image
            file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
            file_input.send_keys(image_path)

            # Wait for search results
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.CbirSites-Items')))
            time.sleep(2)  # Add a small delay to ensure results are loaded

            # Extract results
            results = []
            result_elements = driver.find_elements(By.CSS_SELECTOR, '.CbirSites-Item')

            self.log(f"Found {len(result_elements)} Yandex results")

            for i, element in enumerate(result_elements):
                if i >= max_results:
                    break

                try:
                    title_element = element.find_element(By.CSS_SELECTOR, '.CbirSites-ItemTitle')
                    url_element = element.find_element(By.CSS_SELECTOR, 'a')

                    result = {
                        'title': title_element.text,
                        'title_en': translate_text(title_element.text),
                        'url': url_element.get_attribute('href'),
                        'source': 'yandex',
                        'rank': i + 1
                    }
                    results.append(result)
                except Exception as e:
                    self.log(f"Error extracting result {i}: {e}")

            driver.quit()
            return results

        except Exception as e:
            self.log(f"Error in Yandex search: {e}")
            if 'driver' in locals():
                driver.quit()
            return []

    def search_with_bing(self, image_path, max_results=10):
        """
        Perform reverse image search using Bing
        """
        try:
            self.log(f"Starting Bing search for: {image_path}")
            driver = self.setup_driver()
            driver.get('https://www.bing.com/images/discover?FORM=ILPMFT')

            # Find and click image search button
            wait = WebDriverWait(driver, 10)
            search_by_image = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.sw_tpcn')))
            search_by_image.click()

            # Select upload image option
            upload_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='file']")))
            upload_option.send_keys(image_path)

            # Wait for results
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.iuscp')))
            time.sleep(2)  # Add a small delay to ensure results are loaded

            # Extract results
            results = []
            result_elements = driver.find_elements(By.CSS_SELECTOR, '.iusc')

            self.log(f"Found {len(result_elements)} Bing results")

            for i, element in enumerate(result_elements):
                if i >= max_results:
                    break

                try:
                    data_json = element.get_attribute('m')
                    if data_json:
                        data = json.loads(data_json)

                        result = {
                            'title': data.get('t', 'No title'),
                            'title_en': data.get('t', 'No title'),  # Assuming Bing returns English
                            'url': data.get('purl', '#'),
                            'source': 'bing',
                            'rank': i + 1
                        }
                        results.append(result)
                except Exception as e:
                    self.log(f"Error extracting Bing result {i}: {e}")

            driver.quit()
            return results

        except Exception as e:
            self.log(f"Error in Bing search: {e}")
            if 'driver' in locals():
                driver.quit()
            return []



    def extract_entities_with_ner(self, text):
        """Extract named entities using a specialized NER model"""
        if not self.ner_model or not text:
            return []

        try:
            # Get entities from text
            entities = self.ner_model(text)

            # Process and merge entities
            current_entity = {"word": "", "entity": "", "score": 0}
            merged_entities = []

            for item in entities:
                # Start a new entity
                if item["entity"].startswith("B-"):
                    # Save the previous entity if it exists
                    if current_entity["word"]:
                        merged_entities.append(current_entity)

                    current_entity = {
                        "word": item["word"],
                        "entity": item["entity"][2:],  # Remove B- prefix
                        "score": item["score"]
                    }
                # Continue an entity
                elif item["entity"].startswith("I-") and current_entity["entity"] == item["entity"][2:]:
                    current_entity["word"] += " " + item["word"]
                    current_entity["score"] = (current_entity["score"] + item["score"]) / 2
                # Start a new entity if mismatch
                else:
                    if current_entity["word"]:
                        merged_entities.append(current_entity)
                    current_entity = {
                        "word": item["word"],
                        "entity": item["entity"][2:] if item["entity"].startswith("B-") else item["entity"],
                        "score": item["score"]
                    }

            # Add the last entity if it exists
            if current_entity["word"]:
                merged_entities.append(current_entity)

            # Filter for high confidence and relevant entities
            relevant_entities = [
                e for e in merged_entities
                if e["score"] > 0.8 and e["entity"] in ["ORG", "PRODUCT", "MISC", "PER", "LOC"]
            ]

            return relevant_entities

        except Exception as e:
            self.log(f"Error in NER processing: {e}")
            return []

    def get_word_embeddings(self, words, target_categories):
        """
        Calculate semantic similarity between extracted words and target categories
        using BERT embeddings, basically like using a transformer
        """
        if not self.tokenizer or not self.bert_model or not words:
            return {}

        try:
            # Get embeddings for the words
            word_encodings = self.tokenizer(words, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                word_outputs = self.bert_model(**word_encodings, output_hidden_states=True)
                word_embeddings = word_outputs.hidden_states[-1][:, 0, :].numpy()  # Use CLS token embedding

            # Get embeddings for target categories
            category_encodings = self.tokenizer(list(target_categories), padding=True, truncation=True,
                                                return_tensors="pt")
            with torch.no_grad():
                category_outputs = self.bert_model(**category_encodings, output_hidden_states=True)
                category_embeddings = category_outputs.hidden_states[-1][:, 0, :].numpy()

            # Calculate cosine similarity
            similarities = {}
            for i, word in enumerate(words):
                word_similarities = {}
                for j, category in enumerate(target_categories):
                    sim = cosine_similarity([word_embeddings[i]], [category_embeddings[j]])[0][0]
                    word_similarities[category] = float(sim)
                similarities[word] = word_similarities

            return similarities

        except Exception as e:
            self.log(f"Error calculating word embeddings: {e}")
            return {}

    def extract_keywords_with_tfidf(self, search_results):
        """
        Extract keywords using TF-IDF to find the most important words in search results
        This is better than simple frequency counting
        """
        # Collect all text from search results
        all_texts = []
        for result in search_results:
            text = ""
            if 'title_en' in result and result['title_en']:
                text += result['title_en'] + " "
            if 'snippet' in result and result['snippet']:
                text += result['snippet'] + " "
            all_texts.append(text)

        if not all_texts:
            return []

        try:
            # Use TF-IDF to find important words
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words=list(STOP_WORDS) + list(DOMAIN_STOP_WORDS.get('general', set())),
                ngram_range=(1, 2)  # Allow for single words and pairs
            )

            # Transform texts to TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Get feature names (words)
            feature_names = vectorizer.get_feature_names_out()

            # Sum TF-IDF scores across all documents
            tfidf_sums = tfidf_matrix.sum(axis=0).A1

            # Create list of (word, score) tuples
            word_scores = [(feature_names[i], tfidf_sums[i]) for i in range(len(feature_names))]

            # Sort by score
            word_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top words with their scores
            return word_scores[:10]

        except Exception as e:
            self.log(f"Error in TF-IDF processing: {e}")
            return []

    def apply_general_classification(self, text, extracted_keywords):
        """
        Apply general classification for various types of objects in images
        """
        if not self.zero_shot_model:
            return None

        try:
            # General high-level categories
            general_categories = [
                "person", "animal", "vehicle", "building", "food",
                "electronics", "nature", "sports", "artwork", "furniture"
            ]

            result = self.zero_shot_model(text, general_categories)

            self.log(f"General classification: {result}")

            # If we have a confident match
            if result['scores'][0] > 0.4:
                # Get the main category
                main_category = result['labels'][0]

                # Try to get more specific within that category
                specific_categories = self.category_mapping.get(main_category, [])

                if specific_categories:
                    # Add any related keywords from our extraction
                    for keyword, _ in extracted_keywords:
                        keyword_lower = keyword.lower()
                        if keyword_lower not in specific_categories and len(keyword_lower) > 2:
                            specific_categories.append(keyword_lower)

                    # Get a more specific classification
                    specific_result = self.zero_shot_model(text, specific_categories)

                    if specific_result['scores'][0] > 0.5:
                        self.log(f"Specific classification: {specific_result['labels'][0]}")
                        return specific_result['labels'][0]

                return main_category

            # If nothing specific, return "object"
            return "object"

        except Exception as e:
            self.log(f"Error in general classification: {e}")
            return None

    def verify_with_clip(self, image_path, candidate_labels):
        """
        Use CLIP model to verify candidate labels against the actual image
        This helps filter out labels that don't match what's in the image
        """
        if not self.use_clip or not self.clip_model or not self.clip_processor:
            return candidate_labels

        try:
            # Load the image
            from PIL import Image
            image = Image.open(image_path)

            # Prepare text candidates
            texts = [f"a photo of a {label}" for label in candidate_labels]

            # Process inputs
            inputs = self.clip_processor(text=texts, images=image, return_tensors="pt", padding=True)

            # Get model predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)

            # Calculate similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).numpy()[0]

            # Combine labels with scores
            label_scores = [(label, float(score)) for label, score in zip(candidate_labels, probs)]

            # Sort by score
            label_scores.sort(key=lambda x: x[1], reverse=True)

            self.log(f"CLIP verification results: {label_scores}")

            # Filter for high confidence
            verified_labels = [label for label, score in label_scores if score > 0.2]

            return verified_labels if verified_labels else [label_scores[0][0]]

        except Exception as e:
            self.log(f"Error in CLIP verification: {e}")
            return candidate_labels

    def process_search_results(self, search_results):
        """
        Process search results using advanced NLP techniques
        to extract more accurate image labels
        """
        if not search_results:
            return None, None

        # Step 1: Combine texts from all search results
        all_text = " ".join([
            f"{result.get('title_en', '')} {result.get('snippet', '')}"
            for result in search_results
        ])

        # Step 2: Extract keywords using TF-IDF
        keywords = self.extract_keywords_with_tfidf(search_results)
        self.log(f"TF-IDF keywords: {keywords}")

        # Step 3: Extract named entities
        entities = self.extract_entities_with_ner(all_text)
        self.log(f"Named entities: {entities}")

        # Step 4: Apply general classification
        general_label = self.apply_general_classification(all_text, keywords)
        self.log(f"General object label: {general_label}")

        # Step 5: Generate candidate labels from different sources
        candidate_labels = []

        # Add TF-IDF keywords
        candidate_labels.extend([word for word, _ in keywords[:5]])

        # Add high-confidence entities
        candidate_labels.extend([
            entity["word"] for entity in entities
            if entity["score"] > 0.8 and entity["entity"] in ["PRODUCT", "ORG"]
        ])

        # Add general label if available
        if general_label:
            candidate_labels.append(general_label)

        # Ensure we have unique labels
        candidate_labels = list(set(candidate_labels))

        self.log(f"Final candidate labels: {candidate_labels}")

        return candidate_labels, general_label

    def extract_image_name_fallback(self, image_path):
        """Extract a potential label from the image filename as last resort"""
        try:
            # Get filename without extension
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Clean up the filename
            clean_name = re.sub(r'[^a-zA-Z]', ' ', name_without_ext).strip()

            # If filename has multiple words, return the longest one
            if ' ' in clean_name:
                words = clean_name.split()
                return max(words, key=len).lower()

            return clean_name.lower() if clean_name else None
        except:
            return None

    def identify_image(self, image_path):
        """
        Enhanced main method to identify an image using multiple search engines,
        advanced NLP, and CLIP verification
        """
        # Ensure image_path is absolute
        image_path = os.path.abspath(image_path)
        self.log(f"Processing image: {image_path}")

        # Collect results from multiple search engines
        all_results = []

        # Try Yandex
        self.log(f"Searching Yandex for: {image_path}")
        yandex_results = self.search_with_yandex(image_path)
        all_results.extend(yandex_results)

        # Try Bing
        if len(all_results) < 1:
            self.log("Getting additional results from Bing...")
            bing_results = self.search_with_bing(image_path)
            all_results.extend(bing_results)

        # Try Google if we still need more results


        # Log results for debugging
        self.log(f"Combined search results: {len(all_results)} items")

        # If no results, try to extract info from filename
        if not all_results:
            self.log("No search results found, trying to extract info from filename")
            return self.extract_image_name_fallback(image_path)

        # Process search results to get candidate labels
        candidate_labels, general_label = self.process_search_results(all_results)

        if not candidate_labels:
            self.log("Couldn't extract any labels, falling back to filename")
            return self.extract_image_name_fallback(image_path)

        # Use CLIP to verify which labels actually match the image
        if self.use_clip:
            verified_labels = self.verify_with_clip(image_path, candidate_labels)
            self.log(f"CLIP verified labels: {verified_labels}")

            if verified_labels:
                # If we have a general category label and it's verified, prioritize it
                if general_label and general_label in verified_labels:
                    return general_label
                return verified_labels[0]

        # If CLIP verification failed or wasn't used, prioritize the general label
        if general_label:
            return general_label

        # Otherwise return the first candidate label
        return candidate_labels[0]


# Example usage with enhancements
if __name__ == "__main__":
    try:
        # Create instance with CLIP verification enabled
        image_searcher = ReverseImageSearch(debug=True, use_clip=True)

        # Use path to the image
        image_path = r"C:\Users\jaspi\Downloads\th.jpg"

        label = image_searcher.identify_image(image_path)
        print(f"Identified as: {label}")

    except Exception as e:
        print(f"Error in main execution: {e}")