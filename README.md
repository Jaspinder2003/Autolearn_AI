#  AutoLearner AI — Self-Learning Image Classifier

A self-learning image classification system that **autonomously identifies unknown objects** by combining a CNN classifier with reverse image search and web scraping. When the model encounters an image it can't confidently classify, it searches the web, identifies the object, scrapes training data, and retrains itself — all without human intervention.

---

## Key Features

- **Self-Expanding Knowledge** — Automatically learns new object categories it hasn't seen before
- **Reverse Image Search** — Uses Yandex and Bing to identify unknown images via visual similarity
- **Advanced NLP Pipeline** — Extracts object labels from search results using TF-IDF, Named Entity Recognition (NER), zero-shot classification (BART), and BERT embeddings
- **CLIP Verification** — Cross-references candidate labels against the actual image using OpenAI's CLIP model
- **Automated Web Scraping** — Downloads training images from Google Images via Selenium
- **CNN Training & Inference** — Trains a convolutional neural network with data augmentation and evaluates on test sets
- **Self-Expanding CNN Architecture** — Experimental PyTorch model that dynamically adds convolutional blocks as complexity grows

---

## Project Structure

```
self_learning_ai/
├── reverse_image_search.py   # Core: identifies unknown images via reverse search + NLP
├── webscraper.py              # Scrapes Google Images for training data
├── trainer.py                 # Trains the CNN classifier (TensorFlow/Keras)
├── model_runner.py            # Runs inference on images using the trained model
├── testTrainer.py             # Experimental self-expanding CNN (PyTorch)
├── runner.cpp                 # C++ entry-point stub for the full pipeline
├── notes.txt                  # Design notes and research references
├── error.log                  # Runtime error log
├── models/                    # Saved trained models (.h5)
├── training_images/           # Training dataset (organized by class)
├── test_images/               # Test dataset (organized by class)
├── scraped_images/            # Auto-downloaded images from web scraping
└── unknown_images/            # Images the model couldn't classify
```

---

## How It Works

```
┌─────────────┐     Low Confidence     ┌──────────────────────┐
│ CNN Model   │ ─────────────────────► │ Reverse Image Search │
│ (Inference) │                        │ (Yandex / Bing)      │
└─────────────┘                        └──────────┬───────────┘
       ▲                                          │
       │                                          ▼
       │                               ┌──────────────────────┐
       │                               │ NLP Pipeline          │
       │                               │ • TF-IDF Keywords     │
       │                               │ • NER Entities        │
       │                               │ • Zero-Shot Classify  │
       │                               │ • CLIP Verification   │
       │                               └──────────┬───────────┘
       │                                          │
       │                                          ▼
       │                               ┌──────────────────────┐
       │    Retrain with new data      │ Web Scraper           │
       │ ◄──────────────────────────── │ (Google Images)       │
       │                               └──────────────────────┘
```

1. **Classify** — The CNN model attempts to classify an input image
2. **Detect Uncertainty** — If confidence is below the threshold, the image is flagged as unknown
3. **Reverse Search** — The unknown image is uploaded to Yandex/Bing to find visually similar results
4. **Extract Labels** — NLP techniques analyze search result titles/snippets to determine what the object is
5. **Verify with CLIP** — Candidate labels are verified against the image using CLIP similarity scores
6. **Scrape Training Data** — Google Images is scraped for the identified object to build a training set
7. **Retrain** — The CNN is retrained with the newly acquired data, expanding its knowledge

---

## Tech Stack

| Component              | Technology                                              |
|------------------------|---------------------------------------------------------|
| **Image Classification** | TensorFlow / Keras (CNN)                              |
| **Experimental Model** | PyTorch (Self-Expanding CNN)                            |
| **Reverse Image Search** | Selenium WebDriver (Yandex, Bing)                    |
| **NLP / Text Analysis** | Hugging Face Transformers (BART, RoBERTa, BERT, CLIP) |
| **Keyword Extraction** | scikit-learn (TF-IDF), NLTK                            |
| **Translation**        | mtranslate                                              |
| **Web Scraping**       | Selenium, Requests, BeautifulSoup                       |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Google Chrome (for Selenium WebDriver)
- [ChromeDriver](https://chromedriver.chromium.org/) matching your Chrome version

### Installation

```bash
# Clone the repository
git clone https://github.com/adimulunj/self_learning_ai.git
cd self_learning_ai

# Install dependencies
pip install tensorflow torch torchvision transformers
pip install selenium requests beautifulsoup4
pip install scikit-learn nltk mtranslate Pillow numpy matplotlib
```

### Usage

#### Train the CNN Model
```bash
python trainer.py
```
Place your training images in `training_images/` organized by class (one subdirectory per class), and test images in `test_images/`.

#### Run Inference
```bash
python model_runner.py <path_to_image>
```

#### Identify an Unknown Image (Reverse Search)
```python
from reverse_image_search import ReverseImageSearch

searcher = ReverseImageSearch(debug=True, use_clip=True)
label = searcher.identify_image("path/to/unknown_image.jpg")
print(f"Identified as: {label}")
```

#### Scrape Training Images
```python
from webscraper import imageSearch, save_images

urls = imageSearch("golden retriever", numberOfImages=20)
save_images(urls, "golden retriever")
```

---

## Model Architecture

### CNN Classifier (TensorFlow)

```
Input (128×128×3)
  → Conv2D(32) → MaxPool
  → Conv2D(64) → MaxPool
  → Conv2D(128) → MaxPool
  → Flatten → Dense(128) → Dropout(0.5)
  → Dense(num_classes, softmax)
```

**Data Augmentation:** Random horizontal flips, brightness adjustment, and contrast adjustment are applied during training.

### Self-Expanding CNN (PyTorch — Experimental)

A dynamic architecture that starts with a base number of convolutional blocks and can **expand at runtime** by appending near-identity convolutional layers with small Gaussian noise — allowing the network to grow in capacity without catastrophic forgetting.

---

## Research Notes

- **Out-of-Distribution (OOD) Detection** is used to determine when an image doesn't belong to any known class. Methods considered include maximum softmax scoring, Mahalanobis distance, and Monte Carlo Dropout.
- **Known Limitation:** The model may exhibit overconfidence on similar-looking objects (e.g., a toy boat vs. a real boat), leading to ineffective retraining cycles.
- See [`notes.txt`](notes.txt) for full design notes and references.

---

## License

This project is for educational and research purposes.

---

## Contributing

this repo shows a copy of project developed in a team consisting of
-Tauheed Ali
-Bazhao Wang
-Aditya Mulunjkar
-Jaspinder Singh Maan
-Alvish Prasla

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
