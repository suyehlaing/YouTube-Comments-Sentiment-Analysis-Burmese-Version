## YouTube Comments Semantic Analysis (Burmese)

Python | FastAPI | React | Transformers

A full-stack web application for performing zero-shot semantic analysis on Burmese language comments from YouTube videos. This project was developed as part of my NLP elective course at MIIT, and reflects my growing interest and passion for applying NLP to real-world Burmese-language data.

### Project Overview

The goal of this project is simple but meaningful:

    To understand how people react to a YouTube video by automatically analyzing its comment section.

For example, by analyzing comments on a Burmese news video, we can quickly see whether the audience reaction is positive, negative, critical, supportive, or emotional.

This tool automates the process of gathering comments from a YouTube video and analyzing their semantic content using a powerful zero-shot classification model. Instead of being limited to pre-defined categories like "positive" or "negative," this approach can dynamically classify comments based on user-defined criteria, making it highly adaptable for understanding nuances in Burmese discourse.

Key Features:

  Comment Scraping: Fetches all comments from a given YouTube video URL using the YouTube Data API v3.
  
  Zero-Shot Semantic Analysis: Leverages the joeddav/xlm-roberta-large-xnli model to classify comments without any task-specific training.
  
  Efficient Model Loading: Uses a local caching system (local_model.py) to download the model once, significantly reducing subsequent load times.
  
  Modern Web Architecture: Features a responsive React frontend with a high-performance FastAPI backend.

### Prerequisites

    Python 3.8+
    Node.js and npm
    A Google Cloud Project with the YouTube Data API v3 enabled and an API key.


### Motivations

As someone passionate about NLP, I wanted to explore how language models can help in real Burmese online environments. Burmese is a low-resource language, and building a dataset from scratch is difficult, so I looked for a project where real-world data is naturally available.

YouTube comment sections felt perfect—they contain raw reactions, emotions, criticisms, and discussions. I wanted to know:

    “Can we automatically understand whether people support or disagree with a news video just by analyzing its comments?”

This project is my attempt to bring NLP closer to everyday Burmese digital content and show how pretrained models can still give meaningful insights, even in low-resource settings.

### Acknowledgments

  Hugging Face for the Transformers library and the joeddav/xlm-roberta-large-xnli model.
  
  YouTube Data API v3 for providing access to comment data.
