# Chatbot using Mistral with History context
Start chatting with Mistral model locally. This is especially useful if you don't want to share your data but still want to use chatGPT like model
 Easiest way to get API from huggingface, this way you don't to have a GPU to run the model.

## Model supported:
- Mistral Instruct

## Setup

Step 1: Create environment

```bash
# Using conda
 conda env create -f environment.yml
 
# Using venv
python3 -m venv env
pip install -r requirements.txt
```
Step 2: Install pytorch from 
https://pytorch.org/get-started/locally/

Step 3: Set environment variable

Set these to store transformers models and cache to a different location. Huggingface will download models to this location.

```bash
HF_API_KEY = <get api key from huggingface>
HF_DATASETS_CACHE = <path to cache | defaults to .cache>
HF_HOME = <path to store transformers models | defaults to .cache>
HUGGINGFACE_HUB_CACHE = <path to hub cache | defaults to .cache>
```


Step 4: Run the chatbot

```bash
python main.py
```