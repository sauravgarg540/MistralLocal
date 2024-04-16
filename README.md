# Chat with Mistral with History context
Unleashing Language Magic: This project showcases the versatility of Mistral LLM by offering two groundbreaking deployment options - a locally deployed LLM for unparalleled privacy and control, and an LLM deployed on the Hugging Face serverless platform for seamless scalability. Leveraging the power of Gradio for intuitive interaction and Hugging Face for robust model integration, we've crafted an innovative solution that empowers users to harness the full potential of language models with ease and flexibility.

## Serverless or Local deployment
By default, the api call is made to model hosted on huggingface. You can also run the model locally or serverless by setting the environment variable `DEPLOYMENT = local/serverless`.

For the local we suggest to use a machine with GPU for faster inference.

You can choose to completely run the model locally on cpu by setting
```bash
USE_GPU = False
```

## Huggingface model supported:
- mistral_instruct: mistralai/Mistral-7B-Instruct-v0.2
- mistral: mistralai/Mistral-7B-v0.1

Easily switch between models by changing environment variable
```bash
MODEL = mistral_instruct #default
# or
MODEL = mistral    
```

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