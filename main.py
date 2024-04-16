import time

import gradio as gr

from src.language_model import LLModel

mistral_model = LLModel()


def predict(message, *args, **kwargs):

    bot_message = mistral_model.generate_response(message)

    partial_message = ""
    for chunk in bot_message:
        partial_message = partial_message + chunk
        time.sleep(0.001)
        yield partial_message


gr.ChatInterface(predict).launch()
