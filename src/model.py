import os
from google import genai
from google.genai import types
import ast

from config import SEED
from data import *
from utils import *

### Google AI Studio ###

class GoogleAIStudio():
    def __init__(self, model="gemini-2.0-flash"):
        # assert model in {"gemini-2.0-flash", "gemini-2.0-flash-lite"}
        self.model = model

    def predict_one(self, prompt):
        client = genai.Client(api_key=os.environ["GOOGLE_AI_KEY"])
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                seed=SEED,
                temperature=0
            ),
        )

        return response.text

    def evaluate_one(self, prompt, query_idx, return_full_answer=False):
        eval_pr = self.predict_one(prompt)
        return eval_pr if return_full_answer else parse_eval_str_to_dict(eval_pr)
    
    def predict_and_evaluate_one(self, promptBuilder, query_idx, return_full_answer=False):
        inference_prompt = promptBuilder.build_inference_prompt(query_idx)
        answer_pr = self.predict_one(inference_prompt)
        answer_gt = get_one_answer_gt(promptBuilder.by_model, promptBuilder.image_resizing_mode, promptBuilder.output_mode, query_idx)[query_idx]
        eval_prompt = promptBuilder.build_eval_prompt(answer_gt, answer_pr)
        eval_pr = self.evaluate_one(eval_prompt, query_idx, return_full_answer)
        return query_idx, eval_pr

if __name__ == "__main__":

    vlm = GoogleAIStudio(model="gemma-3-27b-it")
    a = vlm.predict_one("What is the capital of France?")
