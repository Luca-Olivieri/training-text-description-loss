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

    def evaluate_one(self, eval_prompt, parse_to_dict=True):
        eval_pr = self.predict_one(eval_prompt)
        return parse_eval_str_to_dict(eval_pr) if parse_to_dict else eval_pr
    
    def predict_and_evaluate_one(self, promptBuilder, query_idx, parse_to_dict=True):
        inference_prompt = promptBuilder.build_inference_prompt(query_idx)
        answer_pr = self.predict_one(inference_prompt)
        eval_prompt = promptBuilder.build_eval_prompt(query_idx,  answer_pr)
        eval_pr = self.evaluate_one(eval_prompt, parse_to_dict)
        return query_idx, eval_pr
    
    def class_splitted_predict_one(self, promptBuilder, query_idx):
        class_splitted_inference_prompts, significant_classes = promptBuilder.build_class_splitted_inference_prompts(query_idx, return_significant_classes=True)
        pos_class_to_answer_pr_dict = {}
        for pos_class, inf_prompt in zip(significant_classes, class_splitted_inference_prompts):
            pos_class_to_answer_pr_dict[pos_class] = self.predict_one(inf_prompt)
        return pos_class_to_answer_pr_dict
    
    def class_splitted_evaluate_one(self, pos_class_2_eval_prompt, parse_to_dict=True):
        pos_class_2_eval_pr_dict = {}
        significant_classes = pos_class_2_eval_prompt.keys()
        eval_prompts = pos_class_2_eval_prompt.values()
        for pos_class, eval_prompt in zip(significant_classes, eval_prompts):
            pos_class_2_eval_pr_dict[pos_class] = self.evaluate_one(eval_prompt, parse_to_dict)
        return pos_class_2_eval_pr_dict

    def class_splitted_predict_and_evaluate_one(self, promptBuilder, query_idx, parse_to_dict=True):
        pos_class_2_answer_pr = self.class_splitted_predict_one(promptBuilder, query_idx)
        pos_class_2_eval_prompt = promptBuilder.build_class_splitted_eval_prompt(query_idx, pos_class_2_answer_pr)
        pos_class_2_eval_pr = self.class_splitted_evaluate_one(pos_class_2_eval_prompt, parse_to_dict)
        return query_idx, pos_class_2_eval_pr

if __name__ == "__main__":

    vlm = GoogleAIStudio(model="gemma-3-27b-it")
    a = vlm.predict_one("What is the capital of France?")
