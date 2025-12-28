import json
import os
import random
from PromptEvol import PromptIndividual

class GenerationLoader:
    def __init__(self, file_path, max_prompts=9999):
        self.file_path = file_path
        self.max_prompts = max_prompts
        if self.file_path.endswith(".json"):
            print(f"Loading prompts from {file_path}...")
            self.generation = self.load_prompts_from_json(file_path)[:self.max_prompts]
        elif isinstance(self.file_path, list):
            print(f"Loading prompts from multiple files: {file_path}...")
            self.generation = []
            for fp in self.file_path:
                self.generation += self.load_prompts_from_json(fp)[:self.max_prompts]
        elif os.path.isdir(self.file_path):
            print(f"Loading prompts from directory: {file_path}...")
            self.generation = []
            for fname in sorted(os.listdir(self.file_path)):
                if fname.endswith(".json"):
                    fp = os.path.join(self.file_path, fname)
                    self.generation += self.load_prompts_from_json(fp)[:self.max_prompts]
        print(f"Total prompts loaded: {len(self.generation)}")
    
    def load_prompts_from_json(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            prompts = json.load(f, )
        
        generation = []
        for ind, prompt in enumerate(prompts):
            pind = PromptIndividual(prompt)
            generation.append(pind)
        return generation

    def init_generation(self):
        return self.generation