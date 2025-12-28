import random
import numpy as np
import re
from .individual import PromptIndividual
from abc import ABC, abstractmethod
import json

class BaseEvolBehaviour(ABC):
    def __init__(self, p):
        self.p = p
        
    @abstractmethod
    def perform(self):
        pass
    
class CrossOver(BaseEvolBehaviour):
    def __init__(self, p=0.5):
        super().__init__(p)
        
    def perform(self, parent1, parent2):
        child = PromptIndividual(parent1.get_nodes())
        if np.random.rand() < self.p:
            p1 = parent1.get_nodes().copy()
            p2 = parent2.get_nodes().copy()
            if list(p2.keys()) and list(p1.keys()):
                to_cross = random.choice(list(p2.keys()))
                p1[to_cross] = p2[to_cross]
                to_remove = random.choice(list(p1.keys()))
                del p1[to_remove]
                child_dict = p1
                child = PromptIndividual(child_dict)
        return child

class Mutation(BaseEvolBehaviour):
    def __init__(self, p=0.5, p_llm=0.7):
        super().__init__(p)
        self.p_llm = p_llm
        
    def parse_json_string(self, json_str):
        """
        Parse a string containing JSON data wrapped in markdown code blocks
        
        Args:
            json_str (str): String containing JSON, typically wrapped in ```json ... ```
        
        Returns:
            dict: Parsed JSON data as Python dictionary, or None if parsing fails
        """
        # Method 1: Use regex to extract JSON content between ```json and ``` markers
        json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                # print(f"JSON parsing error: {e}")
                return None
        else:
            # If no markers found, try parsing the entire string directly
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # print("No valid JSON content found")
                return None
        
    def perform(self, indiv1, indiv2, llm):
        child = indiv1
        if np.random.rand() < self.p:
            
            #############  Content Mutation  #############
            if np.random.rand() < self.p_llm:
                p = child.get_nodes()
                k = list(p.keys())
                to_mutate_key = random.choice(k)
                text = child.get_text()
            
                
                prompt = f"""The current prompt is \"{text}\". Provide a more detailed and enhanced description for \"{to_mutate_key}\".
                The output FORMAT is like->
                ```json
                {{
                    \"{to_mutate_key}\": "<to improve as detailed as possible>"
                }}
                ```"""
                
                input_text = prompt + "<no_think>"
                # # response stream
                mutated_value = llm.response(input_text, False)
                pattern0 = r'</think>\s*(.*?)$'
                match0 = re.search(pattern0, mutated_value, re.DOTALL)
                if match0:
                    mutated_value = match0.group(1).strip()
                    mutated_json = self.parse_json_string(mutated_value)
                    
                    if mutated_json is not None:
                        # try:
                        p[to_mutate_key] = mutated_json[to_mutate_key]
                        mutated_json = p
                        child = PromptIndividual(mutated_json)
                        # except Exception as e:
                        #     # print(f"Mutation Warning: {e}")
                        #     # print(e)
                        #     return child
            else:
                p = child.get_nodes()
                k = list(p.keys())
                
                keys = k
                if keys:
                    to_mutate_key = random.choice(keys)
                    del p[to_mutate_key]        
                child = PromptIndividual(p)

        return child

class BaseEvolBehaviourPerformer:
    def __init__(self, eb_list=None):
        if eb_list is None:
            print("Warning: you didn't set the evolutionary behaviours for performer!")
        self.eb_list = eb_list if eb_list is not None else [CrossOver, Mutation]

    def perform(self, individual1, individual2, **kwargs):
        llm = kwargs['llm']
        best = kwargs['best']
        for evol_behaviour in self.eb_list:
            if isinstance(evol_behaviour, CrossOver):
                child = self.crossover = evol_behaviour.perform(individual1, individual2)
            elif isinstance(evol_behaviour, Mutation):
                child = evol_behaviour.perform(child, llm)
        return child
    
    def __call__(self, individual1, individual2, **kwargs):
        return self.perform(individual1, individual2, **kwargs)
    
    def __repr__(self):
        contain = [type(eb).__name__ + ": p=" + str(eb.p) + "\n" for eb in self.eb_list]
        content = ""
        for ind, name in enumerate(contain, start=1):
            content += str(ind) + ". " + name 
        return "\nThe evolutionary behaviours contain: \n" + content