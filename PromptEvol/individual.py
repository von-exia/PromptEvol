import json

class PromptIndividual:
    """
    Class representing a prompt individual for genetic programming
    Each individual contains structured components that can be crossed over
    """
    def __init__(self, json_dict=None):
        self.json_dict = json_dict
        if self.json_dict is not None: 
            self.create_individual_from_template(json_dict)
        else:
            print("Warning: the content of the prompt is not initialized")
            print("Using ```load_evolved()``` to initialize the individual")
        self.score = -1
        self.evaled = False
        
    def save_prompts(self, output_filename):
        """Save evolved prompts to JSON file"""
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(self.structure, f, ensure_ascii=False, indent=2)
        print(f"Prompts saved to: {output_filename}")
        
    def load_evolved(self, file_path):
        """Load evolved prompts with JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f, )
        self.create_individual_from_template(json_dict)
        print(f"Prompts loaded from: {file_path}")
    
    def create_individual_from_template(self, json_dict):
        self.template = ""
        self.prompt = ""
        self.structure = {}
        for key, value in json_dict.items():
            if key not in ["prompt_id", "generation_timestamp"]:
                self.template += key + "-"
                self.prompt += f'{key}: {value}\n'
                self.structure[key] = value
        self.template = self.template[:-1]
        
    def get_text(self):
        return self.prompt
    
    def get_nodes(self):
        return self.structure
    
    def __repr__(self):
        return self.prompt
    

if __name__ == "__main__":
    import json
    with open("init_prompt/Task-Action-Goal.json", "r") as f:
        prompts = json.load(f)
    for prompt in prompts:
        pind = PromptIndividual(prompt)
