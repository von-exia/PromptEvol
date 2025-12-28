from PromptEvol.behaviour import BaseEvolBehaviourPerformer, BaseEvolBehaviour, CrossOver, Mutation
from PromptEvol.individual import PromptIndividual
import numpy as np
import random
import re


class Shuffle(BaseEvolBehaviour):
    def __init__(self, p):
        super().__init__(p)
        
    def perform(self, individual):
        child = individual
        if np.random.rand() < self.p:
            json_dict = individual.get_nodes()
            keys = list(json_dict.keys())
            random.shuffle(keys)
            json_dict = {key: json_dict[key] for key in keys}
            child = PromptIndividual(json_dict)
        return child
    

class BestIndivGuidedCrossOver(BaseEvolBehaviour):
    def __init__(self, p):
        super().__init__(p)
        self.crossover = CrossOver(p=p)
        
    def perform(self, indiv, best=None):
        child = indiv
        if best is not None:
            child = self.crossover.perform(indiv, best)
        return child
    

class BestIndivGuidedMutation(Mutation):
    def __init__(self, p):
        super().__init__(p)
        
    def perform(self, individual, llm, best=None):
        child = individual
        if np.random.rand() < self.p and best is not None:
            p = individual.get_nodes()
            text = individual.get_text()
            to_mutate_key = random.sample(list(p.keys()), k=1)[0]
            
            best_text = best.get_text()
            best_structure = best.get_nodes()
            
            json_template = ""
            for name in list(best_structure.keys()):
                json_template += f'"{name}": "description for {name}",\n'
            json_template += f'"{to_mutate_key}": "description for {to_mutate_key}",\n'
            
            
            # instruction prompt
            prompt = f"""**Task**: Improve the current prompt by fusing it with the best-performing prompt.

            **Best Performing Prompt** (reference for excellence):
            \"\"\"{best_text}\"\"\"

            **Current Prompt** (to be improved):
            \"\"\"{text}\"\"\"

            **Instructions**:
            1. Analyze why the best prompt is effective
            2. Identify strengths from both prompts
            3. Create a new prompt that combines their advantages
            4. Ensure the new prompt is coherent and well-structured

            **Output Format** (strictly follow this JSON structure):
            ```json
            {{
                {json_template}
            }}
            ```.
            """
            
            input_text = prompt + "<no_think>"
            # # response stream
            mutated_value = llm.response(input_text, False)
            pattern0 = r'</think>\s*(.*?)$'
            match0 = re.search(pattern0, mutated_value, re.DOTALL)
            mutated_json = match0.group(1).strip()
            mutated_json = self.parse_json_string(mutated_json)
            if mutated_json is not None:
                # print("Extraction Successfully!")
                # print(mutated_json)
                child = PromptIndividual(mutated_json)
        return child

        
class SeqEvolBehaviourPerformer(BaseEvolBehaviourPerformer):
    def __init__(self, eb_list=[CrossOver(), Mutation()]):
        super().__init__(eb_list)
        # best
        # eb_list = [
        #     CrossOver(p=1),
        #     Shuffle(p=0.1),
        #     Mutation(p=0.3, p_llm=0.9),
        #     BestIndivGuidedCrossOver(p=0.3),
        #     BestIndivGuidedMutation(p=0.3)
        # ]
        eb_list = [
            # CrossOver(p=0.9),
            # Shuffle(p=0.1),
            # Mutation(p=1., p_llm=1.),
            # BestIndivGuidedCrossOver(p=0.2),
            BestIndivGuidedMutation(p=1.),
        ]
        self.eb_list = eb_list
        
        self.operation_handlers = {
            CrossOver: lambda eb, child, ctx: eb.perform(child, ctx['indiv2']),
            Mutation: lambda eb, child, ctx: eb.perform(child, ctx['indiv2'], ctx['llm']),
            Shuffle: lambda eb, child, ctx: eb.perform(child),
            BestIndivGuidedCrossOver: lambda eb, child, ctx: eb.perform(child, ctx['best']),
            BestIndivGuidedMutation: lambda eb, child, ctx: eb.perform(child, llm=ctx['llm'], best=ctx['best'])
        }
    

    def perform(self, individual1, individual2, **kwargs):
        kwargs['indiv2'] = individual2
        child = individual1
        for evol_behaviour in self.eb_list:
            behaviour_type = type(evol_behaviour)
            handler = self.operation_handlers.get(behaviour_type)
            if handler:
                child = handler(evol_behaviour, child, kwargs)
            else:
                print(f"Warning: the handler of {behaviour_type} is not defined!")
        return child
    
    
class OneofEvolBehaviourPerformer(BaseEvolBehaviourPerformer):
    def __init__(self, eb_list=[CrossOver(), Mutation()]):
        super().__init__(eb_list)
        eb_list = [
            CrossOver(p=1.),
            Mutation(p=1., p_llm=0.95),
            BestIndivGuidedCrossOver(p=1.),
            BestIndivGuidedMutation(p=1.),
        ]
        self.eb_list = eb_list
        
        self.operation_handlers = {
            CrossOver: lambda eb, child, ctx: eb.perform(child, ctx['indiv2']),
            Mutation: lambda eb, child, ctx: eb.perform(child, ctx['indiv2'], ctx['llm']),
            Shuffle: lambda eb, child, ctx: eb.perform(child),
            BestIndivGuidedCrossOver: lambda eb, child, ctx: eb.perform(child, ctx['best']),
            BestIndivGuidedMutation: lambda eb, child, ctx: eb.perform(child, llm=ctx['llm'], best=ctx['best'])
        }
    

    def perform(self, individual1, individual2, **kwargs):
        kwargs['indiv2'] = individual2
        child = individual1
        p = np.random.rand()
        p1 = 0.3
        p2 = 0.8
        p3 = 0.9
        if p < p1:
            handler = self.operation_handlers.get(CrossOver)
            child = handler(self.eb_list[0], child, kwargs)
        elif p1 <= p < p2:
            handler = self.operation_handlers.get(Mutation)
            child = handler(self.eb_list[1], child, kwargs)
        elif p2 <= p < p3:
            handler = self.operation_handlers.get(BestIndivGuidedCrossOver)
            child = handler(self.eb_list[2], child, kwargs)
        else:
            handler = self.operation_handlers.get(BestIndivGuidedMutation)
            child = handler(self.eb_list[3], child, kwargs)
        return child


class TermiteEvolBehaviourPerformer(BaseEvolBehaviourPerformer):
    def __init__(self, eb_list=[CrossOver(), Mutation()]):
        super().__init__(eb_list)
        eb_list = [
            CrossOver(p=1.),
            Mutation(p=0.3, p_llm=0.9),
            Shuffle(p=0.1),
        ]
        self.eb_list = eb_list
        
        eb_list2 = [
            BestIndivGuidedCrossOver(p=0.),
            BestIndivGuidedMutation(p=1.)
        ]
        self.eb_list2 = eb_list2
        
        self.operation_handlers = {
            CrossOver: lambda eb, child, ctx: eb.perform(child, ctx['indiv2']),
            Mutation: lambda eb, child, ctx: eb.perform(child, ctx['indiv2'], ctx['llm']),
            Shuffle: lambda eb, child, ctx: eb.perform(child),
            BestIndivGuidedCrossOver: lambda eb, child, ctx: eb.perform(child, ctx['best']),
            BestIndivGuidedMutation: lambda eb, child, ctx: eb.perform(child, llm=ctx['llm'], best=ctx['best'])
        }
    
    def __repr__(self):
        contain = [type(eb).__name__ + ": p=" + str(eb.p) + "\n" for eb in self.eb_list]
        content = ""
        for ind, name in enumerate(contain, start=1):
            content += str(ind) + ". " + name 
        content1 = "\nThe evolutionary behaviours of worker contain: \n" + content
        
        contain2 = [type(eb).__name__ + ": p=" + str(eb.p) + "\n" for eb in self.eb_list2]
        content2 = ""
        for ind, name in enumerate(contain2, start=1):
            content2 += str(ind) + ". " + name 
        content2 = "\nThe evolutionary behaviours of soldier contain: \n" + content2
        return content1 + content2

    def perform(self, individual1, individual2, **kwargs):
        kwargs['indiv2'] = individual2
        child = individual1
        if np.random.rand() < 0.:
            for evol_behaviour in self.eb_list:
                behaviour_type = type(evol_behaviour)
                handler = self.operation_handlers.get(behaviour_type)
                if handler:
                    child = handler(evol_behaviour, child, kwargs)
                else:
                    print(f"Warning: the handler of {behaviour_type} is not defined!")
        else:
            for evol_behaviour in self.eb_list2:
                behaviour_type = type(evol_behaviour)
                handler = self.operation_handlers.get(behaviour_type)
                if handler:
                    child = handler(evol_behaviour, child, kwargs)
                else:
                    print(f"Warning: the handler of {behaviour_type} is not defined!")
        return child           

    
