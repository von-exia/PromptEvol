import MNN.llm as llm
import random
import pandas as pd
import re
import time
import json
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from abc import ABC, abstractmethod
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy

from .individual import PromptIndividual
from .behaviour import CrossOver, Mutation, BaseEvolBehaviourPerformer
from .utils.function import batch_softmax


class BaseEnvironment(ABC):
    def __init__(self, model_name, eb_performer=None):
        self.llm = self.initialize_model(model_name)
        if eb_performer is None:
            print("Warning: you do not define the evolution behaviour performer, using default BaseEvolBehaviourPerformer...")
        self.eb_performer = eb_performer if eb_performer is not None else BaseEvolBehaviourPerformer()
    
    def initialize_model(self, model_name):
        """Initialize and load the LLM model"""
        print(f"Initializing model from: {model_name}")
        model = llm.create(model_name)
        model.load()
        return model
    
    def evolution(self, generation):
        g = deepcopy(generation)
        new_generation = deepcopy(generation)
        
        total_pairs = len(g) // 2
        current_pair = 0
        

        with tqdm(
            total=total_pairs,
            desc="🧬 Evolutionary Process",
            unit="pair",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour='green'
        ) as progress_bar:
            
            while len(g) >= 2:
                current_pair += 1
                
                ind1 = random.choice(g)
                g.remove(ind1)
                ind2 = random.choice(g)
                g.remove(ind2)
                
                ind1 = PromptIndividual(ind1.get_nodes())
                ind2 = PromptIndividual(ind2.get_nodes())
                child1 = self.evolution_step(ind1, ind2)
                child2 = self.evolution_step(ind2, ind1)
                
                new_generation.append(child1)
                new_generation.append(child2)
                
                # update progress bar
                progress_bar.update(1)
                
                # show current stats
                progress_bar.set_postfix({
                    'parents': f"{len(generation)} → {len(g)}",
                    'children': len(new_generation) - len(generation),
                    'total': len(new_generation)
                })
                
                # output 4 times
                # if current_pair % max(1, total_pairs // 4) == 0:
                #     tqdm.write(f"🎯 Processed {current_pair}/{total_pairs} pairs, population: {len(new_generation)}")
        tqdm.write(f"✅ Evolution complete! Population grew from {len(generation)} to {len(new_generation)}")
        return new_generation  
    
    
    def evolution_step(self, individual1, individual2):
        child = self.eb_performer(individual1, individual2, self.llm)
        return child
    
    
    def hierarchical_importance_sorted_with_remove(self, individuals_list):
        """
        Delete duplicate individuals based on the structure dictionary and retain the first one that appears
        """
        individuals_list = sorted(
        individuals_list,
        key=lambda x: (x.score, x.evaled, -len(x.get_text())),
        reverse=True
        )
        
        seen_structures = set()
        unique_individuals = []
        
        for individual in individuals_list:
            # Convert the structure dictionary into a hashable string
            # structure_str = json.dumps(individual.get_nodes(), sort_keys=True)
            indiv_text = individual.get_text()
            
            if indiv_text not in seen_structures:
                seen_structures.add(indiv_text)
                unique_individuals.append(individual)
        
        return unique_individuals
    
    @abstractmethod
    def evaluate_individual(self, individual):
        pass
    
    @abstractmethod
    def evaluate_generation(self, individual):
        pass
    
    @abstractmethod
    def extract_label_from_text(self, text):
        pass
    
    @abstractmethod
    def load_eval_set(self, eval_set, eval_num):
        pass


class SentimentAnalysisEnvironment(BaseEnvironment):
    def __init__(self, 
                model_name, 
                eb_performer,
                eval_set='dataset/sentiment_analysis.csv',
                eval_num=10, 
                top_k=50,
                emb_model_name="Qwen3-0.6B-embedding",
                n_cluster=3
                ):
        super().__init__(model_name, eb_performer)
        print("Initializing all models...")
        self.eval_set = self.load_eval_set(eval_set, eval_num)
        self.emb_model = self.init_emb_model(emb_model_name)
        self.top_k = top_k
        self.n_cluster = n_cluster
        self.best = None
        print(f"All models initialized successfully.")
        
    def init_emb_model(self, emb_model_name):
        emb_model = SentenceTransformer(emb_model_name)
        return emb_model
        
    def load_eval_set(self, eval_set, eval_num=10):
        """Load evaluation dataset"""
        df = pd.read_csv(eval_set)
        
        def sample_per_class(df, num_per_class=10, random_state=42):
            sampled_dfs = []
            for label in ['negative', 'positive']: 
                label_df = df[df['sentiment'] == label]
                if len(label_df) >= num_per_class:
                    # sampled_label = label_df.sample(num_per_class, random_state=random_state)
                    sampled_label = label_df[:num_per_class]
                else:
                    print(f"Warning: class {label} only {len(label_df)} samples, using all samples")
                    sampled_label = label_df
                sampled_dfs.append(sampled_label)
        
            return pd.concat(sampled_dfs, ignore_index=True)
        sampled_df = sample_per_class(df, num_per_class=eval_num)
        print(f"The size of sampled dataset: {len(sampled_df)}")
        print(f"The distribution of sampled dataset: {dict(Counter(sampled_df['sentiment']))}")
        return sampled_df
    
    def evolution_step(self, individual1, individual2):
        child = self.eb_performer(individual1, individual2, llm=self.llm, best=self.best)
        return child
    
    def extract_label_from_text(self, text):
        """
        Extract sentiment classification result (0 or 1) from text containing <think> tags
        
        Args:
            text (str): Complete text containing <think> tags
            
        Returns:
            int: Extracted sentiment label (0 or 1), returns None if not found
        """
        # Method 1: Extract all content after </think>, then match 0 or 1
        pattern0 = r'</think>\s*(.*?)$'
        match0 = re.search(pattern0, text, re.DOTALL)
        text = match0.group(1).strip()
        # print("text after think:", text)
        
        match1 = re.search(r'\b[01]\b', text)
        if match1:
            # print("Match1:", match1)
            return int(match1.group())
        
        # Method 2: If method 1 fails, look for explicit answer patterns in the entire text
        pattern2 = r'Answer[:\s]*(\d)'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            # print("Match2:", match2)
            return int(match2.group(1))
        
        # 多种匹配模式
        patterns = [
            r"The sentiment is\s+(\w+)",
            r"sentiment:\s*(\w+)",
            r"is\s+(\w+)\s*sentiment",
            r"\*\*Answer:\*\*\s*.*?(\bnegative\b|\bpositive\b|\bneutral\b)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sentiment = match.group(1).lower()
                if sentiment == 'negative':
                    return 0
                elif sentiment == 'positive':
                    return 1
                return -1
        
        # Method 3: Find bolded or emphasized numbers
        pattern3 = r'\*\*(\d)\*\*|\*(\d)\*'
        match3 = re.search(pattern3, text)
        if match3:
            # Extract the first non-empty group
            # print("Match3:", match3)
            digit = match3.group(1) or match3.group(2)
            return int(digit)
        
        # Method 4:
        pattern4 = r'\*\*(negative|positive)\*\*'
        matches4 = re.findall(pattern4, text, re.IGNORECASE)  
        if matches4:
            # print("Match4:", matches4)
            if matches4[0].lower() == 'negative':
                return 0
            elif matches4[0].lower() == 'positive':
                return 1
            
        pattern4 = r'negative|positive'
        matches4 = re.findall(pattern4, text, re.IGNORECASE)   
        if matches4:
            # print("Match4.5:", matches4)
            if matches4[0].lower() == 'negative':
                return 0
            elif matches4[0].lower() == 'positive':
                return 1
                
        # Method 5: Last resort, find any 0 or 1 in the entire text
        pattern5 = r'\b[01]\b'
        matches5 = re.findall(pattern5, text)
        if matches5:
            # Return the last match (typically the thinking process mentions possible options first, then gives the answer last)
            return int(matches5[-1])
        
        return -1
    
    def evaluate_individual(self, individual):
        text_to_label = {
            'positive': 1,
            'negative': 0, 
        }

        label_to_text = {
                    1: 'positive',
                    0: 'negative', 
                }
        
        # To test
        true_labels = []
        predicted_labels = []
        texts = []
        response_times = []
        
        # print("Start evaluation...")
        # start_time = time.time()

        
        for idx, row in self.eval_set.iterrows():
            tweet = row['text']
            true_label = text_to_label[row['sentiment']]
            
            input_text = individual.get_text() + tweet + "<no_think>"
            
            inference_start = time.time()
            try:
                out = self.llm.response(input_text, False)
                inference_time = time.time() - inference_start
                response_times.append(inference_time)
                
                pred_label = self.extract_label_from_text(out)
                
                true_labels.append(true_label)
                if pred_label == -1 or pred_label not in [0, 1]:
                    pred_label = not true_label
                predicted_labels.append(pred_label)
                texts.append(tweet)
                
                # if pred_label is None:
                #     print(f"Sample {idx}: invalid label, the output is: '{out}'")
                # else:
                #     status = "✓" if pred_label == true_label else "✗"
                #     print(f"Sample {idx}: Prediction={pred_label}({label_to_text [pred_label]}), Truth={true_label}({label_to_text [true_label]}) {status}")
            
            except Exception as e:
                # print(f"Sample {idx} error: {e}")
                inference_time = time.time() - inference_start
                response_times.append(inference_time)
                true_labels.append(true_label)
                predicted_labels.append(not true_label)

        # compute metrics
        if len(true_labels) > 0:
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
            recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
            # print(f"ACC: {accuracy:.4f}; Pre: {precision:.4f}; Recall: {recall:.4f}; F1: {f1:.4f}")
            # return (accuracy + f1) / 2.
            return (accuracy + precision + recall) / 3.
        return 0
    
    
    def evaluate_generation(self, gen):
        """
        evaluate a generation of individuals and return their scores
        
        Args:
            generation: list of PromptIndividual instances
            
        Returns:
            top_generation: top_k individuals with highest scores
            top_scores: corresponding scores
        """
        gen_text = [gen[i].get_text() for i in range(len(gen))]

        print("🔢 Performing embedding process")
        start = time.time()
        gen_embeddings = self.emb_model.encode(gen_text, precision="float32", normalize_embeddings=True)
        end = time.time()
        print(f"✅ Embedding finished, taken time: {end - start:.4f} s")


        # Specify the number of clustering center
        n_clusters = min(self.n_cluster , len(gen_text))

        ############ Perform K-means ############
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(gen_embeddings)

        # Obtain the cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Find the closest samples as Representative Indiviudals (RIs)
        closest_indices = []
        center_emb = []
        gen_cosine = cosine_similarity(cluster_centers, gen_embeddings)
        for i, center in enumerate(cluster_centers):
            # Compute the cosine similarity between centers and all samples
            similarities = gen_cosine[i]
            
            # Find the best match individual
            closest_idx = np.argmax(similarities)
            closest_indices.append(closest_idx)
            center_emb.append(gen_embeddings[closest_idx])
        ############ Perform K-means ###############
        
        ############### Evaluate Representative Individuals (RIs) ###############
        progress_bar = tqdm(
            total=len(closest_indices),
            desc="🎯 Evaluating Representative Individuals",
            unit="individual",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour='cyan'
        )
        
        ris_scores = []
        for i, idx in enumerate(closest_indices):
            center_prompt = gen[idx]
            if center_prompt.evaled:
                tqdm.write(f"📊 Progress: cluster evaluated early, score - {center_prompt.score:.4f}")
                score = center_prompt.score
            else:
                score = self.evaluate_individual(center_prompt)
                center_prompt.evaled = True
                center_prompt.score = score
            ris_scores.append(score)
            
            progress_bar.update(1)
            if (i + 1) % max(1, len(closest_indices) // 10) == 0:  # output every 10%
                tqdm.write(f"📊 Progress: {i+1}/{len(closest_indices)} cluster evaluated, score - {score:.4f}")
        progress_bar.close()
        ############### Evaluate Representative Individuals (RIs) ###############
        
        ############### Evaluate Top not evalated Individuals ###############
        progress_bar = tqdm(
            total=2,
            desc="🎯 Evaluating Top 2 not evaluated individuals",
            unit="individual",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour='cyan'
        )
        
        cnt = 0
        for i, indiv in enumerate(gen):
            if not indiv.evaled and indiv.score != -1:
                score = self.evaluate_individual(indiv)
                indiv.evaled = True
                indiv.score = score
            
                progress_bar.update(1)
                tqdm.write(f"📊 Progress: Top {i+1}/{len(gen)} evaluated, score - {score:.4f}")
                
                cnt += 1
                if cnt == 2:
                    break
        progress_bar.close()
        ############### Evaluate Top not evalated Individuals ###############
        
        ############### Weak-supervison based on RIs ###############
        scores = []
        for i, prompt in enumerate(gen):
            # get the clustering label of current prompt
            cluster_id = int(cluster_labels[i])
            
            # if it is evaluated, directly use its fitness socre
            if prompt.evaled:
                scores.append(prompt.score)
                # print(f"RI Prompt {i} : evaluated = {prompt.score:.4f};")
            else:
                # Computing the similarity between current individual and the representative individuals
                similarity = cosine_similarity(gen_embeddings[i].reshape(1, -1), center_emb[cluster_id].reshape(1, -1))[0, 0]
                
                # Adjust fitness based on similarity
                ri_score = ris_scores[cluster_id]
                # Avoding cosine = 1, we trust the evaluated sample more
                # random_factor = random.uniform(1e-4, 1e-2)
                random_factor = random.uniform(-0.005, 0.005)
                adjusted_score = ri_score * (similarity + random_factor) 
                
                # Ensure the score in valid range
                adjusted_score = max(0, min(1, adjusted_score))
                
                # Record the score back to the prompt
                if prompt.score != -1:
                    adjusted_score = prompt.score * 0.2 + adjusted_score * 0.8
                prompt.score = adjusted_score
                scores.append(adjusted_score)
            
                # print(f"Prompt {i} : sim = {similarity:.4f}; RI's scores: {ri_score:.4f}; " +
                #     f"adjusted score = {adjusted_score:.4f}")
        ############### Weak-supervison based on RIs ###############
        
        top_indices = np.argsort(scores)[::-1]
        top_generation = [gen[i] for i in top_indices]# [:self.top_k]
    
        
        top_generation = self.hierarchical_importance_sorted_with_remove(top_generation)[:self.top_k]
        top_scores = [indiv.score for indiv in top_generation]
        
        for indiv in top_generation:
            if indiv.evaled:
                self.best = top_generation[0]
                print(f"The best evaluated individual is {indiv.get_text()}")
                print(f"The best evaluated score is {indiv.score:.4f}")
                break
        

        print(f"\n=== Overall Score ===")
        print(f"Avg. Score: {np.mean(top_scores):.4f}")
        print(f"Max Score: {np.max(top_scores):.4f}")
        print(f"Min score: {np.min(top_scores):.4f}")
            
        return top_generation, top_scores
