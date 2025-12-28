from PromptEvol.environment import SentimentAnalysisEnvironment
from PromptEvol.generation_loader import GenerationLoader
from performer import SeqEvolBehaviourPerformer, OneofEvolBehaviourPerformer, TermiteEvolBehaviourPerformer
import argparse
import os

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prompt Evolution for Sentiment Analysis")
    parser.add_argument("--file_path", type=str, default="init_prompt", 
                       help="Path to the initial prompt file")
    parser.add_argument("--max_prompts", type=int, default=1, 
                       help="Maximum number of prompts to load")
    parser.add_argument("--model_name", type=str, default="Qwen3-0.6B-MNN/", 
                       help="Name/path of the model to use")
    parser.add_argument("--model_embed", type=str, default="Qwen3-0.6B-embedding/",
                        help="Name/path of the embedding model to use")
    parser.add_argument("--eval_set", type=str, default="dataset/sentiment_analysis.csv", 
                       help="Path to the evaluation dataset")
    parser.add_argument("--eval_num", type=int, default=1, 
                       help="Number of examples to use for evaluation")
    parser.add_argument("--top_k", type=int, default=1, 
                       help="Number of top individuals to select")
    parser.add_argument("--n_cluster", type=int, default=3, 
                       help="Number of the clusters in K-means for faster evaluation")
    parser.add_argument("--epochs", type=int, default=1, 
                       help="Number of evolution epochs to run")
    parser.add_argument("--output_dir", type=str, default="output", 
                       help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    return parser.parse_args()


def set_random_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    print(f"The random seed is set as {seed}.")

def main():
    args = parse_arguments()
    set_random_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the generation with prompts from file
    generation = GenerationLoader(args.file_path, max_prompts=args.max_prompts).init_generation()
    
    # eb_performer = SeqEvolBehaviourPerformer()
    eb_performer = OneofEvolBehaviourPerformer()
    # eb_performer = TermiteEvolBehaviourPerformer()
    print(eb_performer)
    
    # Initialize the sentiment analysis environment
    env = SentimentAnalysisEnvironment(
        model_name=args.model_name,
        emb_model_name=args.model_embed,
        eb_performer=eb_performer,
        eval_set=args.eval_set,
        eval_num=args.eval_num,
        top_k=args.top_k,
        n_cluster=args.n_cluster
    )
    
    # Run evolution for specified number of epochs
    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Evolve the current generation
        new_generation = env.evolution(generation)
        
        # Evaluate and display top performing individuals
        generation, scores = env.evaluate_generation(new_generation)

        for ind, (indiv, score) in enumerate(zip(generation, scores)):
            print(f"Top Individual {ind+1} score: {score:.4f}; evaluated: {indiv.evaled}")
            print(f"{indiv.get_text()}")
            if ind >= 2:
                break
        
        # Save the best individual to file
        output_file = f"{args.output_dir}/evolved_epoch{epoch+1}_top1.json"
        generation[0].save_prompts(output_file)


if __name__ == "__main__":
    main()