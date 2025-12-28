import argparse
import sys
import re
import time
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import MNN.llm as llm
from PromptEvol.individual import PromptIndividual
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis with LLM')
    parser.add_argument('--model_path', help='Path to model configuration file')
    parser.add_argument('--data_path', default='dataset/sentiment_analysis.csv', 
                       help='Path to dataset CSV file (default: dataset/sentiment_analysis.csv)')
    parser.add_argument('--samples_per_class', type=int, default=50,
                       help='Number of samples per class (default: 50)')
    parser.add_argument('--prompt_file', help='Path to prompt file (for file or evolved sources)')
    parser.add_argument('--output_file', default='sentiment_analysis_results.csv',
                       help='Output file for detailed results (default: sentiment_analysis_results.csv)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for sampling (default: 42)')
    
    return parser.parse_args()


def load_model(model_config_path):
    """Load the LLM model"""
    print(f"Loading model from: {model_config_path}")
    model = llm.create(model_config_path)
    model.load()
    print("Model loaded successfully")
    return model


def load_and_sample_data(data_path, samples_per_class=50, random_state=42):
    """Load dataset and sample equal number from each class"""
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset size: {len(df)}")
    print(f"Label distribution: {dict(Counter(df['sentiment']))}")
    
    # Sample data
    sampled_dfs = []
    for label in ['negative', 'positive']:
        label_df = df[df['sentiment'] == label]
        if len(label_df) >= samples_per_class:
            # Using the last samples as in original code
            sampled_label = label_df[-samples_per_class:]
        else:
            print(f"Warning: Class {label} has only {len(label_df)} samples, using all")
            sampled_label = label_df
        sampled_dfs.append(sampled_label)
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Sampled dataset size: {len(sampled_df)}")
    print(f"Sampled label distribution: {dict(Counter(sampled_df['sentiment']))}")
    
    return sampled_df


def load_prompt(prompt_file=None):
    """Load prompt from different sources"""
    if  prompt_file:
        print(f"Loading evolved prompt from: {prompt_file}")
        prompt_individual = PromptIndividual()
        prompt_individual.load_evolved(prompt_file)
        prompt = prompt_individual.get_text()
    else:
        # Default prompt
        prompt ="""Analyze the sentiment in this text."""
        print("Using default prompt")
    
    print(f"Prompt loaded: {prompt}")
    return prompt


def extract_label_from_text(text):
    """
    Extract sentiment classification result (0 or 1) from text containing <think> tags
    """
    # Label mapping for text conversion
    text_to_label_map = {'negative': 0, 'positive': 1}
    
    # Method 1: Extract content after </think> and match 0 or 1
    pattern0 = r'</think>\s*(.*?)$'
    match0 = re.search(pattern0, text, re.DOTALL)
    if match0:
        text = match0.group(1).strip()
    
    # Try multiple extraction patterns
    patterns = [
        (r'\b[01]\b', lambda m: int(m.group())),  # Direct 0/1 match
        (r'Answer[:\s]*(\d)', lambda m: int(m.group(1))),  # Answer pattern
        (r'The sentiment is\s+(\w+)', lambda m: text_to_label_map.get(m.group(1).lower(), -1)),  # Text sentiment
        (r'sentiment:\s*(\w+)', lambda m: text_to_label_map.get(m.group(1).lower(), -1)),  # Sentiment label
        (r'is\s+(\w+)\s*sentiment', lambda m: text_to_label_map.get(m.group(1).lower(), -1)),  # Is X sentiment
        (r'\*\*Answer:\*\*\s*.*?(\bnegative\b|\bpositive\b)', lambda m: text_to_label_map.get(m.group(1).lower(), -1)),  # Bold answer
        (r'\*\*(\d)\*\*|\*(\d)\*', lambda m: int(m.group(1) or m.group(2))),  # Bold numbers
        (r'\*\*(negative|positive)\*\*', lambda m: text_to_label_map.get(m.group(1).lower(), -1)),  # Bold text
        (r'negative|positive', lambda m: text_to_label_map.get(m.group(0).lower(), -1)),  # Any text match
    ]
    
    for pattern, converter in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                result = converter(match)
                if result != -1:
                    return result
            except (ValueError, IndexError):
                continue
    
    # Final attempt: find any 0 or 1 in text
    final_match = re.findall(r'\b[01]\b', text)
    if final_match:
        return int(final_match[-1])
    
    return -1


def evaluate_model(model, df, prompt):
    """
    Evaluate model performance on the dataset
    
    Returns:
        Dictionary with evaluation results
    """
    # Label mappings
    text_to_label = {'positive': 1, 'negative': 0}
    label_to_text = {1: 'positive', 0: 'negative'}
    
    true_labels = []
    predicted_labels = []
    texts = []
    response_times = []
    invalid_count = 0
    raw_outputs = []
    error_samples = []
    
    print("Starting evaluation...")
    start_time = time.time()
    
    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔬 Evaluating", colour='MAGENTA'):
        tweet = row['text']
        true_label = text_to_label[row['sentiment']]
        
        # Prepare input text
        input_text = prompt + tweet + "<no_think>"
        
        # Measure inference time
        inference_start = time.time()
        try:
            output = model.response(input_text, False)
            inference_time = time.time() - inference_start
            response_times.append(inference_time)
            
            pred_label = extract_label_from_text(output)
            raw_outputs.append(output)
            
            if pred_label == -1:
                pred_label = not true_label  # Default to wrong prediction
                error_samples.append({
                    'index': idx,
                    'text': tweet,
                    'true_label': true_label,
                    'raw_output': output,
                    'error_type': 'extraction_failed'
                })
                invalid_count += 1
            else:
                # Only store correct/incorrect status, don't print every sample
                status = "✓" if pred_label == true_label else "✗"
                if status == "✗":
                    error_samples.append({
                        'index': idx,
                        'text': tweet,
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'raw_output': output,
                        'error_type': 'wrong_prediction'
                    })
            
            true_labels.append(true_label)
            predicted_labels.append(pred_label)
            texts.append(tweet)
            
        except Exception as e:
            print(f"Sample {idx} error: {e}")
            inference_time = time.time() - inference_start
            response_times.append(inference_time)
            true_labels.append(true_label)
            predicted_labels.append(not true_label)  # Default to wrong prediction
            raw_outputs.append("ERROR")
            error_samples.append({
                'index': idx,
                'text': tweet,
                'true_label': true_label,
                'error_type': 'exception',
                'exception': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Print error summary
    if error_samples:
        print(f"\nError Summary:")
        extraction_errors = [e for e in error_samples if e['error_type'] == 'extraction_failed']
        wrong_predictions = [e for e in error_samples if e['error_type'] == 'wrong_prediction']
        exceptions = [e for e in error_samples if e['error_type'] == 'exception']
        
        print(f"  - Label extraction failed: {len(extraction_errors)} samples")
        print(f"  - Wrong predictions: {len(wrong_predictions)} samples")
        print(f"  - Exceptions: {len(exceptions)} samples")
        
        # Print first few errors as examples
        if extraction_errors:
            print(f"\nFirst extraction error example:")
            error = extraction_errors[0]
            print(f"  Sample {error['index']}: '{error['text'][:50]}...'")
            print(f"  Raw output: '{error['raw_output'][:100]}...'")
    
    return {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'texts': texts,
        'response_times': response_times,
        'raw_outputs': raw_outputs,
        'total_time': total_time,
        'invalid_count': invalid_count,
        'total_samples': len(df),
        'error_samples': error_samples
    }


def calculate_metrics(results):
    """Calculate evaluation metrics from results"""
    true_labels = results['true_labels']
    predicted_labels = results['predicted_labels']
    response_times = results['response_times']
    
    metrics = {
        'total_time': results['total_time'],
        'invalid_predictions': results['invalid_count'],
        'valid_samples': len(true_labels) - results['invalid_count'],
        'total_samples': results['total_samples'],
    }
    
    if response_times:
        metrics['avg_inference_time'] = np.mean(response_times)
        metrics['std_inference_time'] = np.std(response_times)
    
    if len(true_labels) > 0:
        metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
        metrics['precision'] = precision_score(true_labels, predicted_labels, 
                                             average='binary', zero_division=0)
        metrics['recall'] = recall_score(true_labels, predicted_labels, 
                                       average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(true_labels, predicted_labels, 
                                     average='binary', zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            true_labels, predicted_labels, labels=[0, 1]
        )
    
    return metrics


def save_detailed_results(results, output_file):
    """Save detailed results to CSV file"""
    label_to_text = {1: 'positive', 0: 'negative'}
    
    results_df = pd.DataFrame({
        'text': results['texts'],
        'true_label': results['true_labels'],
        'predicted_label': results['predicted_labels'],
        'true_label_name': [label_to_text[l] for l in results['true_labels']],
        'predicted_label_name': [label_to_text[l] for l in results['predicted_labels']],
        'correct': [1 if results['true_labels'][i] == results['predicted_labels'][i] else 0 
                   for i in range(len(results['true_labels']))],
        'raw_output': results['raw_outputs']
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    # Save error samples separately if there are any
    if results['error_samples']:
        error_df = pd.DataFrame(results['error_samples'])
        error_file = output_file.replace('.csv', '_errors.csv')
        error_df.to_csv(error_file, index=False)
        print(f"Error samples saved to {error_file}")


def print_metrics(metrics):
    """Print evaluation metrics in a formatted way"""
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Valid predictions: {metrics['valid_samples']}/{metrics['total_samples']}")
    print(f"  Invalid predictions: {metrics['invalid_predictions']}")
    
    print(f"\nPerformance Metrics:")
    if 'accuracy' in metrics:
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    print(f"\nTiming Information:")
    print(f"  Total evaluation time: {metrics['total_time']:.2f} seconds")
    if 'avg_inference_time' in metrics:
        print(f"  Average inference time: {metrics['avg_inference_time']:.2f} seconds")
        print(f"  Std inference time: {metrics['std_inference_time']:.2f} seconds")
    
    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        label_to_text = {0: 'negative', 1: 'positive'}
        cm_df = pd.DataFrame(
            cm,
            index=[f'True {label_to_text[i]}' for i in range(2)],
            columns=[f'Pred {label_to_text[i]}' for i in range(2)]
        )
        print(cm_df)


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load and sample data
    df = load_and_sample_data(
        args.data_path, 
        args.samples_per_class, 
        args.random_state
    )
    
    # Load prompt
    prompt = load_prompt(args.prompt_file)
    
    # Evaluate model
    results = evaluate_model(model, df, prompt)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print_metrics(metrics)
    
    # Save detailed results
    save_detailed_results(results, args.output_file)


if __name__ == "__main__":
    main()