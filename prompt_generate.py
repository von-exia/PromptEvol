import MNN.llm as llm
import argparse
import re
import json
import random
import os
from datetime import datetime
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate multiple LLM prompts and save to JSON')
    
    parser.add_argument('--model_path', type=str, 
                       help='Path to the model configuration file')
    parser.add_argument('-n', '--num_prompts', type=int, required=True,
                       help='Number of prompts to generate')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output JSON filename (default: init_prompt/template.json)')
    parser.add_argument('--template', type=str, default="Role-Task-Format",
                       help='Template structure (default: Role-Task-Format)')
    parser.add_argument('--task_prompt', type=str, default="""
    Your purpose is to analyze the sentiment in the given text.
    Remember to keep the continuity of this system prompt, so that I can directly concatenate them to form a complete prompt for sentiment analysis.
    If you think the sentiment is negative, respond with 0; if you think the sentiment is positive, respond with 1.""",
                       help='Define your task to generate individuals')
    return parser.parse_args()


def initialize_model(config_path):
    """Initialize and load the LLM model"""
    print(f"Initializing model from: {config_path}")
    model = llm.create(config_path)
    model.load()
    return model


def extract_json_from_response(response, nodes):
    """Extract JSON content from model response"""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        
        json_data = {}
        cnt = 0
        for name in nodes:
            match = re.search(r'\"{name}\": "*(.*?)"', response, re.DOTALL)
            if match:
                cnt += 1
                item = {name: match.group(1).strip()}
                json_data.update(item)
        if cnt == len(nodes):
            json.loads(json_str)
            return json_data
        return None
    except Exception as e:
        print(f"JSON extraction error: {str(e)}")
        return None
        


def create_prompt_template(template, task_prompt):
    """Create a template prompt"""
    nodes = template.split("-")
    json_text = ""
    for name in nodes:
        json_text += f'"{name}": "<to describe as detailed as possible>",\n'

    prompt = f"""
    You should first generate a system prompt with {template} structure as more detailed as possible, then transform this prompt as JSON format:
    FOMAT ->
    ```json
    {{
        {json_text}
    }}
    ```
    """
    
    prompt = task_prompt + prompt
    return prompt, nodes

def generate_prompt_with_template(model, prompt):
    """Generate a single prompt using the LLM model"""
    input_text = prompt + " <no_think> "
    response = model.response(input_text, False)
    return response

def save_prompts_to_file(prompts_data, output_filename):
    """Save generated prompts to JSON file"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, ensure_ascii=False, indent=2)
    print(f"Prompts saved to: {output_filename}")

def main():
    args = parse_arguments()
    
    # Initialize model
    model = initialize_model(args.model_path)
    
    # Generate prompt template
    template_prompt, nodes = create_prompt_template(args.template, args.task_prompt)
    print(f"Template prompt created:{template_prompt} ")
    
    # Store all generated prompts
    all_prompts = []
    successful_count = 0
    
    start = time.time()
    print(f"Generating {args.num_prompts} prompts...")
    
    for i in range(args.num_prompts):
        print(f"Generating prompt {i+1}/{args.num_prompts}")
        prompt = template_prompt + f" (Prompt ID: {i+1})"
        
        try:
            # Generate response from model
            response = generate_prompt_with_template(model, prompt)
            print(response)
            
            # Extract JSON data
            json_data = extract_json_from_response(response, nodes)
            
            prompt_data = {
                "prompt_id": i + 1,
                "generation_timestamp": datetime.now().isoformat(),
                # "full_response": response
            }
            
            if json_data:
                # Merge JSON data with metadata
                prompt_data.update(json_data)
                successful_count += 1
                print(f"  ✓ Successfully generated prompt {i+1}")
            else:
                prompt_data["error"] = "No valid JSON format found in response"
                print(f"  ⚠ Warning: No JSON found in response {i+1}")
            all_prompts.append(prompt_data)
            
        except Exception as e:
            print(f"  ✗ Error generating prompt {i+1}: {str(e)}")
            all_prompts.append({
                "prompt_id": i + 1,
                "generation_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "full_response": ""
            })
    
    # Determine output filename
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"init_prompt/{args.template}.json"
    
    # Save results
    save_prompts_to_file(all_prompts, output_filename)
    end = time.time()
    
    # Print summary
    print(f"\n=== Generation Summary ===")
    print(f"Total prompts attempted: {args.num_prompts}")
    print(f"Successfully generated: {successful_count}")
    print(f"Failed: {args.num_prompts - successful_count}")
    print(f"Output file: {output_filename}")
    print(f"Total time taken: {(end - start)/60:.2f} minutes")
    print(f"Average time per prompt: {(end - start)/args.num_prompts:.2f} seconds")
    
    # Preview successful prompts
    # successful_prompts = [p for p in all_prompts if 'error' not in p or p.get('error') == 'No valid JSON format found in response']
    # if successful_prompts:
    #     print(f"\nPreview of first 3 prompts:")
    #     for i, prompt in enumerate(successful_prompts[:3]):
    #         print(f"\n--- Prompt {i+1} ---")
    #         for key, value in prompt.items():
    #             if key not in ['full_response', 'error']:
    #                 print(f"  {key}: {value}")

if __name__ == "__main__":
    main()