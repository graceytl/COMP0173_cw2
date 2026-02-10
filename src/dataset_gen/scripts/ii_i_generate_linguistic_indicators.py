"""
Generate linguistic indicators for the set of filtered sentences using Llama3.3-70B-Instruct.
Reads texts from CSV, processes them in batches, and saves structured JSON results for compilation.

Example usage:
    Remote:
    uv run python scripts/ii_i_generate_linguistic_indicators.py --input filtered_stereotypes.csv --output-dir outputs/linguistic_indicators --batch-size 5 --model us.meta.llama3-3-70b-instruct-v1:0

    Local vLLM:
    uv run python scripts/ii_i_generate_linguistic_indicators.py --input filtered_stereotypes.csv --output-dir outputs/linguistic_indicators --batch-size 5

    Compile results:
    uv run python scripts/ii_i_generate_linguistic_indicators.py --compile --output-dir outputs/linguistic_indicators
"""

import argparse
import json
import logging
import os
import pandas as pd
from src.dataset_gen.scripts.prompt import PROMPT_TEMPLATE, EXAMPLES

from src.dataset_gen.scripts.i_i_stereotype_detection import compile_json_checkpoints, process_csv

from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")  # Load environment variables

logging.basicConfig(level=logging.INFO)

API_ENDPOINT = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
team_id = os.getenv("TEAM_ID")
api_key = os.getenv("API_KEY")

# System prompt combining template and examples
SYSTEM_PROMPT = PROMPT_TEMPLATE + "\n\nExamples:\n" + json.dumps(EXAMPLES, indent=2)

# Output schema for structured responses
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string"},
                    "output": {
                        "type": "object",
                        "properties": {
                            "has_category_label": {"type": "string", "enum": ["yes", "no"]},
                            "full_label": {"type": "string"},
                            "target_type": {"type": "string", "enum": ["specific target", "generic target", "not-applicable"]},
                            "connotation": {"type": "string", "enum": ["negative", "positive", "neutral", "not-applicable"]},
                            "gram_form": {"type": "string", "enum": ["noun", "other", "not-applicable"]},
                            "ling_form": {"type": "string", "enum": ["generic", "subset", "individual", "not-applicable"]},
                            "information": {"type": "string"},
                            "situation": {"type": "string", "enum": ["situational behaviour", "enduring characteristics", "other", "not-applicable"]},
                            "situation_evaluation": {"type": "string", "enum": ["negative", "neutral", "positive", "not-applicable"]},
                            "generalization": {"type": "string", "enum": ["abstract", "concrete", "not-applicable"]}
                        },
                        "required": ["has_category_label", "full_label", "target_type", "connotation", "gram_form", "ling_form", "information", "situation", "situation_evaluation", "generalization"]
                    }
                },
                "required": ["sentence", "output"]
            }
        }
    },
    "required": ["results"]
}


def update_csv_with_results(
    input_csv: str,
    results_json: str,
    output_csv: str = None
):
    """
    Update the original CSV with analysis results.
    
    Args:
        input_csv: Original CSV file
        results_json: JSON file with analysis results
        output_csv: Output CSV path (default: input_csv with '_analysed' suffix)
    """
    df = pd.read_csv(input_csv)
    
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    # Create a mapping from sentence to results
    results_map = {r['sentence']: r['output'] for r in results}
    
    # Add new columns for each output field
    output_fields = [
        'has_category_label', 'full_label', 'target_type', 'connotation',
        'gram_form', 'ling_form', 'information', 'situation',
        'situation_evaluation', 'generalization'
    ]
    
    for field in output_fields:
        df[field] = df['text'].map(lambda x: results_map.get(x, {}).get(field, None))
    
    # Save
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_analysed.csv')
    
    df.to_csv(output_csv, index=False)
    logging.info(f"✓ Updated CSV saved to {output_csv}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse stereotype sentences using LLM model")
    parser.add_argument("--input", "-i", default="stereotypes_filtered.csv", help="Input CSV file")
    parser.add_argument("--output-dir", "-o", default="outputs/linguistic_indicators", help="Output directory")
    parser.add_argument("--model", "-m", default="us.meta.llama3-3-70b-instruct-v1:0", help="Model name")

    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Batch size")
    parser.add_argument("--start-from", "-s", type=int, default=0, help="Which row index to start from (in case of resume)")
    parser.add_argument("--compile", "-c", action="store_true", help="Compile checkpoint files only")
    parser.add_argument("--update-csv", "-u", action="store_true", help="Update CSV with results")
    parser.add_argument("--results-json", "-r", help="Results JSON for updating CSV")
    
    args = parser.parse_args()
    
    if args.compile:
        # Just compile existing checkpoints
        compile_json_checkpoints(args.output_dir + "_llama3.3")
    elif args.update_csv:
        # Update CSV with results
        results_json = args.results_json or f"{args.output_dir}/all_results.json"
        update_csv_with_results(args.input, results_json)
    else:
        # Process CSV
        logging.info("Starting stereotype analysis...")
        logging.info(f"Input: {args.input}")
        logging.info(f"Output: {args.output_dir}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Model: {args.model}")
        
        results, failed = process_csv(
            input_csv=args.input,
            output_dir=args.output_dir,
            model=args.model,
            prompt=SYSTEM_PROMPT,
            output_schema=OUTPUT_SCHEMA,
            vllm=False,
            batch_size=args.batch_size,
            start_from=args.start_from
        )
        
        logging.info(f"\n{'='*50}")
        logging.info(f"COMPLETE: {len(results)} sentences analysed")
        if failed:
            logging.warning(f"FAILED: {len(failed)} batches need retry")