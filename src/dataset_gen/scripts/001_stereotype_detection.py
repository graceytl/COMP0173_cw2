"""
Script for analysing stereotype sentences.
Processes CSV input and outputs JSON results for compilation.

Example usage:
    Remote:
    uv run python scripts/001_stereotype_detection.py --input stereotypes.csv --output-dir analysis_outputs --batch-size 5 --model us.meta.llama3-3-70b-instruct-v1:0

    Local vLLM:
    uv run python scripts/001_stereotype_detection.py --input stereotypes.csv --output-dir analysis_outputs --batch-size 5 --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --vllm

    Compile results:
    uv run python scripts/001_stereotype_detection.py --compile --output-dir analysis_outputs
"""

import os
import re
from openai import OpenAI
import logging
import pandas as pd
import json
import time
from pathlib import Path

import requests
from src.dataset_gen.scripts.prompt import STEREOTYPE_PROMPT, STEREOTYPE_DETECTION_EXAMPLES

from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")  # Load environment variables

logging.basicConfig(level=logging.INFO)

API_ENDPOINT = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
team_id = os.getenv("TEAM_ID")
api_key = os.getenv("API_KEY")

# Connect to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="python"
)

# System prompt combining template and examples
SYSTEM_PROMPT = STEREOTYPE_PROMPT + "\n\nExamples:\n" + json.dumps(STEREOTYPE_DETECTION_EXAMPLES, indent=2)

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
                            "beliefs_expectancies": {"type": "string", "enum": ["yes", "no", "not-applicable"]},
                            "information": {"type": "string"},
                            "behavior_features_traits": {"type": "string", "enum": ["yes", "no", "not-applicable"]},
                            "stereotype": {"type": "string", "enum": ["yes", "no", "not-applicable"]},
                        },
                        "required": ["has_category_label", "full_label", "beliefs_expectancies", "information", "behavior_features_traits", "stereotype"]
                    }
                },
                "required": ["sentence", "output"]
            }
        }
    },
    "required": ["results"]
}


def analyse_sentences(n_batch: int, sentences: list, output_path: Path, model: str = "Qwen/Qwen2.5-7B-Instruct", vllm: bool = True) -> list:
    """
    Analyse sentences using local vLLM server with structured JSON output.
    
    Args:
        sentences: List of sentences to analyse
        model: Model name
        vllm: Whether to use vLLM or remote server (boto3 client)
        
    Returns:
        List of analysis results or None if failed
    """

    # Escape any problematic characters in sentences
    clean_sentences = []
    for s in sentences:
        # Replace problematic characters
        clean = str(s).replace('\n', ' ').replace('\r', ' ')
        clean_sentences.append(clean)

    if vllm:
        logging.info(f"Analyzing {len(clean_sentences)} sentences with model {model} using vLLM server...")
    
        prompt = SYSTEM_PROMPT + f"\n\nAnalyse these sentences:\n{json.dumps(clean_sentences, ensure_ascii=False)}"
        return vllm_analysis(n_batch, prompt, output_path, model=model)
    else:
        logging.info(f"Analyzing {len(clean_sentences)} sentences with model {model} using remote API...")

        prompt = SYSTEM_PROMPT + f"\n\nAnalyse these sentences:\n{json.dumps(clean_sentences, ensure_ascii=False)}"
        formatted_prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),

        tools = [
                {
                    "toolSpec": {
                        "name": "analysis_results",
                        "description": "Structured JSON output for stereotype analysis",
                        "inputSchema": {
                            "json": OUTPUT_SCHEMA
                        }
                    }
                }
            ]
        
        return remote_analysis(n_batch, formatted_prompt, output_path, model=model, tools=tools)
        

def vllm_analysis(n_batch: int, prompt: str, output_path: Path, model: str = "Qwen/Qwen2.5-7B-Instruct") -> list:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=6450,
            top_p=0.75,
            temperature=0.0,  # 0 temperature for consistent outputs following method in paper
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analysis_results",
                    "schema": OUTPUT_SCHEMA
                }
            },
        )
        
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content
            
            # Log raw response for debugging
            logging.debug(f"Raw response length: {len(text)}")
            
            return check_response(n_batch, text, output_path)
        else:
            logging.warning("No response choices received")
            return None
            
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return None


def log_full_request(response):
    req = response.request
    print(f"--- Request Log ---")
    print(f"Method: {req.method}")
    print(f"URL: {req.url}")
    print(f"Headers: {req.headers}")
    # Decode body if it's bytes (common for JSON/text)
    body = req.body.decode('utf-8') if isinstance(req.body, bytes) else req.body
    print(f"Body: {body}")
    print(f"--- End Log ---")


def remote_analysis(n_batch: int, prompt: tuple[str], output_path: Path, model: str = "us.meta.llama3-3-70b-instruct-v1:0", tools: list = None) -> list:
    logging.info(f"Sending request to remote API with model {model}...")

    try:
        response = requests.post(
                API_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Team-ID": team_id,
                    "X-API-Token": api_key,
                },
                json={
                    "participant_id": team_id,
                    "api_token": api_key,
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_gen_len": 8192,
                    "tools": tools,
                        "tool_config": {
                            "toolChoice": {
                                "tool": {"name": "analysis_results"}
                            }
                        },
                },
                timeout=300
            )
        logging.info(f"Received response {response.json()}")
        log_full_request(response)
        response = response.json()
                        
        if response["content"] and len(response["content"]) > 0:
            text = response["content"][0]["text"]

            json_match = re.search(r'(\[.*\])', text, re.DOTALL)
            text = json_match.group(1).encode().decode('unicode_escape')

            return check_response(n_batch, text, output_path)
        else:
            logging.warning("No response choices received")
            return None
            
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        print(e)
        return None


def check_response(n_batch: int, response: str, output_path: Path) -> list:
    """Check if response is valid JSON and matches expected schema."""
    
    # Check if response was truncated
    if not response.strip().endswith('}') and not response.strip().endswith(']'):
        logging.warning(f"Response appears truncated. Last 50 chars: {repr(response[-50:])}")
        # Try to fix truncated JSON by closing brackets
        response = fix_truncated_json(response)
    try:
        parsed = response
        parsed = json.loads(parsed)
        return parsed if isinstance(parsed, list) else parsed.get("results", [])
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error: {e}")
        logging.error(f"Response preview: {response[:500]}...")
        
        # Try to extract valid JSON array from response
        results = extract_json_array(response)
        if results:
            return results
            
        # Save malformed response for debugging
        save_malformed_response(output_path, response, f"{n_batch}.txt")
        return None


def fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON by closing open brackets."""
    # Count open/close brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    # Add missing closing brackets
    fixed = text.rstrip()
    
    # Try to close any open strings first
    if fixed.count('"') % 2 == 1:
        fixed += '"'
    
    # Close braces and brackets
    fixed += '}' * max(0, open_braces)
    fixed += ']' * max(0, open_brackets)
    
    return fixed


def extract_json_array(text: str) -> list:
    """Try to extract a valid JSON array from potentially malformed response."""
    import re
    
    # Try to find the results array
    patterns = [
        r'"results"\s*:\s*(\[.*\])',  # Look for "results": [...]
        r'(\[\s*\{.*\}\s*\])',  # Look for [{...}]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception as e:
                logging.error(f"Failed to parse extracted JSON array from pattern: {pattern}: {e}")
    
    # Try parsing individual objects
    results = []
    obj_pattern = r'\{\s*"sentence"\s*:.*?"output"\s*:\s*\{[^}]+\}\s*\}'
    for match in re.finditer(obj_pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            results.append(obj)
        except Exception as e:
            logging.error(f"Failed to parse individual JSON object: {e}")
    
    if results:
        logging.info(f"Extracted {len(results)} results from malformed JSON")
        return results
    
    return None


def save_malformed_response(output_path: Path, text: str, filename: str):
    """Save malformed response for debugging."""
    output_path = output_path / "malformed_responses"
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(text)
    logging.info(f"Saved malformed response to {filepath}")


def process_csv(
    input_csv: str,
    output_dir: str = "analysis_outputs",
    model: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    vllm: bool = True,
    batch_size: int = 10,
    start_from: int = 0,
):
    """
    Process a CSV file and generate JSON analysis results.
    
    Args:
        input_csv: Path to input CSV with 'text' column
        output_dir: Directory for output JSON files
        model: Model name on vLLM serverdh
        vllm: Whether to use local server or remote API
        batch_size: Number of sentences per batch
        start_from: Row index to start from (for resuming)
    """
    # Setup
    output_dir = output_dir + ("_qwen2.5" if vllm else "_llama3.3")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} rows from {input_csv}")
    
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    all_results = []
    failed_batches = []
    total_batches = (len(df) - start_from) // batch_size + 1
    
    for i in range(start_from, len(df), batch_size):
        batch_num = (i - start_from) // batch_size + 1
        end_idx = min(i + batch_size, len(df))
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Batch {batch_num}/{total_batches}: rows {i}-{end_idx-1}")
        logging.info(f"{'='*50}")
        
        # Get sentences for this batch
        sentences = df['text'].iloc[i:end_idx].values.tolist()
        
        # analyse with retry
        results = None
        for attempt in range(3):
            results = analyse_sentences(batch_num, sentences, output_path, model=model, vllm=vllm)
            
            if results:
                break
            logging.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)
        
        if results:
            all_results.extend(results)
            
            # Save checkpoint
            checkpoint_file = output_path / f"batch_{i:05d}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"✓ Saved {len(results)} results to {checkpoint_file}")
        else:
            failed_batches.append(i)
            logging.error(f"✗ Batch {batch_num} failed after 3 attempts")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save combined results
    combined_file = output_path / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\n✓ Combined results saved to {combined_file}")
    
    # Save failed batches for retry
    if failed_batches:
        failed_file = output_path / "failed_batches.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_batches, f)
        logging.warning(f"✗ {len(failed_batches)} batches failed. Saved to {failed_file}")
    
    return all_results, failed_batches


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


def compile_json_checkpoints(checkpoint_dir: str, output_file: str = "all_results.json"):
    """
    Compile all checkpoint JSON files into a single file.
    
    Args:
        checkpoint_dir: Directory containing batch_*.json files
        output_file: Output file path
    """
    checkpoint_path = Path(checkpoint_dir)
    output_file = checkpoint_path / output_file
    all_results = []
    
    for json_file in sorted(checkpoint_path.glob("batch_*.json")):
        with open(json_file, 'r') as f:
            results = json.load(f)
            all_results.extend(results)
        logging.info(f"Loaded {len(results)} results from {json_file.name}")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"✓ Compiled {len(all_results)} total results to {output_file}")
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse stereotype sentences using LLM model")
    parser.add_argument("--input", "-i", default="stereotypes.csv", help="Input CSV file")
    parser.add_argument("--output-dir", "-o", default="analysis_outputs", help="Output directory")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", help="Model name")
    parser.add_argument("--vllm", "-v", action="store_true", help="Use vLLM local server")

    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size")
    parser.add_argument("--start-from", "-s", type=int, default=0, help="Which row index to start from (in case of resume)")
    parser.add_argument("--compile", "-c", action="store_true", help="Compile checkpoint files only")
    parser.add_argument("--update-csv", "-u", action="store_true", help="Update CSV with results")
    parser.add_argument("--results-json", "-r", help="Results JSON for updating CSV")
    
    args = parser.parse_args()
    
    if args.compile:
        # Just compile existing checkpoints
        compile_json_checkpoints(args.output_dir + ("_qwen2.5" if args.vllm else "_llama3.3"))
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
            vllm=args.vllm,
            batch_size=args.batch_size,
            start_from=args.start_from
        )
        
        logging.info(f"\n{'='*50}")
        logging.info(f"COMPLETE: {len(results)} sentences analysed")
        if failed:
            logging.warning(f"FAILED: {len(failed)} batches need retry")
    


