"""
Clean up results by removing extra texts and reprocessing missing texts.
- Removes texts from results that are not in the original CSV
- Reruns the LLM to get results for texts missing from results
"""

import argparse
import json
import logging
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from src.dataset_gen.scripts.prompt import PROMPT_TEMPLATE, EXAMPLES

logging.basicConfig(level=logging.INFO)

# Connect to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="python"
)

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


def analyze_sentences(sentences: list, model: str = "fusechat") -> list:
    """Analyze sentences using local vLLM server with structured JSON output."""
    clean_sentences = []
    for s in sentences:
        clean = str(s).replace('\n', ' ').replace('\r', ' ')
        clean_sentences.append(clean)
    
    prompt = SYSTEM_PROMPT + f"\n\nAnalyze these sentences:\n{json.dumps(clean_sentences, ensure_ascii=False)}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6450,
            top_p=0.75,
            temperature=0.1,
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
            
            # Fix truncated JSON if needed
            if not text.strip().endswith('}') and not text.strip().endswith(']'):
                text = fix_truncated_json(text)
            
            try:
                parsed = json.loads(text)
                return parsed.get("results", [])
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error: {e}")
                return None
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return None


def fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON by closing open brackets."""
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    fixed = text.rstrip()
    
    if fixed.count('"') % 2 == 1:
        fixed += '"'
    
    fixed += '}' * max(0, open_braces)
    fixed += ']' * max(0, open_brackets)
    
    return fixed


def clean_and_reprocess(
    input_csv: str,
    results_json: str,
    output_json: str = None,
    batch_size: int = 5,
    model: str = "fusechat",
    dry_run: bool = False
):
    """
    Clean results and reprocess missing texts.
    
    Args:
        input_csv: Path to the input CSV file
        results_json: Path to the compiled results JSON
        output_json: Path for the cleaned output JSON (default: overwrite input)
        batch_size: Batch size for reprocessing
        model: Model name on vLLM server
        dry_run: If True, only report what would be done without making changes
    """
    # Load CSV
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    csv_texts = set(df['text'].astype(str).tolist())
    
    # Load results
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    # Create mapping of sentence -> result
    results_map = {r['sentence']: r for r in results if 'sentence' in r}
    result_texts = set(results_map.keys())
    
    # Find missing and extra texts
    missing_texts = csv_texts - result_texts
    extra_texts = result_texts - csv_texts
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    print(f"Total texts in CSV:      {len(csv_texts)}")
    print(f"Total texts in results:  {len(result_texts)}")
    print(f"Missing from results:    {len(missing_texts)}")
    print(f"Extra in results:        {len(extra_texts)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("\n[DRY RUN] Would perform the following actions:")
        print(f"  - Remove {len(extra_texts)} extra texts from results")
        print(f"  - Reprocess {len(missing_texts)} missing texts")
        if len(missing_texts) > 0:
            print("\nMissing texts to reprocess:")
            for i, text in enumerate(sorted(missing_texts)[:10], 1):
                display_text = text[:70] + "..." if len(text) > 70 else text
                print(f"  {i}. {display_text}")
            if len(missing_texts) > 10:
                print(f"  ... and {len(missing_texts) - 10} more")
        return
    
    # Step 1: Remove extra texts
    if len(extra_texts) > 0:
        print(f"\n[1/2] Removing {len(extra_texts)} extra texts from results...")
        for text in extra_texts:
            del results_map[text]
        print(f"  ✓ Removed {len(extra_texts)} extra texts")
    else:
        print("\n[1/2] No extra texts to remove")
    
    # Step 2: Reprocess missing texts
    if len(missing_texts) > 0:
        print(f"\n[2/2] Reprocessing {len(missing_texts)} missing texts...")
        missing_list = sorted(missing_texts)
        new_results = []
        failed_texts = []
        
        total_batches = (len(missing_list) + batch_size - 1) // batch_size
        
        for i in range(0, len(missing_list), batch_size):
            batch_num = i // batch_size + 1
            batch = missing_list[i:i + batch_size]
            
            logging.info(f"  Batch {batch_num}/{total_batches}: processing {len(batch)} texts...")
            
            # Try up to 3 times
            results_batch = None
            for attempt in range(3):
                results_batch = analyze_sentences(batch, model=model)
                if results_batch:
                    break
                logging.warning(f"    Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)
            
            if results_batch:
                new_results.extend(results_batch)
                logging.info(f"    ✓ Got {len(results_batch)} results")
            else:
                failed_texts.extend(batch)
                logging.error("    ✗ Batch failed after 3 attempts")
            
            time.sleep(0.5)
        
        # Add new results to map
        for r in new_results:
            if 'sentence' in r:
                results_map[r['sentence']] = r
        
        print(f"  ✓ Processed {len(new_results)} new results")
        if failed_texts:
            print(f"  ✗ Failed to process {len(failed_texts)} texts")
            
            # Save failed texts for later retry
            failed_file = Path(results_json).parent / "failed_texts.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_texts, f, indent=2)
            print(f"  Saved failed texts to {failed_file}")
    else:
        print("\n[2/2] No missing texts to reprocess")
    
    # Save cleaned results
    cleaned_results = list(results_map.values())
    output_path = output_json or results_json
    
    # Backup original if overwriting
    if output_path == results_json:
        backup_path = results_json.replace('.json', '_backup.json')
        with open(backup_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Backed up original to {backup_path}")
    
    with open(output_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    print(f"✓ Saved cleaned results ({len(cleaned_results)} entries) to {output_path}")
    
    # Final verification
    final_texts = set(r['sentence'] for r in cleaned_results if 'sentence' in r)
    still_missing = csv_texts - final_texts
    
    print(f"\n{'='*60}")
    print("FINAL STATUS")
    print(f"{'='*60}")
    print(f"Total results:    {len(cleaned_results)}")
    print(f"Still missing:    {len(still_missing)}")
    
    if len(still_missing) == 0:
        print("✓ All CSV texts now have results!")
    else:
        print(f"✗ {len(still_missing)} texts still need processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean results and reprocess missing texts"
    )
    parser.add_argument(
        "--input", "-i",
        default="stereotypes.csv",
        help="Input CSV file with 'text' column"
    )
    parser.add_argument(
        "--results", "-r",
        default="analysis_outputs/compiled_results.json",
        help="Compiled results JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file (default: overwrite results file)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Batch size for reprocessing"
    )
    parser.add_argument(
        "--model", "-m",
        default="fusechat",
        help="Model name on vLLM server"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Only show what would be done, don't make changes"
    )
    
    args = parser.parse_args()
    
    clean_and_reprocess(
        input_csv=args.input,
        results_json=args.results,
        output_json=args.output,
        batch_size=args.batch_size,
        model=args.model,
        dry_run=args.dry_run
    )
