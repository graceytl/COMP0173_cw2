"""
Verify that all texts from the input CSV are present in the compiled results JSON.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def verify_results(input_csv: str, results_json: str, verbose: bool = False):
    """
    Check that all texts in the CSV are present in the results JSON.
    
    Args:
        input_csv: Path to the input CSV file with 'text' column
        results_json: Path to the compiled results JSON file
        verbose: Print detailed information about missing texts
    
    Returns:
        Tuple of (missing_texts, matched_count, total_count)
    """
    # Load CSV
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    csv_texts = set(df['text'].astype(str).tolist())
    total_csv = len(csv_texts)
    
    # Load results JSON
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    # Extract sentences from results
    result_texts = set(r['sentence'] for r in results if 'sentence' in r)
    total_results = len(result_texts)
    
    # Find missing texts (in CSV but not in results)
    missing_texts = csv_texts - result_texts
    
    # Find extra texts (in results but not in CSV)
    extra_texts = result_texts - csv_texts
    
    # Find matched texts
    matched_texts = csv_texts & result_texts
    
    # Print summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total texts in CSV:      {total_csv}")
    print(f"Total texts in results:  {total_results}")
    print(f"Matched texts:           {len(matched_texts)}")
    print(f"Missing from results:    {len(missing_texts)}")
    print(f"Extra in results:        {len(extra_texts)}")
    print(f"{'='*60}")
    
    if len(missing_texts) == 0:
        print("✓ All CSV texts are present in the results!")
    else:
        print(f"✗ {len(missing_texts)} texts are missing from results")
        
        if verbose:
            print(f"\nMissing texts:")
            for i, text in enumerate(sorted(missing_texts), 1):
                # Truncate long texts for display
                display_text = text[:80] + "..." if len(text) > 80 else text
                print(f"  {i}. {display_text}")
    
    if len(extra_texts) > 0 and verbose:
        print(f"\nExtra texts in results (not in CSV):")
        for i, text in enumerate(sorted(extra_texts), 1):
            display_text = text[:80] + "..." if len(text) > 80 else text
            print(f"  {i}. {display_text}")
    
    return missing_texts, len(matched_texts), total_csv


def save_missing_texts(missing_texts: set, output_file: str):
    """Save missing texts to a file for review."""
    with open(output_file, 'w') as f:
        for text in sorted(missing_texts):
            f.write(text + '\n')
    print(f"\n✓ Missing texts saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify all CSV texts are present in compiled results"
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
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information about missing/extra texts"
    )
    parser.add_argument(
        "--save-missing", "-s",
        help="Save missing texts to specified file"
    )
    
    args = parser.parse_args()
    
    # Verify results
    missing, matched, total = verify_results(
        args.input, 
        args.results,
        verbose=args.verbose
    )
    
    # Save missing texts if requested
    if args.save_missing and len(missing) > 0:
        save_missing_texts(missing, args.save_missing)
    
    # Exit with error code if texts are missing
    if len(missing) > 0:
        exit(1)
    else:
        exit(0)
