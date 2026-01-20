"""Check CSV for problematic characters that could break JSON parsing."""
import pandas as pd

df = pd.read_csv('stereotypes.csv')
print(f'Total rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

# Check for texts with problematic characters
problem_rows = []
for i, text in enumerate(df['text']):
    issues = []
    if pd.isna(text):
        issues.append('NaN')
    else:
        text_str = str(text)
        if '"' in text_str:
            issues.append('quotes')
        if '\n' in text_str or '\r' in text_str:
            issues.append('newline')
        if '\\' in text_str:
            issues.append('backslash')
        if len(text_str) > 500:
            issues.append(f'long({len(text_str)})')
    
    if issues:
        problem_rows.append((i, issues, str(text)[:100] if not pd.isna(text) else 'NaN'))

print(f'\nProblem rows: {len(problem_rows)}')
for row, issues, preview in problem_rows[:30]:
    print(f'  Row {row}: {issues} -> {repr(preview)}')

# Also check if any text when converted to JSON would be problematic
print('\n--- Testing JSON serialization ---')
import json
for i, text in enumerate(df['text'].head(100)):
    try:
        json.dumps(str(text))
    except Exception as e:
        print(f'Row {i} JSON error: {e}')
        print(f'  Text: {repr(str(text)[:100])}')
