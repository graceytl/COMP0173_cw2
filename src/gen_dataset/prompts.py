STATEMENT_GEN ="""You are assisting with academic research on bias detection in AI systems. This work is part of a research project studying how to identify discriminatory language in online content from real humans and LLMs.

RESEARCH CONTEXT:
We are building a training dataset to teach AI models to recognize three categories of statements about demographic groups in the UK. This dataset will be used to train classifiers that can detect and flag potentially problematic content.

YOUR TASK:
For the given theme, generate examples of statements. These examples should represent what real-world content might look like, and sound as close to human speech as possible, so the detection model can learn to identify it.

For each statement, ensure it clearly fits the criteria for its category. Use realistic language and context relevant to the theme provided. Examples should be diverse in phrasing and structure. It is fine to utilise online sources for inspiration, but do not copy any text verbatim.

The statements should be marked by surrounding critical tokens with triple equals signs (===) to highlight key phrases that indicates the category, harmfulness, and target group.

STATEMENT CATEGORIES:

1. NORMATIVE (Statements expressing harmful stereotypes or biases):
   - Contains emotionally charged language or blame
   - Makes generalizations about groups
   - Uses divisive "us vs them" framing
   - Example "marked_text": "It is a disgrace that we are putting ===boat arrivals=== in luxury flats while ===our own homeless veterans=== rot on the streets."
   - Label: "NORMATIVE"

2. DESCRIPTIVE (Factual policy/legal statements):
   - References actual UK law, policy, or regulations
   - Uses neutral, formal language
   - Explains differences based on legal criteria, not bias
   - Example "marked_text": "Local authorities allocate ===social housing=== based on ===statutory need and a habitual residence test==="
   - Label: "DESCRIPTIVE"

3. CORRELATION (Statistical/demographic observations):
   - Reports statistical patterns or trends
   - Lacks emotional language or value judgments
   - Uses data-focused terminology
   - Could be harmful or neutral depending on context, or if misapplied
   - Example "marked_text": "===Census data=== suggests that new ===migrant=== populations are statistically more likely to reside in the private rented sector."
   - Example 2 "marked_text": "The number of ===asylum seekers=== arriving by sea has ===increased by 30%=== compared to last year, which has caused ===significant disruption to local communities==="
   - Label: "CORRELATION"

DETAILED INSTRUCTIONS:
   
THEME: {theme}

RELEVANT TERMS (This is a NON-EXHAUSTIVE list of example terms given for extra context, whilst it is not necessary to use all of them, incorporate them as appropriate.): {flags}

MORE EXAMPLES: {examples}

REQUIRED OUTPUT:
Generate EXACTLY 5 examples of each type (5 NORMATIVE, 5 DESCRIPTIVE, 5 CORRELATION for 15 TOTAL statements). Format as a JSON array with this structure:
[
  {{
    "marked_text": "statement with ===marked tokens===",
    "label": "normative",
  }},
  {{
    "marked_text": "statement with ===marked tokens===",
    "label": "descriptive",
  }},
  {{
    "marked_text": "statement with ===marked tokens===",
    "label": "correlative",
  }},
  ... (4 more sets)
]

Ensure all fields are filled with realistic, complete examples that would help train a detection model.

"""