from transformers import pipeline
import numpy as np
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from datasets import Dataset

# Select the random sample of observations to use methods on
def sample_observations(file_path, k, seed):
    data = pd.read_csv(file_path)
    
    combinations = data.groupby(['group'])
    
    sampled_data = pd.DataFrame(columns=data.columns)
    
    for name, group in combinations:
        same_label = group[group['predicted_label'] == group['actual_label']]
        diff_label = group[group['predicted_label'] != group['actual_label']]
        
        if len(same_label) >= k:
            same_sample = same_label.sample(n=k, random_state=seed)
        else:
            same_sample = same_label
        
        if len(diff_label) >= k:
            diff_sample = diff_label.sample(n=k, random_state=seed)
        else:
            diff_sample = diff_label
        
        sampled_data = pd.concat([sampled_data, same_sample, diff_sample], axis=0)
    
    sampled_data.reset_index(drop=True, inplace=True)
    
    print(sampled_data)
    sampled_data = Dataset.from_pandas(sampled_data)
    
    return sampled_data


# Define function to compute SHAP values
def shap_analysis(sampled_data, model_path, device=0, batch_size=32):
    """
    Compute SHAP values for a Dataset object.
    
    Args:
        sampled_data: HuggingFace Dataset object with columns: text, group, predicted_label, actual_label
        model_path: Path to the trained model
        device: Device to use (0 for GPU, -1 for CPU)
        batch_size: Batch size for pipeline processing
    
    Returns:
        pd.DataFrame with SHAP values per token
    """
    pipe = pipeline("text-classification", model=model_path, top_k=None, device=device, batch_size=batch_size)
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')  
    explainer = shap.Explainer(pipe, masker)

    results = []
    class_names = ['LABEL_0', 'LABEL_1']
    
    # Process all texts in batch for SHAP
    all_texts = sampled_data['text']
    print(f"Computing SHAP values for {len(all_texts)} samples...")
    shap_values = explainer(all_texts)
    
    for index in range(len(sampled_data)):
        row = sampled_data[index]
        
        print(f"Processing {index+1}/{len(sampled_data)} - Categorisation: {row['group']} - Predicted: {row['predicted_label']} - Actual: {row['actual_label']}")
        label_index = class_names.index("LABEL_1")  
        
        specific_shap_values = shap_values[index, :, label_index].values
        
        tokens = re.findall(r'\w+', row['text'])
        for token, value in zip(tokens, specific_shap_values):
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_shap': value,
                'sentence': row['text'],
                'categorisation': row['group'],
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label']
            })
                
    return pd.DataFrame(results)


# Define function to compute LIME values 
def custom_tokenizer(text):
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path, device=0, batch_size=32):
    """
    Compute LIME values for a Dataset object.
    
    Args:
        sampled_data: HuggingFace Dataset object with columns: text, group, predicted_label, actual_label
        model_path: Path to the trained model
        device: Device to use (0 for GPU, -1 for CPU)
        batch_size: Batch size for pipeline processing
    
    Returns:
        pd.DataFrame with LIME values per token
    """
    pipe = pipeline("text-classification", model=model_path, top_k=None, device=device, batch_size=batch_size)
    
    def predict_proba(texts):
        # Pipeline handles batching internally with batch_size parameter
        preds = pipe(texts, top_k=None)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        return probabilities    
    
    explainer = LimeTextExplainer(class_names=['LABEL_0', 'LABEL_1'], split_expression=lambda x: custom_tokenizer(x))  
    
    results = []
    
    print(f"Computing LIME values for {len(sampled_data)} samples...")
    for index in range(len(sampled_data)):
        row = sampled_data[index]
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        exp = explainer.explain_instance(text_input, predict_proba, num_features=len(tokens), num_samples=100)
        
        print(f"Processing {index+1}/{len(sampled_data)} - Categorisation: {row['group']} - Predicted: {row['predicted_label']} - Actual: {row['actual_label']}")

        explanation_list = exp.as_list(label=1)
        
        token_value_dict = {token: value for token, value in explanation_list}

        for token in tokens:
            value = token_value_dict.get(token, 0)  
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_lime': value,
                'sentence': text_input,
                'categorisation': row['group'],
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label']
            })

    return pd.DataFrame(results)


# Define helper functions
def compute_cosine_similarity(vector1, vector2):
    """Compute cosine similarity between two vectors (accepts lists or arrays)."""
    v1 = np.array(vector1).reshape(1, -1)
    v2 = np.array(vector2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

def compute_pearson_correlation(vector1, vector2):
    """Compute Pearson correlation between two vectors (accepts lists or arrays)."""
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    if len(v1) < 2 or len(v2) < 2:
        return np.nan
    correlation, _ = pearsonr(v1, v2)
    return correlation

def to_probability_distribution(values):
    """Convert values to a probability distribution."""
    values = np.array(values, dtype=float)
    min_val = np.min(values)
    if min_val < 0:
        values = values + abs(min_val)
    total = np.sum(values)
    if total > 0:
        values = values / total
    return values

def compute_js_divergence(vector1, vector2):
    """Compute Jensen-Shannon divergence between two vectors (accepts lists or arrays)."""
    prob1 = to_probability_distribution(vector1)
    prob2 = to_probability_distribution(vector2)
    return jensenshannon(prob1, prob2) 