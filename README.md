# Repo for COMP0173 Coursework 2
This repo holds the code for SN 17004522's submission for COMP0173: Artificial Intelligence for Sustainable Development.

This repo uses Python 3.10.

## Repo set-up
To clone this repo run: ```https://github.com/graceytl/COMP0173_cw2.git```
This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Package dependencies and requirements can be edited and found under `pyproject.toml`.

To install all dependencies and set-up the project, run `uv sync` on the top level of the repo.

### .env
In order to run any parts of the repo utilising Llama-3-3-70B-Instruct (or other models for testing) using the Bedrock API, a `.env` file should be added on the local machine.

```
TEAM_ID=<your-team-id>
API_KEY=<your-api-key>
```
### Project structure
This repo is split into multiple components under the `src` folder. Relevant filse for the coursework have been noted below:

```
src
|
|- dataset-gen
|   |- crows_pairs (removal of us-centric crows-pairs)
|   |- scripts (!! All notebooks for stereotype detection and linguistic indicator extraction)
|- gen_dataset
|   |- get_prompts.py (!! Script for generating LLM statements using vLLM)
|   |- prompts.py
|   |- themes.py (Identified themes relevant to the UK)
|   |- models.py
|- HEARTS
|   |- custom_albertv2 (!! Model from retraining)
|   |- model_training.ipynb (!! Replication of original paper code)
|   |- model_adaption.ipynb (!! Re-training the model with adapted dataset)
|- model_explainability
|   |- shap_lime_analysis_custom_ds.ipynb (!! Running on new data)
|   |- shap_lime_analysis.ipynb (!! Replicating original repo's SHAP LIME analysis)
|- quantifying_stereotypes
|   |- model (!! The logistic regression model stored as .joblib)
|   |- **.ipynb (!! The notebooks for training the logsitic regression model on extracted linguistic features)
|- stereotype_final_all_info.csv (The final table of generated stereotypes and all sociolinguistic features, plus the ALBERT-v2 scores as predicted by model)
```
## Implementation

### Proposed pipeline
1. Manual extraction of target groups identified in the UK from EMGSD.
2. LLM generation of additional stereotypes for groups not found in EMGSD.
3. Filtering the database (remove US-centric CrowS-Pairs statements, etc.).
4. Identifying "potential" stereotypes.
5. Generate linguistic indicators.
6. Train a logistic regression model on the linguistic indicators to create ground-truth labels for whether a statement is a stereotype or not.
7. Fine-tune ALBERT-v2 model for classification.

### Dataset generation
Using a local [vLLM](https://docs.vllm.ai/en/latest/) setup, a [FuseChat model](https://huggingface.co/FuseAI/FuseChat-7B-VaRM) is run for the stereotype sentence generations.
The models were mostly run on a RTX 4070 Ti Super.

```shell
uv run vllm serve "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" \
  --download-dir /scratch0/gracelin/qwen2.5/.cache \--dtype bfloat16 \
  --api-key python \
  --gpu-memory-utilization 0.9
```
### Dataset refinement
Detecting which statements from the full set of a combination of extracted relevant [EMGSD](https://huggingface.co/datasets/holistic-ai/EMGSD) statements, the LLM generated statements, and manually curated statements, is performed using `src/dataset_gen/scripts/i_i_stereotype_detection.py`

### Extracting linguistic indicators
Linguistic indicators are extracted using Llama-3.3-70B-Instruct, chosing for it's high reasoning. The script `src/dataset_gen/scripts/ii_i_generate_linguistic_indicators.py` takes a .csv of texts, and processes batches of statements to extract the following linguistic features:

Features for scsc:
- `target`
- `information`
- `ling_form`
- `target_type`
- `connotation`
- `gram_form`
- `situation`
- `generalization`
- `situation_evaluation`

### Quantifying stereotypes
Following the method from ["detecting-linguistic-indicators"](https://github.com/r-goerge/Detecting-Linguistic-Indicators-for-Stereotype-Assessment-with-LLMs), a logistic regression model is trained using the human-labelled BWS score from ["quantifying-stereotypes-in-language"](https://github.com/nlply/quantifying-stereotypes-in-language) as the ground-truth pairs.

The LR model is only trained on the sentences that appear in the above repo.

The input features are compiled from the lingusitic features as follows:
- `generalization_category_label` = [`ling_form`]_[`target_type`]
- `connotation` = `connotation`
- `gram_form` = `gram_form`
- `generalization_situation` = [`situation`]_[`generalization`]
- `situation_evaluation` = `situation_evaluation`

### HEARTS replication
Please see `model_training.ipynb` for the replication of the original paper.
Please refer to `model_adaption.ipynb` for ALBERT-v2 fine-tuning on the new dataset.

### Explainability (SHAP & LIME)
The model's outputs for whether a statement is a stereotype or not, and the corresponding SHAP & LIME graphs can be found under `src/model_explainability/`
