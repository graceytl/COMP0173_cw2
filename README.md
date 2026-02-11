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

### References
[1]	T. King, Z. Wu, A. Koshiyama, E. Kazim, and P. Treleaven, ‘HEARTS: A Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection’, arXiv.org. Accessed: Nov. 13, 2025. [Online]. Available: https://arxiv.org/abs/2409.11579v3
[2]	N. Demchak, X. Guan, Z. Wu, Z. Xu, A. Koshiyama, and E. Kazim, ‘Assessing Bias in Metric Models for LLM Open-Ended Generation Bias Benchmarks’, Oct. 14, 2024, arXiv: arXiv:2410.11059. doi: 10.48550/arXiv.2410.11059.
[3]	V. K. Felkner, H.-C. H. Chang, E. Jang, and J. May, ‘WinoQueer: A Community-in-the-Loop Benchmark for Anti-LGBTQ+ Bias in Large Language Models’, Oct. 17, 2024, arXiv: arXiv:2306.15087. doi: 10.48550/arXiv.2306.15087.
[4]	A. Jha, A. Davani, C. K. Reddy, S. Dave, V. Prabhakaran, and S. Dev, ‘SeeGULL: A Stereotype Benchmark with Broad Geo-Cultural Coverage Leveraging Generative Models’, May 19, 2023, arXiv: arXiv:2305.11840. doi: 10.48550/arXiv.2305.11840.
[5]	Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, ‘ALBERT: A Lite BERT for Self-supervised Learning of Language Representations’, Feb. 09, 2020, arXiv: arXiv:1909.11942. doi: 10.48550/arXiv.1909.11942.
[6]	R. Goldshmidt and M. Horovicz, ‘TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation’, Jul. 22, 2024, arXiv: arXiv:2407.10114. doi: 10.48550/arXiv.2407.10114.
[7]	M. T. Ribeiro, S. Singh, and C. Guestrin, ‘“Why Should I Trust You?”: Explaining the Predictions of Any Classifier’, in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, in KDD ’16. New York, NY, USA: Association for Computing Machinery, Aug. 2016, pp. 1135–1144. doi: 10.1145/2939672.2939778.
[8]	R. Görge et al., ‘Textual Data Bias Detection and Mitigation -- An Extensible Pipeline with Experimental Evaluation’, Dec. 12, 2025, arXiv: arXiv:2512.10734. doi: 10.48550/arXiv.2512.10734.
[9]	‘State of HATE 2024 - Pessimism, decline and the rising Radical Right’, HOPE not hate. Accessed: Feb. 11, 2026. [Online]. Available: https://hopenothate.org.uk/state-of-hate-2024/
[10]	E. Lambert and J. D. Scientist, ‘Using self-hosted Large Language Models (LLMs) securely in Government – Digital trade’. Accessed: Feb. 11, 2026. [Online]. Available: https://digitaltrade.blog.gov.uk/2025/07/09/using-self-hosted-large-language-models-llms-securely-in-government/
[11]	S. L. Blodgett, G. Lopez, A. Olteanu, R. Sim, and H. Wallach, ‘Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets’, in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli, Eds, Online: Association for Computational Linguistics, Aug. 2021, pp. 1004–1015. doi: 10.18653/v1/2021.acl-long.81.
[12]	‘Ethnic group - Census Maps, ONS’. Accessed: Feb. 11, 2026. [Online]. Available: https://www.ons.gov.uk/census/maps/choropleth/identity/ethnic-group/ethnic-group-tb-20b/asian-asian-british-or-asian-welsh-bangladeshi
[13]	‘Religion - Census Maps, ONS’. Accessed: Feb. 11, 2026. [Online]. Available: https://www.ons.gov.uk/census/maps/choropleth/identity/religion/religion-tb/no-religion
[14]	N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman, ‘CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models’, Sep. 30, 2020, arXiv: arXiv:2010.00133. doi: 10.48550/arXiv.2010.00133.
[15]	A. Névéol, Y. Dupont, J. Bezançon, and K. Fort, ‘French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English’, in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds, Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 8521–8531. doi: 10.18653/v1/2022.acl-long.583.
[16]	E. Levon, D. Sharma, and C. Ilbury, ‘Speaking Up’, Nov. 2022, Accessed: Feb. 11, 2026. [Online]. Available: https://www.suttontrust.com/our-research/speaking-up-accents-social-mobility/
[17]	‘Qwen/Qwen2.5-7B-Instruct · Hugging Face’. Accessed: Feb. 11, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
[18]	‘meta-llama/Llama-3.3-70B-Instruct · Hugging Face’. Accessed: Feb. 11, 2026. [Online]. Available: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
[19]	Y. Liu, ‘Quantifying Stereotypes in Language’, in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), Y. Graham and M. Purver, Eds, St. Julian’s, Malta: Association for Computational Linguistics, Mar. 2024, pp. 1223–1240. doi: 10.18653/v1/2024.eacl-long.74.