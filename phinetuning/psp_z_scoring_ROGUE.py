import pandas as pd
from datasets import load_metric,Dataset, concatenate_datasets
import random
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import re

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    # api_version="2023-03-15-preview",
    api_version="2024-02-15-preview",
    azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

# Load the rouge metric
rouge_metric = load_metric("rouge", trust_remote_code=True)
torch.random.manual_seed(0) 

def calculate_rouge(row):
	true_summary = row['GT_Summary']
	rouge_scores1 = rouge_metric.compute(predictions=[row['Model_1']], references=[true_summary])
	rouge_scores2 = rouge_metric.compute(predictions=[row['Model_2 (Early Stop)']], references=[true_summary])
	rouge_scores3 = rouge_metric.compute(predictions=[row['Model_3 (Only Good)']], references=[true_summary])
	rouge_scores4 = rouge_metric.compute(predictions=[row['Model_4 (With Corrections)']], references=[true_summary])
	rouge_scores5 = rouge_metric.compute(predictions=[f"<BEGIN SUMMARY>\n\n{row['Model_5 (Base Model Phi-3.5 with DPO)']}\n\n<END SUMMARY>"], references=[true_summary])
	rouge_scores0 = rouge_metric.compute(predictions=[row['Base Model (Phi-3.5)']], references=[true_summary])
	rouge_scores_gpt = rouge_metric.compute(predictions=[row['GPT-4o']], references=[true_summary])
	
	row['Model_1 rouge1']=rouge_scores1['rouge1'].mid.fmeasure
	row['Model_2 (Early Stop) rouge1']=rouge_scores2['rouge1'].mid.fmeasure
	row['Model_3 (Only Good) rouge1']=rouge_scores3['rouge1'].mid.fmeasure
	row['Model_4 (With Corrections) rouge1']=rouge_scores4['rouge1'].mid.fmeasure
	row['Model_5 (Base Model Phi-3.5 with DPO) rouge1']=rouge_scores5['rouge1'].mid.fmeasure
	row['Base Model (Phi-3.5) rouge1']=rouge_scores0['rouge1'].mid.fmeasure
	row['GPT-4o rouge1']=rouge_scores_gpt['rouge1'].mid.fmeasure


	row['Model_1 rouge2']=rouge_scores1['rouge2'].mid.fmeasure
	row['Model_2 (Early Stop) rouge2']=rouge_scores2['rouge2'].mid.fmeasure
	row['Model_3 (Only Good) rouge2']=rouge_scores3['rouge2'].mid.fmeasure
	row['Model_4 (With Corrections) rouge2']=rouge_scores4['rouge2'].mid.fmeasure
	row['Model_5 (Base Model Phi-3.5 with DPO) rouge2']=rouge_scores5['rouge2'].mid.fmeasure
	row['Base Model (Phi-3.5) rouge2']=rouge_scores0['rouge2'].mid.fmeasure
	row['GPT-4o rouge2']=rouge_scores_gpt['rouge2'].mid.fmeasure

	row['Model_1 rougeL']=rouge_scores1['rougeL'].mid.fmeasure
	row['Model_2 (Early Stop) rougeL']=rouge_scores2['rougeL'].mid.fmeasure
	row['Model_3 (Only Good) rougeL']=rouge_scores3['rougeL'].mid.fmeasure
	row['Model_4 (With Corrections) rougeL']=rouge_scores4['rougeL'].mid.fmeasure
	row['Model_5 (Base Model Phi-3.5 with DPO) rougeL']=rouge_scores5['rougeL'].mid.fmeasure
	row['Base Model (Phi-3.5) rougeL']=rouge_scores0['rougeL'].mid.fmeasure
	row['GPT-4o rougeL']=rouge_scores_gpt['rougeL'].mid.fmeasure

	row['Model_1 rougeLsum']=rouge_scores1['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores1 else None
	row['Model_2 (Early Stop) rougeLsum']=rouge_scores2['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores2 else None
	row['Model_3 (Only Good) rougeLsum']=rouge_scores3['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores3 else None
	row['Model_4 (With Corrections) rougeLsum']=rouge_scores4['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores4 else None
	row['Model_5 (Base Model Phi-3.5 with DPO) rougeLsum']=rouge_scores5['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores5 else None
	row['Base Model (Phi-3.5) rougeLsum']=rouge_scores0['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores0 else None
	row['GPT-4o rougeLsum']=rouge_scores_gpt['rougeLsum'].mid.fmeasure if 'rougeLsum' in rouge_scores_gpt else None
	
	print(row)
	
	return row

# Apply the ROUGE calculation for each row in the dataframe
# Load the dataset
file_path = './results/results_different_summaries.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)
df = df.apply(calculate_rouge, axis=1)

df.to_csv('./scores/different_summaries_rogue.csv', index=False, encoding='utf-8')  # Replace with the actual file path