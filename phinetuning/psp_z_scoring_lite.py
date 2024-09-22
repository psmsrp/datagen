import pandas as pd
from datasets import load_metric,Dataset, concatenate_datasets
import random
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import re

# Apply the ROUGE calculation for each row in the dataframe
# Load the dataset
file_path = './scores/binary_different_summaries.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

total=mis_m1=mis_m2=mis_m3=mis_m4=mis_bm=mis_gpt=_01_m1=_01_m2=_01_m3=_01_m4=_01_bm=_01_gpt=_10_m1=_10_m2=_10_m3=_10_m4=_10_bm=_10_gpt=_00_m1=_00_m2=_00_m3=_00_m4=_00_bm=_00_gpt=_11_m1=_11_m2=_11_m3=_11_m4=_11_bm=_11_gpt=0

for i,(gt,m1,m2,m3,m4,bm,gpt) in enumerate(zip(df['Label GT_Summary'],df['Label Model_1'],df['Label Model_2 (Early Stop)'], df['Label Model_3 (Only Good)'],df['Label Model_4 (With Corrections)'], df['Label Base Model (Phi-3.5)'], df['Label GPT-4o'])): 
# for i,(gt,m1,m2,m3,bm,gpt) in enumerate(zip(df['Label GT_Summary'],df['Label Model_1'],df['Label Model_2 (Early Stop)'], df['Label Model_3 (Only Good)'], df['Label Base Model (Phi-3.5)'], df['Label GPT-4o'])): 

	total+=1

	mis_m1= mis_m1 if int(gt)==int(m1) else mis_m1+1
	mis_m2= mis_m2 if int(gt)==int(m2) else mis_m2+1
	mis_m3= mis_m3 if int(gt)==int(m3) else mis_m3+1
	mis_m4= mis_m4 if int(gt)==int(m4) else mis_m4+1
	mis_bm= mis_bm if int(gt)==int(bm) else mis_bm+1
	mis_gpt= mis_gpt if int(gt)==int(gpt) else mis_gpt+1

	_01_m1= _01_m1+1 if int(gt)!=int(m1) and int(gt)==0 else _01_m1
	_01_m2= _01_m2+1 if int(gt)!=int(m2) and int(gt)==0 else _01_m2
	_01_m3= _01_m3+1 if int(gt)!=int(m3) and int(gt)==0 else _01_m3
	_01_m4= _01_m4+1 if int(gt)!=int(m4) and int(gt)==0 else _01_m4
	_01_bm= _01_bm+1 if int(gt)!=int(bm) and int(gt)==0 else _01_bm
	_01_gpt= _01_gpt+1 if int(gt)!=int(gpt) and int(gt)==0 else _01_gpt

	_00_m1= _00_m1+1 if int(gt)==int(m1) and int(gt)==0 else _00_m1
	_00_m2= _00_m2+1 if int(gt)==int(m2) and int(gt)==0 else _00_m2
	_00_m3= _00_m3+1 if int(gt)==int(m3) and int(gt)==0 else _00_m3
	_00_m4= _00_m4+1 if int(gt)==int(m4) and int(gt)==0 else _00_m4
	_00_bm= _00_bm+1 if int(gt)==int(bm) and int(gt)==0 else _00_bm
	_00_gpt= _00_gpt+1 if int(gt)==int(gpt) and int(gt)==0 else _00_gpt

	_11_m1= _11_m1+1 if int(gt)==int(m1) and int(gt)==1  else _11_m1
	_11_m2= _11_m2+1 if int(gt)==int(m2) and int(gt)==1 else _11_m2
	_11_m3= _11_m3+1 if int(gt)==int(m3) and int(gt)==1  else _11_m3
	_11_m4= _11_m4+1 if int(gt)==int(m4) and int(gt)==1  else _11_m4
	_11_bm= _11_bm+1 if int(gt)==int(bm) and int(gt)==1  else _11_bm
	_11_gpt= _11_gpt+1 if int(gt)==int(gpt) and int(gt)==1  else _11_gpt

	_10_m1= _10_m1+1 if int(gt)!=int(m1) and int(gt)==1 else _10_m1
	_10_m2= _10_m2+1 if int(gt)!=int(m2) and int(gt)==1 else _10_m2
	_10_m3= _10_m3+1 if int(gt)!=int(m3) and int(gt)==1 else _10_m3
	_10_m4= _10_m4+1 if int(gt)!=int(m4) and int(gt)==1 else _10_m4
	_10_bm= _10_bm+1 if int(gt)!=int(bm) and int(gt)==1 else _10_bm
	_10_gpt= _10_gpt+1 if int(gt)!=int(gpt) and int(gt)==1 else _10_gpt




# Display results
print("\nBase Model (Phi-3.5) Metrics:")
print(f"Mismatch: {mis_bm}/{total} or {(mis_bm/total):.4f}")
print(f"Good-to-Bad count: {_10_bm:.4f}")
print(f"Bad-to-Bad count: {_00_bm:.4f}")
print(f"Bad-to-Good count: {_01_bm:.4f}")
print(f"Good-to-Good count: {_11_bm:.4f}")
print(f"Positive alignment: {(_01_bm + _11_bm)}/{total} or {((_01_bm + _11_bm)/total):.4f}")
print(f"Negative deviation: {(_00_bm + _10_bm)}/{total} or {((_00_bm + _10_bm)/total):.4f}")

print("-------------------------------------------------------------")


print("\nModel_1 Metrics:")
print(f"Mismatch: {mis_m1}/{total} or {(mis_m1/total):.4f}")
print(f"Good-to-Bad count: {_10_m1:.4f}")
print(f"Bad-to-Bad count: {_00_m1:.4f}")
print(f"Bad-to-Good count: {_01_m1:.4f}")
print(f"Good-to-Good count: {_11_m1:.4f}")
print(f"Positive alignment: {(_01_m1 + _11_m1)}/{total} or {((_01_m1 + _11_m1)/total):.4f}")
print(f"Negative deviation: {(_00_m1 + _10_m1)}/{total} or {((_00_m1 + _10_m1)/total):.4f}")

print("-------------------------------------------------------------")

print("Model_2 (Early Stop) Metrics:")
print(f"Mismatch: {mis_m2}/{total} or {(mis_m2/total):.4f}")
print(f"Good-to-Bad count: {_10_m2:.4f}")
print(f"Bad-to-Bad count: {_00_m2:.4f}")
print(f"Bad-to-Good count: {_01_m2:.4f}")
print(f"Good-to-Good count: {_11_m2:.4f}")
print(f"Positive alignment: {(_01_m2 + _11_m2)}/{total} or {((_01_m2 + _11_m2)/total):.4f}")
print(f"Negative deviation: {(_00_m2 + _10_m2)}/{total} or {((_00_m2 + _10_m2)/total):.4f}")

print("-------------------------------------------------------------")


print("Model_3 (Only Good) Metrics:")
print(f"Mismatch: {mis_m3}/{total} or {(mis_m3/total):.4f}")
print(f"Good-to-Bad count: {_10_m3:.4f}")
print(f"Bad-to-Bad count: {_00_m3:.4f}")
print(f"Bad-to-Good count: {_01_m3:.4f}")
print(f"Good-to-Good count: {_11_m3:.4f}")
print(f"Positive alignment: {(_01_m3 + _11_m3)}/{total} or {((_01_m3 + _11_m3)/total):.4f}")
print(f"Negative deviation: {(_00_m3 + _10_m3)}/{total} or {((_00_m3 + _10_m3)/total):.4f}")

print("-------------------------------------------------------------")


print("Model_4 (With Corrections) Metrics:")
print(f"Mismatch: {mis_m4}/{total} or {(mis_m4/total):.4f}")
print(f"Good-to-Bad count: {_10_m4:.4f}")
print(f"Bad-to-Bad count: {_00_m4:.4f}")
print(f"Bad-to-Good count: {_01_m4:.4f}")
print(f"Good-to-Good count: {_11_m4:.4f}")
print(f"Positive alignment: {(_01_m4 + _11_m4)}/{total} or {((_01_m4 + _11_m4)/total):.4f}")
print(f"Negative deviation: {(_00_m4 + _10_m4)}/{total} or {((_00_m4 + _10_m4)/total):.4f}")

print("-------------------------------------------------------------")


print("GPT-4o:")
print(f"Mismatch: {mis_gpt}/{total} or {(mis_gpt/total):.4f}")
print(f"Good-to-Bad count: {_10_gpt:.4f}")
print(f"Bad-to-Bad count: {_00_gpt:.4f}")
print(f"Bad-to-Good count: {_01_gpt:.4f}")
print(f"Good-to-Good count: {_11_gpt:.4f}")
print(f"Positive alignment: {(_01_gpt + _11_gpt)}/{total} or {((_01_gpt + _11_gpt)/total):.4f}")
print(f"Negative deviation: {(_00_gpt + _10_gpt)}/{total} or {((_00_gpt + _10_gpt)/total):.4f}")


# df.to_csv('./scores/different_summaries_rogue.csv', index=False, encoding='utf-8')  # Replace with the actual file path