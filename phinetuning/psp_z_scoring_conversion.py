import pandas as pd
# from azure.identity import AzureCliCredential, get_bearer_token_provider
# from openai import AzureOpenAI
# import torch 
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

# torch.random.manual_seed(0) 

# token_provider = get_bearer_token_provider(
#     AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
# )

# client = AzureOpenAI(
#     # api_version="2023-03-15-preview",
#     api_version="2024-02-15-preview",
#     azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
#     azure_ad_token_provider=token_provider
# )

# Load the CSV file into a DataFrame
file_path = './scores/binary_different_summaries.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)
print(df)

import re

# Function to extract just the number
# Function to extract just the number
def extract_number(score_str):
    # match = re.search(r'<BEGIN SCORE>\s*(\d+)\s*<END SCORE>', str(score_str))
    # if match:
    #     return int(match.group(1).strip())  # Extract the number as an integer
    # return score_str  # Return original if no match
        return 1 if 'GOOD' in str(score_str).upper()  else 0  # Convert GOOD to 1, BAD to 0

# Apply the function to the 'scores' column and update it
df['Score GT_Summary'] = df['Score GT_Summary'].apply(extract_number)
df['Score Model_1'] = df['Score Model_1'].apply(extract_number)
df['Score Model_2 (Early Stop)'] = df['Score Model_2 (Early Stop)'].apply(extract_number)
df['Score Model_3 (Only Good)'] = df['Score Model_3 (Only Good)'].apply(extract_number)
df['Score Base Model (Phi-3.5)'] = df['Score Base Model (Phi-3.5)'].apply(extract_number)
df['Score GPT-4o'] = df['Score GPT-4o'].apply(extract_number)

# Display updated DataFrame
print(df)

df.to_csv('./scores/different_summaries2.csv', index=False, encoding='utf-8')  # Replace with the actual file path




# # Iterate over the summary and metadata columns
# for i,(dialog,summary) in enumerate(zip(df['dialog'],df['GT_Summary'])): 
	
# 	# if(i==5):
# 	# 	break
	
# 	print(i,"\n")
     
#     # Create the prompt using both summary and metadata with triple quotes
# 	rv= ''' Provide a privacy preserving summary for the following conversation - '''
# 	prompt = f''' {rv} \n\n {dialog} \n\n'''

# 	messages = [ 
# 		{"role": "user", "content": prompt}, 
# 	] 

	
# 	response = client.chat.completions.create(
#     model="hywaygpt4o", # model = "deployment_name".
#     messages=[
#         {"role": "system", "content": "You are a useful AI Assistant that generates Privacy preserving summaries in 50-250 words."},
#         {"role": "user", "content": prompt},

#     ])

# 	result_df.at[i, 'GPT-4o'] = f"<BEGIN SUMMARY>\n\n{response.choices[0].message.content}\n\n<END SUMMARY>"

# # Print the result DataFrame
# print(result_df)

# # Optionally, save the result DataFrame to a new CSV file
# result_df.to_csv('./results/results_different_summaries.csv', index=False, encoding='utf-8')  # Replace with the actual file path