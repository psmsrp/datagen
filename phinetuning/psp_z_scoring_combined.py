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
file_path = './results/results_different_summaries.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)
file_path = './scores/different_summaries.csv'  # Replace with the actual file path
df2 = pd.read_csv(file_path)
file_path = './scores/binary_different_summaries.csv'  # Replace with the actual file path
df3 = pd.read_csv(file_path)
print(df)

for i,(gt,m1,m2,m3,bm,gpt) in enumerate(zip(df2['Score GT_Summary'],df2['Score Model_1'],df2['Score Model_2 (Early Stop)'], df2['Score Model_3 (Only Good)'], df2['Score Base Model (Phi-3.5)'], df2['Score GPT-4o'])): 
    # Apply the function to the 'scores' column and update it
    df.at[i,'Score GT_Summary'] = gt
    df.at[i,'Score Model_1'] = m1
    df.at[i,'Score Model_2 (Early Stop)'] = m2
    df.at[i,'Score Model_3 (Only Good)'] = m3
    df.at[i,'Score Base Model (Phi-3.5)'] = bm
    df.at[i,'Score GPT-4o'] = gpt

for i,(gt,m1,m2,m3,bm,gpt) in enumerate(zip(df3['Score GT_Summary'],df3['Score Model_1'],df3['Score Model_2 (Early Stop)'], df3['Score Model_3 (Only Good)'], df3['Score Base Model (Phi-3.5)'], df3['Score GPT-4o'])): 
    # Apply the function to the 'scores' column and update it
    df.at[i,'Label GT_Summary'] = gt
    df.at[i,'Label Model_1'] = m1
    df.at[i,'Label Model_2 (Early Stop)'] = m2
    df.at[i,'Label Model_3 (Only Good)'] = m3
    df.at[i,'Label Base Model (Phi-3.5)'] = bm
    df.at[i,'Label GPT-4o'] = gpt

# Display updated DataFrame
print(df)

df.to_csv('./scores/final_different_summaries.csv', index=False, encoding='utf-8')  # Replace with the actual file path




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