import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    # api_version="2023-03-15-preview",
    api_version="2024-02-15-preview",
    azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

# Load the CSV file into a DataFrame
file_path = './og_chats.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

result_df = pd.DataFrame(columns=['dialog'])

model = AutoModelForCausalLM.from_pretrained( 
    "./PSTax/",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

model2 = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3.5-mini-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("./PSTax/")  

# Iterate over the summary and metadata columns
for i,dialog in enumerate(df['dialog']): 
	
	# if(i==5):
	# 	break
	
	print(i,"\n")
     
    # Create the prompt using both summary and metadata with triple quotes
	rv= ''' Provide a privacy preserving summary for the following conversation - '''
	prompt = f''' {rv} \n <BEGIN CONVERSATION> \n {dialog} \n <END CONVERSATION>'''

	messages = [ 
		{"role": "user", "content": prompt}, 
	] 

	pipe = pipeline( 
		"text-generation", 
		model=model, 
		tokenizer=tokenizer, 
	) 

	pipe2 = pipeline( 
		"text-generation", 
		model=model2, 
		tokenizer=tokenizer, 
	) 

	generation_args = { 
		"max_new_tokens": 2048, 
		"return_full_text": False, 
		"temperature": 0.0, 
		"do_sample": False, 
	} 

	output = pipe(messages, **generation_args) 
	output2 = pipe2(messages, **generation_args)

	# response = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": "You are a useful AI Assistant."},
    #     {"role": "user", "content": prompt},

    # ])

	result_df.at[i, 'dialog'] = dialog  
	result_df.at[i, 'Phinetuned Model (Phi-3)'] = output[0]['generated_text']
	result_df.at[i, 'Base Model (Phi-3)'] = output2[0]['generated_text']
	# result_df.at[i, 'GPT-4o'] = response.choices[0].message.content

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/results_comparisons.csv', index=False, encoding='utf-8')  # Replace with the actual file path