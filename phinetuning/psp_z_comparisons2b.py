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
file_path = './splits_corrected/final_test_split.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)
f2='./results/results_different_summaries_new.csv'
result_df = pd.read_csv(f2)

# model0 = AutoModelForCausalLM.from_pretrained( 
#     "microsoft/Phi-3.5-mini-instruct",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 

# model1 = AutoModelForCausalLM.from_pretrained( 
#     "./PSTax/",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 

# model2 = AutoModelForCausalLM.from_pretrained( 
#     "./PSTax2/",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 

# model3 = AutoModelForCausalLM.from_pretrained( 
#     "./PSTax3/",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 

# model3_4 = AutoModelForCausalLM.from_pretrained( 
#     "./PS4Tax3/",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# )

model3_Q = AutoModelForCausalLM.from_pretrained( 
    "Qwen/Qwen2.5-7B-Instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
)

# model4 = AutoModelForCausalLM.from_pretrained( 
#     "./PSTax4/",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 

# tokenizer0 = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")  
# tokenizer1 = AutoTokenizer.from_pretrained("./PSTax/")  
# tokenizer2 = AutoTokenizer.from_pretrained("./PSTax2/")  
# tokenizer3 = AutoTokenizer.from_pretrained("./PSTax3/")  
# tokenizer3_4 = AutoTokenizer.from_pretrained("./PS4Tax3/")  
tokenizer3_Q = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  
# tokenizer4 = AutoTokenizer.from_pretrained("./PSTax4/")  

# Iterate over the summary and metadata columns
for i,(dialog,summary) in enumerate(zip(df['dialog'],df['summary'])): 
	
	# if(i==5):
	# 	break
	
	print(i,"\n")
     
    # Create the prompt using both summary and metadata with triple quotes
	rv= ''' Provide a privacy preserving summary for the following conversation - '''
	prompt = f''' {rv} \n\n {dialog} \n\n'''

	messages = [ 
		{"role": "user", "content": prompt}, 
	] 

	output5 = "None"
	flag5= False

	# try:
	# 	output5 =  client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": "You are a useful AI Assistant."},
	# 	{"role": "user", "content": prompt}, 
    # ])
		
	# except Exception as e:
	# 	output5 = str(e)
	# 	print(output5)
	# 	flag5 =True


	# pipe0 = pipeline( 
	# 	"text-generation", 
	# 	model=model0, 
	# 	tokenizer=tokenizer0, 
	# ) 

	# pipe1 = pipeline( 
	# 	"text-generation", 
	# 	model=model1, 
	# 	tokenizer=tokenizer1, 
	# ) 
	# pipe2 = pipeline( 
	# 	"text-generation", 
	# 	model=model2, 
	# 	tokenizer=tokenizer2, 
	# ) 

	# pipe3 = pipeline( 
	# 	"text-generation", 
	# 	model=model3, 
	# 	tokenizer=tokenizer3, 
	# ) 
	
	# pipe3_4 = pipeline( 
	# 	"text-generation", 
	# 	model=model3_4, 
	# 	tokenizer=tokenizer3_4, 
	# ) 
	
	pipe3_Q = pipeline( 
		"text-generation", 
		model=model3_Q, 
		tokenizer=tokenizer3_Q, 
	) 

	# pipe4 = pipeline( 
	# 	"text-generation", 
	# 	model=model4, 
	# 	tokenizer=tokenizer4, 
	# ) 

	generation_args = { 
		"max_new_tokens": 2048, 
		"return_full_text": False, 
		"temperature": 0.0, 
		"do_sample": False, 
	} 

	# output0 = pipe0(messages, **generation_args) 
	# output1 = pipe1(messages, **generation_args) 
	# output2 = pipe2(messages, **generation_args)
	# output3 = pipe3(messages, **generation_args)
	# output3_4 = pipe3_4(messages, **generation_args)
	output3_Q = pipe3_Q(messages, **generation_args)
	# output4 = pipe4(messages, **generation_args)
	

	# response = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": "You are a useful AI Assistant."},
    #     {"role": "user", "content": prompt},

    # ])

	# result_df.at[i, 'dialog'] = dialog  
	# result_df.at[i, 'GT_Summary'] = summary  
	# result_df.at[i, 'Model_1'] = output1[0]['generated_text']
	# result_df.at[i, 'Model_2 (Early Stop)'] = output2[0]['generated_text']
	# result_df.at[i, 'Model_3 (Only Good)'] = output3[0]['generated_text']
	# result_df.at[i, 'Phi-4'] = output3_4[0]['generated_text']
	result_df.at[i, 'Qwen2.5 (Base) '] = output3_Q[0]['generated_text']
	# result_df.at[i, 'Model_4 (With Corrections)'] = output4[0]['generated_text']
	# result_df.at[i, 'Base Model (Phi-3.5)'] = f"<BEGIN SUMMARY>\n\n{output0[0]['generated_text']}\n\n<END SUMMARY>"
	# result_df.at[i, 'GPT-4o'] = f"<BEGIN SUMMARY>\n\n{output5.choices[0].message.content}\n\n<END SUMMARY>" if not flag5 else f"<BEGIN SUMMARY>\n\n{output5}\n\n<END SUMMARY>"
# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/results_different_summaries_new.csv', index=False, encoding='utf-8')  # Replace with the actual file path