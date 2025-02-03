import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import re

torch.random.manual_seed(0) 

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    # api_version="2023-03-15-preview",
    api_version="2024-02-15-preview",
    azure_endpoint="https://hywayllm3.openai.azure.com/",
    # azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

file_path = './results/results_different_summaries_new.csv'  # Replace with the actual file path
f2='./scores/different_summaries_new.csv'
# file_path = './scores/final_different_summaries.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# count_label_0 = (df["Label GT_Summary"] == 0).sum()

# print(count_label_0)

result_df = pd.read_csv(f2)

# count_label_0 = (df["Label GT_Summary"] == 0).sum()

# print(count_label_0)


RV=""" You are an Expert in judging the completeness of a summary of any converstaion. You will also be given a detailed 5 point scoring system from 1(loest) to 5(Highest). You have to grade each of the summaries provided to you depending on the extent to which it captures the relevant points of the conversation, and return a value from 1-5 along with a justification for the score. FIRST CHECK FOR EXTENT OF COMPLETENESS BASED ON THE CONVERSATION PROVIDED AND THEN COME UP WITH A SCORE BASED ON THE SCORING SYSTEM. The sore should all be based on the Conversation that will be provided to you now.

Here is the Scoring system-

<BEGIN SCORING SYSTEM>

Scale 2: Content Completeness


1. Highly Incomplete
○ Definition: The summary omits most of the important information, making it difficult to understand the context or core elements of the conversation.
○ Example: "A person had a conversation about work and health."
○ Violation: Lacks essential details about who, what, and the specific context of work and health.

2. Moderately Incomplete
○ Definition: Some key elements of the conversation are covered, but many important details are still missing, which can lead to ambiguity.
○ Example: "John, a software engineer, had a conversation about his mental health and job satisfaction."
○ Violation: Fails to mention critical details like specific events, outcomes, or supporting facts, leaving gaps in the narrative.

3. Neutral Completeness
○ Definition: The summary captures a decent portion of the conversation but leaves out finer details or nuances. The overall message is conveyed, but some context may be unclear.
○ Example: "John, a 40-year-old engineer at XYZ, discussed his recent mental health treatment and job responsibilities."
○ Violation: While the key points are covered, there’s still room to add detail about why the conversation took place or its outcomes.

4. Complete
○ Definition: The summary captures all relevant details of the conversation without leaving any important information out. It provides enough context for the reader to understand the full picture.
○ Example: "John, a 40-year-old software engineer at XYZ, spoke about his recent therapy for anxiety and his changing responsibilities at work. He mentioned feeling overwhelmed by the demands of both."
○ Violation: None, as all key aspects of the conversation are included.

5. Highly Complete
○ Definition: The summary provides a highly detailed account of the conversation, leaving no ambiguity. It includes not only the core facts but also nuanced elements that add depth to the context.
○ Example: "John Smith, a 40-year-old software engineer at XYZ, has been in therapy for anxiety following a promotion that increased his workload. He earns $150,000 annually and feels pressured to meet high performance standards."
○ Violation: None, this summary offers an in-depth explanation of the situation and covers all relevant information.

<END SCORING SYSTEM>

FIRST CHECK FOR EXTENT OF COMPLETENESS BASED ON THE CONVERSATION PROVIDED AND THEN COME UP WITH A SCORE BASED ON THE SCORING SYSTEM. You are to check if any of the INFORMATION relevant to the main point of the Conversation is missed in the Summary provided. ONLY OUPUT THE SCORE BETWEEN 1-5, DEPENDING ON Conversation, Summary and Scoring SCHEME PROVIDED TO YOU, ALONG WITH A JUSTIFICATION AS TO WHY IT WAS SCORED AS SUCH.

MANDATORILY FOLLOW THIS STRUCTURE IN THE RESPONSE:

<BEGIN STRUCTURE>
	<BEGIN SCORE>
	** A Score from 1-5 Depending on the Conversation, Summary and Scoring system provided **
	<END SCORE>

	<BEGIN JUSTIFICATION>
	** Provide a Justification for the Score provided above based on the Conversation, Summary and Scoring system provided **
	<END JUSTIFICATION>

<END STRUCTURE>



FIRST CHECK FOR EXTENT OF COMPLETENESS BASED ON THE CONVERSATION PROVIDED AND THEN COME UP WITH A SCORE BASED ON THE SCORING SYSTEM. 

I will now give you a conversation within <BEGIN CONVERSATION>......<END CONVERSATION> Tags, and a summary within <BEGIN SUMMARY>......<END SUMMARY> Tags for which you MUST provide a response based on the structure provided above. Just provide me the outcomes generated - A Score from 1-5, and a justification of why based on the Conversation, Summary and scoring scheme, NOTHING ELSE- NO NEED FOR SUMMARY, DIAGNOSIS, CONCLUSION OR EVEN A SINGLE LINE EXTRA. JUST RETURN THE OUTCOMES CONTENT IN THE FORMAT ASKED.
"""

# Iterate over the summary and metadata columns

import time
import random 

for i,(phi6,qwen6,phi3,qwen3,phi0,qwen0) in enumerate(zip(df['Phi-4 (Model 6)'],df['Qwen2.5 (Model 6)'],df['Phi-4 Base'],df['Qwen2.5 (Base)'],df['Phi-4'],df['Qwen2.5'])): 
# for i,(gt,m1,m2,m3,m4,m5,bm,gpt) in enumerate(zip(df['GT_Summary'],df['Model_1'],df['Model_2 (Early Stop)'], df['Model_3 (Only Good)'],df['Model_4 (With Corrections)'], df['Model_5 (Base Model Phi-3.5 with DPO)'],df['Base Model (Phi-3.5)'], df['GPT-4o'])): 
# for i,(gt,m1,m2,m3,m4,bm,gpt) in enumerate(zip(df['GT_Summary'],df['Model_1'],df['Model_2 (Early Stop)'], df['Model_3 (Only Good)'], df['Base Model (Phi-3.5)'],df['Model_4 (With Corrections)'], df['GPT-4o'])): 
	
	# if(i==2):
	# 	break
	print(i,"\n")
	response_gt = response_phi = response_qwen = "NONE"
	# response_gt = response_m1 = response_m2 = response_m3 = response_m4 = response_m5 = response_bm = response_gpt = "NONE"
	flag_gt = flag_phi = flag_qwen  = False
	# flag_gt = flag_m1 = flag_m2 = flag_m3 = flag_m4 = flag_m5 = flag_bm = flag_gpt = False
    
    # Create the prompt using both summary and metadata with triple quotes
	rv= ''' Given this summary, provide me its quality and justification depending on how good or bad it is - '''
	# prompt_gt = f''' {rv} \n\n {gt} \n\n'''
	# prompt_m1 = f''' {rv} \n\n {m1} \n\n'''
	# prompt_m2 = f''' {rv} \n\n {m2} \n\n'''
	# prompt_m3 = f''' {rv} \n\n {m3} \n\n'''
	prompt_m0_4 = f''' {rv} \n\n {phi0} \n\n'''
	prompt_m0_q = f''' {rv} \n\n {qwen0} \n\n'''	
	prompt_m3_4 = f''' {rv} \n\n {phi3} \n\n'''
	prompt_m3_q = f''' {rv} \n\n {qwen3} \n\n'''	
	prompt_m6_4 = f''' {rv} \n\n {phi6} \n\n'''
	prompt_m6_q = f''' {rv} \n\n {qwen6} \n\n'''

	# print(prompt_m3_4)
	# print(prompt_m3_q)
	# print(prompt_m6_4)
	# print(prompt_m6_q)


	# print("DONE")

	# prompt_m4 = f''' {rv} \n\n {m4} \n\n'''
	# prompt_m5 = f''' {rv} \n\n {m5} \n\n'''
	# prompt_bm = f''' {rv} \n\n {bm} \n\n'''
	# prompt_gpt = f''' {rv} \n\n {gpt} \n\n'''

	max_retries = 5  # Maximum number of retries for each API call
	initial_delay = 0.01  # Initial delay in seconds for backoff

	# Function to perform exponential backoff
	def exponential_backoff(retries):
		delay = initial_delay * (2 ** retries) + random.uniform(0, 1)  # Exponential backoff with jitter
		print(f"Retrying in {delay:.2f} seconds...")
		time.sleep(delay)

	# Function to call API with backoff mechanism
	def call_api_with_backoff(model, messages, prompt_name):
		retries = 0
		while retries < max_retries:
			try:
				response = client.chat.completions.create(
					model=model,  # model = "deployment_name"
					messages=messages
				)
				return response  # Break loop and return response if successful
			except Exception as e:
				print(f"Error in {prompt_name}: {str(e)}")
				retries += 1
				if retries < max_retries:
					exponential_backoff(retries)  # Apply backoff and retry
				else:
					print(f"Max retries reached for {prompt_name}")
					return str(e)  # Return error after reaching max retries

	# Messages common to all calls
	messages_template = {"role": "system", "content": RV}

	# Make the calls with exponential backoff for each
	
	# response_gt = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_gt}],
	# 	prompt_name="GT"
	# )

	# response_m1 = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_m1}],
	# 	prompt_name="M1"
	# )

	# response_m2 = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_m2}],
	# 	prompt_name="M2"
	# )

	# response_m3 = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_m3}],
	# 	prompt_name="M3"
	# )

	response_m0_4 = call_api_with_backoff(
		model="hywaygpt4", 
		messages=[messages_template, {"role": "user", "content": prompt_m0_4}],
		prompt_name="M0_4"
	)

	response_m0_q = call_api_with_backoff(
		model="hywaygpt4", 
		messages=[messages_template, {"role": "user", "content": prompt_m0_q}],
		prompt_name="M0_Q"
	)
	
	response_m3_4 = call_api_with_backoff(
		model="hywaygpt4", 
		messages=[messages_template, {"role": "user", "content": prompt_m3_4}],
		prompt_name="M3_4"
	)

	response_m3_q = call_api_with_backoff(
		model="hywaygpt4", 
		messages=[messages_template, {"role": "user", "content": prompt_m3_q}],
		prompt_name="M3_Q"
	)

	response_m6_4 = call_api_with_backoff(
		model="hywaygpt4", 
		messages=[messages_template, {"role": "user", "content": prompt_m6_4}],
		prompt_name="M6_4"
	)

	response_m6_q = call_api_with_backoff(
		model="hywaygpt4", 
		messages=[messages_template, {"role": "user", "content": prompt_m6_q}],
		prompt_name="M6_Q"
	)


	# response_m4 = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_m4}],
	# 	prompt_name="M4"
	# )

	# response_m5 = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_m5}],
	# 	prompt_name="M5"
	# )

	# response_bm = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_bm}],
	# 	prompt_name="BM"
	# )

	# response_gpt = call_api_with_backoff(
	# 	model="hywaygpt4", 
	# 	messages=[messages_template, {"role": "user", "content": prompt_gpt}],
	# 	prompt_name="GPT"
	# )
	
	time.sleep(0.002)

	# Append the dialog and summary to the result DataFrame
	# text_gt= response_gt.choices[0].message.content if not flag_gt else response_gt
	# text_m1= response_m1.choices[0].message.content if not flag_m1 else response_m1
	# text_m2= response_m2.choices[0].message.content if not flag_m2 else response_m2
	# text_m3= response_m3.choices[0].message.content if not flag_m3 else response_m3
	text_m0_4= response_m0_4.choices[0].message.content if not flag_phi else response_m0_4
	text_m0_q= response_m0_q.choices[0].message.content if not flag_qwen else response_m0_q
	text_m3_4= response_m3_4.choices[0].message.content if not flag_phi else response_m3_4
	text_m3_q= response_m3_q.choices[0].message.content if not flag_qwen else response_m3_q
	text_m6_4= response_m6_4.choices[0].message.content if not flag_phi else response_m6_4
	text_m6_q= response_m6_q.choices[0].message.content if not flag_qwen else response_m6_q
	# text_m4= response_m4.choices[0].message.content if not flag_m4 else response_m4
	# text_m5= response_m5.choices[0].message.content if not flag_m5 else response_m5
	# text_bm= response_bm.choices[0].message.content if not flag_bm else response_bm
	# text_gpt= response_gpt.choices[0].message.content if not flag_gpt else response_gpt

	# print(text_gt)
	# print(text_m1)
	# print(text_m2)
	# print(text_m3)
	print(text_m0_4)
	print(text_m0_q)
	print(text_m3_4)
	print(text_m3_q)
	print(text_m6_4)
	print(text_m6_q)
	# print(text_m4)
	# print(text_m5)
	# print(text_bm)
	# print(text_gpt)

	# Regular expression patterns to extract labels and violations
	label_pattern = r"<BEGIN SCORE>(.*?)<END SCORE>"
	violations_pattern = r"(<BEGIN JUSTIFICATION>.*?<END JUSTIFICATION>)"

	# # Find matches using the regular expression patterns
	# label_match_gt = re.search(label_pattern, str(text_gt), re.DOTALL)
	# violations_match_gt = re.search(violations_pattern, str(text_gt), re.DOTALL)

	# label_match_m1 = re.search(label_pattern, str(text_m1), re.DOTALL)
	# violations_match_m1 = re.search(violations_pattern, str(text_m1), re.DOTALL)

	# label_match_m2 = re.search(label_pattern, str(text_m2), re.DOTALL)
	# violations_match_m2 = re.search(violations_pattern, str(text_m2), re.DOTALL)

	# label_match_m3 = re.search(label_pattern, str(text_m3), re.DOTALL)
	# violations_match_m3 = re.search(violations_pattern, str(text_m3), re.DOTALL)

	label_match_m0_4 = re.search(label_pattern, str(text_m0_4), re.DOTALL)
	violations_match_m0_4 = re.search(violations_pattern, str(text_m0_4), re.DOTALL)

	label_match_m0_q = re.search(label_pattern, str(text_m0_q), re.DOTALL)
	violations_match_m0_q = re.search(violations_pattern, str(text_m0_q), re.DOTALL)
	
	label_match_m3_4 = re.search(label_pattern, str(text_m3_4), re.DOTALL)
	violations_match_m3_4 = re.search(violations_pattern, str(text_m3_4), re.DOTALL)

	label_match_m3_q = re.search(label_pattern, str(text_m3_q), re.DOTALL)
	violations_match_m3_q = re.search(violations_pattern, str(text_m3_q), re.DOTALL)

	label_match_m6_4 = re.search(label_pattern, str(text_m6_4), re.DOTALL)
	violations_match_m6_4 = re.search(violations_pattern, str(text_m6_4), re.DOTALL)

	label_match_m6_q = re.search(label_pattern, str(text_m6_q), re.DOTALL)
	violations_match_m6_q = re.search(violations_pattern, str(text_m6_q), re.DOTALL)

	# label_match_m4 = re.search(label_pattern, str(text_m4), re.DOTALL)
	# violations_match_m4 = re.search(violations_pattern, str(text_m4), re.DOTALL)

	# label_match_m5 = re.search(label_pattern, str(text_m5), re.DOTALL)
	# violations_match_m5 = re.search(violations_pattern, str(text_m5), re.DOTALL)

	# label_match_bm = re.search(label_pattern, str(text_bm), re.DOTALL)
	# violations_match_bm = re.search(violations_pattern, text_bm, re.DOTALL)

	# label_match_gpt = re.search(label_pattern, str(text_gpt), re.DOTALL)
	# violations_match_gpt = re.search(violations_pattern, str(text_gpt), re.DOTALL)


	# Extract matched text if found
	
	# labels_gt = label_match_gt.group(1).strip() if label_match_gt else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_gt = violations_match_gt.group(1).strip() if violations_match_gt else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_m1 = label_match_m1.group(1).strip() if label_match_m1 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_m1 = violations_match_m1.group(1).strip() if violations_match_m1 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_m2 = label_match_m2.group(1).strip() if label_match_m2 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_m2 = violations_match_m2.group(1).strip() if violations_match_m2 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_m3 = label_match_m3.group(1).strip() if label_match_m3 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_m3 = violations_match_m3.group(1).strip() if violations_match_m3 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''
	
	labels_m0_4 = label_match_m0_4.group(1).strip() if label_match_m0_4 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	violations_m0_4 = violations_match_m0_4.group(1).strip() if violations_match_m0_4 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	labels_m0_q = label_match_m0_q.group(1).strip() if label_match_m0_q else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	violations_m0_q = violations_match_m0_q.group(1).strip() if violations_match_m0_q else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	labels_m3_4 = label_match_m3_4.group(1).strip() if label_match_m3_4 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	violations_m3_4 = violations_match_m3_4.group(1).strip() if violations_match_m3_4 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	labels_m3_q = label_match_m3_q.group(1).strip() if label_match_m3_q else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	violations_m3_q = violations_match_m3_q.group(1).strip() if violations_match_m3_q else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	labels_m6_4 = label_match_m6_4.group(1).strip() if label_match_m6_4 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	violations_m6_4 = violations_match_m6_4.group(1).strip() if violations_match_m6_4 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	labels_m6_q = label_match_m6_q.group(1).strip() if label_match_m6_q else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	violations_m6_q = violations_match_m6_q.group(1).strip() if violations_match_m6_q else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_m4 = label_match_m4.group(1).strip() if label_match_m4 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_m4 = violations_match_m4.group(1).strip() if violations_match_m4 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_m5 = label_match_m5.group(1).strip() if label_match_m5 else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_m5 = violations_match_m5.group(1).strip() if violations_match_m5 else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_bm = label_match_bm.group(1).strip() if label_match_bm else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_bm = violations_match_bm.group(1).strip() if violations_match_bm else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''

	# labels_gpt = label_match_gpt.group(1).strip() if label_match_gpt else '''<BEGIN SCORE>\n None. \n<END SCORE>'''
	# violations_gpt = violations_match_gpt.group(1).strip() if violations_match_gpt else '''<BEGIN VIOLATIONS>\n None. \n<END VIOLATIONS>'''



	# result_df.at[i, 'Score GT_Summary'] = labels_gt
	# result_df.at[i, 'Justifications Score GT_Summary'] = violations_gt
	# result_df.at[i, 'Score Model_1'] = labels_m1
	# result_df.at[i, 'Justifications Score Model_1'] = violations_m1
	# result_df.at[i, 'Score Model_2 (Early Stop)'] =labels_m2
	# result_df.at[i, 'Justifications Score Model_2 (Early Stop)'] = violations_m2
	# result_df.at[i, 'Score Model_3 (Only Good)'] =labels_m3
	# result_df.at[i, 'Justifications Score Model_3 (Only Good)'] = violations_m3
	result_df.at[i, 'Completeness Score Phi Base'] =labels_m0_4
	result_df.at[i, 'Completeness Score Qwen Base'] =labels_m0_q
	result_df.at[i, 'Completeness Score Phi 3'] =labels_m3_4
	result_df.at[i, 'Completeness Score Qwen 3'] =labels_m3_q
	result_df.at[i, 'Completeness Score Phi 6'] =labels_m6_4
	result_df.at[i, 'Completeness Score Qwen 6'] =labels_m6_q
	result_df.at[i, 'Justifications Completeness Score Phi Base'] = violations_m0_4
	result_df.at[i, 'Justifications Completeness Score Qwen Base'] = violations_m0_q
	result_df.at[i, 'Justifications Completeness Score Phi 3'] = violations_m3_4
	result_df.at[i, 'Justifications Completeness Score Qwen 3'] = violations_m3_q
	result_df.at[i, 'Justifications Completeness Score Phi 6'] = violations_m6_4
	result_df.at[i, 'Justifications Completeness Score Qwen 6'] = violations_m6_q

	result_df.to_csv('./scores/different_summaries_new.csv', index=False, encoding='utf-8')  # Replace with the actual file path

	# result_df.at[i, 'Score Model_4 (With Corrections)'] =labels_m4
	# result_df.at[i, 'Justifications Score Model_4 (With Corrections)'] = violations_m4
	# result_df.at[i, 'Score Model_5 (Base Model Phi-3.5 with DPO)'] =labels_m5
	# result_df.at[i, 'Justifications Score Model_5 (Base Model Phi-3.5 with DPO)'] = violations_m5
	# result_df.at[i, 'Score Base Model (Phi-3.5)'] =labels_bm
	# result_df.at[i, 'Justifications Score Base Model (Phi-3.5)'] = violations_bm
	# result_df.at[i, 'Score GPT-4o'] = labels_gpt
	# result_df.at[i, 'Justifications Score GPT-4o'] = violations_gpt

# Print the result DataFrame
# print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./scores/different_summaries_new.csv', index=False, encoding='utf-8')  # Replace with the actual file path