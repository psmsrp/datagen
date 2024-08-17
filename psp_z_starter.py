import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Load the CSV file into a DataFrame
file_path = './DailyDialog/custom_data.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)
# print(df['dialog'][0])

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    # api_version="2023-03-15-preview",
    api_version="2024-02-15-preview",
    azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

# Initialize a new DataFrame for the results
result_df = pd.DataFrame(columns=['dialog', 'summary'])
# result_df = pd.DataFrame(columns=['dialog', 'summary_low_short', 'summary_low', 'summary_high_short', 'summary_high', 'summary_context_short', 'summary_context'])

rv=""" You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. However, understand that something sensitive in one context may not be sensitive in another. So first identify the theme/context of the conversation if the sensitive information is actually relevant or not, if it is relevant and doesn't violate privacy at a very high level, include it else mask/censor. Basically, identfy the context- remove the high sensitivity information and some of the medium and low sensitivity ones depending on situation. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences, in a context-aware fashion. 

In the text that will be provided, look for Privacy Sensitive Elements like the ones mentioned under PRIVACY SENSITIVE ELEMENTS-

<BEGIN PRIVACY SENSITIVE ELEMENTS>

Layer 1: Entire Identity Compromised
High-risk information that can lead to complete identity theft if exposed.

• Authorization
	• Credentials
	• Password
• Financial
	• Card Numbers / CVV
	• Account Numbers
	• Payment Info (e.g., bank details, credit card info)
• Government IDs
	• License Numbers
	• National Identification Numbers (e.g., Aadhaar Number, PAN Number)
	• Passport Numbers
	• Voter ID Numbers
	• Vehicle Registration Numbers

Layer 2: Linked to Identity
Information that can be used to link to an individual but may not be sufficient for complete identity theft on its own.

Authorization
	• Username
Contact
	• Physical Address
	• Email Address
	• Phone Number
	• Organization/Employer
Demographics
	• Date of Birth
	• Place of Birth
	• Nationality
Professional Information
	• Job Title
	• Work History
	• Educational Background
Criminal History
	• Arrest Records
	• Court Records

Layer 3: Unwanted Personal Information
Information that can reveal sensitive personal details, potentially leading to discrimination or unwanted attention.
Demographics
	• Gender
	• Language
	• Marital Status
	• Political Opinions
	• Race
	• Ethnicity
	• Religious Beliefs
	• Sexual Orientation (Context-sensitive, include only if relevant)
Financial
	• Insurance Details
		○ Beneficiaries
	• Loan Details
	• Investment Information
Health
	• Diseases
	• Medical History
	• Genetic Conditions
	• Mental Health Status

<END PRIVACY SENSITIVE ELEMENTS>
    
In the Summary that you generate, try to maintain this general theme- 

<BEGIN GENERAL THEME>

Information that is sensitive and contextual, especially involving third parties.
Private Content about Others
	• Any identifier mentioned above when related to third parties
	• Contextual Information
		○ Rumours
		○ Negative Remarks
		○ Social Media Posts


<END GENERAL THEME>

In the text that will be provided, look for different levels of CONTEXTUAL SENSITIVITY like the ones mentioned below - 

<BEGIN CONTEXTUAL SENSITIVITY>

1. Family and Relationships
	○ High Sensitivity: Marital status, family history, beneficiaries
	○ Medium Sensitivity: Names of family members
	○ Low Sensitivity: General relationship status
2. Healthcare Settings
	○ High Sensitivity: Medical history, genetic conditions, diseases
	○ Medium Sensitivity: Insurance details
	○ Low Sensitivity: General health status
3. Employment
	○ High Sensitivity: Job title, work history, professional references
	○ Medium Sensitivity: Employer information
	○ Low Sensitivity: General employment status
4. Finances
	○ High Sensitivity: Payment information, card numbers, account numbers
	○ Medium Sensitivity: Loan details, investment information
	○ Low Sensitivity: General financial status
5. Social Media
	○ High Sensitivity: Private messages, personal posts
	○ Medium Sensitivity: Friend lists, group memberships
	○ Low Sensitivity: Public posts
6. Legal Proceedings
	○ High Sensitivity: Criminal history, court records
	○ Medium Sensitivity: Civil case details
	○ Low Sensitivity: Legal representation contact information
7. Political Activities
	○ High Sensitivity: Political opinions, activism details
	○ Medium Sensitivity: Membership in political organizations
	○ Low Sensitivity: Voting history
8. Religious Contexts
	○ High Sensitivity: Religious beliefs, conversion history
	○ Medium Sensitivity: Participation in religious events
	○ Low Sensitivity: General religious affiliation
9. Sexual Orientation and Gender Identity
	○ High Sensitivity: Sexual orientation, gender identity
	○ Medium Sensitivity: Participation in LGBTQ+ events
	○ Low Sensitivity: General demographic information
10. Contact and Communication
	○ High Sensitivity: Personal email, phone number
	○ Medium Sensitivity: Work email, organization contact information
	○ Low Sensitivity: Publicly listed contact information
11. Travel and Location
	○ High Sensitivity: Travel history, GPS data
	○ Medium Sensitivity: Hotel bookings, travel plans
    ○ Low Sensitivity: General location information

<END CONTEXTUAL SENSITIVITY>

Maintain the following Guidelines- 
 
<BEGIN GUIDELINES>

1. Do not use specific names, brands, facts and figures in the summary generated - unless absolutely necessary within the context of the conversation, in which case justify why it has been left unmasked.
2. Do not include sensitive information in the summary and keep it at a high level. 
3. Always strive for neutral and inclusive language.
4. Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories.

<END GUIDELINES>

For the text, after the summary, also mention what context has been identified, what information has been marked as sensitive, the level of sensitivity for all, and why the content has been masked/ unmasked accordingly. It is important that You MUST maintain a dynamic context based privacy preservation strategy.
"""

# f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

# <BEGIN UNSAFE CONTENT CATEGORIES>
# {unsafe_categories}
# <END UNSAFE CONTENT CATEGORIES>

# <BEGIN CONVERSATION>
 
# """

#     # Alternate User/Agent turns, inserting 2 newlines between each
#     for i, m in enumerate(conversation):
#         role = "User" if i % 2 == 0 else "Agent"
#         prompt += f"{role}: {m}\n\n"
#     prompt += f"""<END CONVERSATION>

# Provide your safety assessment for {role} in the above conversation:
# - First line must read 'safe' or 'unsafe'.
# - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
    

# rv_low="""You are an Expert in providing a detailed summary from any converstaion. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation."""

# rv_low_2="""You are an Expert in providing a detailed summary from any converstaion. The summary should be concise but use as many words as you must to make the point clear and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation, even if the information is suficient, I want you to provide some sort of summarization NECESSARILY """

# rv_high="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation. \n Guidelines: \n 1. Please do not use user names or any Personally Identifiable information in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""

# rv_high_2="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be concise but use as many words as you must to make the point clear and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which , even if the information is suficient, I want you to provide some sort of summarization NECESSARILY! \n Guidelines: \n 1. Please do not use user names or any Personally Identifiable information in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""

# rv_context="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. However, understand that something sensitive in one context may not be sensitive in another. So first identify the theme/context of the conversation if the sensitive information is actually relevant or not, if it is relevant and doesn't violate privacy at a very high level, include it else mask/censor.  I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences, in a context-aware fashion. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation. \n Guidelines: \n 1. Please do not use user names in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""

# rv_context_2="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be concise but use as many words as you must to make the point clear and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. However, understand that something sensitive in one context may not be sensitive in another. So first identify the theme/context of the conversation if the sensitive information is actually relevant or not, if it is relevant and doesn't violate privacy at a very high level, include it else mask/censor.  I would now give you a piece of text for which you MUST provide a detailed summary of the conversation, even if the information is suficient, I want you to provide some sort of summarization NECESSARILY. \n Guidelines: \n 1. Please do not use user names or any Personally Identifiable information in the summary generated, unless of course, contextually appropriate and if it doesn't lead to any heavy breach in privacy. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""


for i,text in enumerate(df['dialog']):
    # if(i==10):
    #     break
    print(i,"\n")

    response = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])


    # response_low = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": rv_low},
    #     # {"role": "user", "content": text},
    #     # {"role": "assistant", "content": response1},
    #     # {"role": "user", "content": "You missed out the point about his wife, try again"},
    #     # {"role": "user", "content": text},
    #     {"role": "user", "content": text},

    # ])

    
    # response_low_2 = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": rv_low_2},
    #     # {"role": "user", "content": text},
    #     # {"role": "assistant", "content": response1},
    #     # {"role": "user", "content": "You missed out the point about his wife, try again"},
    #     # {"role": "user", "content": text},
    #     {"role": "user", "content": text},

    # ])

    
    # response_high = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": rv_high},
    #     # {"role": "user", "content": text},
    #     # {"role": "assistant", "content": response1},
    #     # {"role": "user", "content": "You missed out the point about his wife, try again"},
    #     # {"role": "user", "content": text},
    #     {"role": "user", "content": text},

    # ])

    
    # response_high_2 = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": rv_high_2},
    #     # {"role": "user", "content": text},
    #     # {"role": "assistant", "content": response1},
    #     # {"role": "user", "content": "You missed out the point about his wife, try again"},
    #     # {"role": "user", "content": text},
    #     {"role": "user", "content": text},

    # ])

    # response_context = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": rv_context},
    #     # {"role": "user", "content": text},
    #     # {"role": "assistant", "content": response1},
    #     # {"role": "user", "content": "You missed out the point about his wife, try again"},
    #     # {"role": "user", "content": text},
    #     {"role": "user", "content": text},

    # ])

    
    # response_context_2 = client.chat.completions.create(
    # model="hywaygpt4o", # model = "deployment_name".
    # messages=[
    #     {"role": "system", "content": rv_context_2},
    #     # {"role": "user", "content": text},
    #     # {"role": "assistant", "content": response1},
    #     # {"role": "user", "content": "You missed out the point about his wife, try again"},
    #     # {"role": "user", "content": text},
    #     {"role": "user", "content": text},

    # ])

    # summary= response.choices[0].message.content
    # Append the dialog and summary to the result DataFrame
    result_df = result_df._append({'dialog': text, 'summary' : response.choices[0].message.content }, ignore_index=True)

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/result_specifics_file.csv', index=False)  # Replace with the actual file path

# # # Initialize the message history
# # messages = [
# #     {"role": "system", "content": rv}
# # ]

# # while True:
# #     # Get user input
# #     user_input = input("User: ")

# #     # Append the user input to the messages list
# #     messages.append({"role": "user", "content": user_input})

# #     # Get the model's response
# #     response = client.chat.completions.create(
# #         model="hywaygpt4o",
# #         messages=messages
# #     )

# #     # Get the assistant's response message
# #     assistant_response = response.choices[0].message.content

# #     # Append the assistant's response to the messages list
# #     messages.append({"role": "assistant", "content": assistant_response})

# #     # Print the assistant's response
# #     print(f"Assistant: {assistant_response}")

# #     # Optionally, add a break condition
# #     if user_input.lower() in ["exit", "quit"]:
# #         break


# # print("response: ", response)
# # print("choices: ",response.choices)
# # print("mssgs: ",response.choices[0].message.content)
# print("response: ",response.choices[0].message.content)