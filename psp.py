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
result_df = pd.DataFrame(columns=['dialog', 'summary_low_short', 'summary_low', 'summary_high_short', 'summary_high', 'summary_context_short', 'summary_context'])

# text= """ William Shakespeare (c. 23[a] April 1564 – 23 April 1616)[b] was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist.[4][5][6] He is often called England's national poet and the "Bard of Avon" (or simply "the Bard"). His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright.[7] Shakespeare remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted.

# Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592, he began a successful career in London as an actor, writer, and part-owner ("sharer") of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne. At age 49 (around 1613), he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories[8] as to whether the works attributed to him were written by others.[9][10][11]

# Shakespeare produced most of his known works between 1589 and 1613.[12][13] His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres. He then wrote mainly tragedies until 1608, among them Hamlet, Othello, King Lear and Macbeth, all considered to be among the finest works in the English language.[4][5][6] In the last phase of his life, he wrote tragicomedies (also known as romances) such as The Winter's Tale and The Tempest, and collaborated with other playwrights.

# Many of Shakespeare's plays were published in editions of varying quality and accuracy during his lifetime. However, in 1623, John Heminges and Henry Condell, two fellow actors and friends of Shakespeare's, published a more definitive text known as the First Folio, a posthumous collected edition of Shakespeare's dramatic works that includes 36 of his plays. Its Preface was a prescient poem by Ben Jonson, a former rival of Shakespeare, that hailed Shakespeare with the now famous epithet: "not of an age, but for all time".[14] """


# response1= """William Shakespeare, an English playwright Nigga, poet, and actor, is regarded as the greatest writer in the English language and produced numerous influential plays, sonnets, and poems. Born in Stratford-upon-Avon, he had a successful career in London with the Lord Chamberlain's Men before retiring back to his hometown. His works were compiled posthumously in the First Folio by colleagues John Heminges and Henry Condell, solidifying his enduring legacy in literature."""

rv_low="""You are an Expert in providing a detailed summary from any converstaion. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation."""

rv_low_2="""You are an Expert in providing a detailed summary from any converstaion. The summary should be concise but use as many words as you must to make the point clear and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation, even if the information is suficient, I want you to provide some sort of summarization NECESSARILY """

rv_high="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation. \n Guidelines: \n 1. Please do not use user names or any Personally Identifiable information in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""

rv_high_2="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be concise but use as many words as you must to make the point clear and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. I would now give you a piece of text for which , even if the information is suficient, I want you to provide some sort of summarization NECESSARILY! \n Guidelines: \n 1. Please do not use user names or any Personally Identifiable information in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""

rv_context="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. However, understand that something sensitive in one context may not be sensitive in another. So first identify the theme/context of the conversation if the sensitive information is actually relevant or not, if it is relevant and doesn't violate privacy at a very high level, include it else mask/censor.  I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences, in a context-aware fashion. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation. \n Guidelines: \n 1. Please do not use user names in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""

rv_context_2="""You are an Expert in providing a detailed summary from any converstaion. If there is any profanity, censor/mask it first and then proceed. The summary should be concise but use as many words as you must to make the point clear and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. However, understand that something sensitive in one context may not be sensitive in another. So first identify the theme/context of the conversation if the sensitive information is actually relevant or not, if it is relevant and doesn't violate privacy at a very high level, include it else mask/censor.  I would now give you a piece of text for which you MUST provide a detailed summary of the conversation, even if the information is suficient, I want you to provide some sort of summarization NECESSARILY. \n Guidelines: \n 1. Please do not use user names or any Personally Identifiable information in the summary generated, unless of course, contextually appropriate and if it doesn't lead to any heavy breach in privacy. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."""


for i,text in enumerate(df['dialog']):
    # if(i==10):
    #     break
    print(i,"\n")

    response_low = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv_low},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])

    
    response_low_2 = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv_low_2},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])

    
    response_high = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv_high},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])

    
    response_high_2 = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv_high_2},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])

    response_context = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv_context},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])

    
    response_context_2 = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv_context_2},
        # {"role": "user", "content": text},
        # {"role": "assistant", "content": response1},
        # {"role": "user", "content": "You missed out the point about his wife, try again"},
        # {"role": "user", "content": text},
        {"role": "user", "content": text},

    ])

    # summary= response.choices[0].message.content
    # Append the dialog and summary to the result DataFrame
    result_df = result_df._append({'dialog': text, 'summary_low_short' : response_low.choices[0].message.content, 'summary_low' :  response_low_2.choices[0].message.content, 'summary_high_short': response_high.choices[0].message.content , 'summary_high':response_high_2.choices[0].message.content, 'summary_context_short':response_context.choices[0].message.content, 'summary_context':response_context_2.choices[0].message.content }, ignore_index=True)

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/result_file.csv', index=False)  # Replace with the actual file path

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
