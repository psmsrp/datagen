import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import random
import keyboard



# Load the CSV file into a DataFrame
file_path = './greg.csv'  # Replace with the actual file path
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

curr=[]
Words={}
count=0
# Iterate over the summary and metadata columns
for i, (word, meaning) in enumerate(zip(df['word'], df['meaning'])): 
	if(i%30==0):
		if(len(curr)):
			Words[count]=curr
			curr=[]
			count+=1
				
	curr.append({"word":word, "meaning":meaning})

curr.append({"word":word, "meaning":meaning})

     

def get_random_word():
    key = random.choice(list(Words.keys()))
    return random.choice(Words[key])

def gpt_call(prompt):
      
    response = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You're a helpful AI Assitant. Given a word,  State the word, its meaning and pronunciation as well. For pronunciation, along with phonetics, write the pronunciation in simple English as well, like SEE-SAW-YEAH format. Provide a few easy to understand example sentences and a few easy to remember ways to remember a given word and its meaning provided. Also mention 5-10 Synonyms or similar words along with an example of its usage and pronunciation. Do the same for 5-10 such antonyms as well."},
        {"role": "user", "content": prompt},

    ])
    
    return response.choices[0].message.content.strip()

def main_loop():
    while True:
        word_data = get_random_word()
        print(f"Word: {word_data['word']}")
        flag=False

        while True:
            user_input = str(input("Press 'space' to show meaning, 'enter' to call GPT, or 'q' to quit: ").strip().lower())
            if user_input == '':
                print(f"Meaning: {word_data['meaning']}")
                break
            elif user_input == '\'':
                print("Calling GPT...")
                prompt = f"Word: '{word_data['word']}' \n\n Meaning: '{word_data['meaning']}"
                response = gpt_call(prompt)
                print(f"GPT Response: {response}")
                break
            elif user_input == 's':
                word=str(input("Enter word : ").strip().lower())
                print("Calling GPT...")
                prompt = f"Word: '{word}'"
                response = gpt_call(prompt)
                print(f"GPT Response: {response}")
                break
            elif user_input == 'q':
                 print("Exiting GPT...")
                 flag= True
                 break
        
        if flag:
            break


if __name__ == "__main__":
    main_loop()


	



            
    
