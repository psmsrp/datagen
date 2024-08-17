from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    # api_version="2023-03-15-preview",
    api_version="2024-02-15-preview",
    azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

rv= """ Enter System message here """

text= """ Enter text here """

response = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv},
        {"role": "user", "content": text},
    ]
)

print("Response: ",response.choices[0].message.content)
