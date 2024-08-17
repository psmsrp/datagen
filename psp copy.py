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


text= """ William Shakespeare (c. 23[a] April 1564 – 23 April 1616)[b] was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist.[4][5][6] He is often called England's national poet and the "Bard of Avon" (or simply "the Bard"). His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright.[7] Shakespeare remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted.

Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592, he began a successful career in London as an actor, writer, and part-owner ("sharer") of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne. At age 49 (around 1613), he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories[8] as to whether the works attributed to him were written by others.[9][10][11]

Shakespeare produced most of his known works between 1589 and 1613.[12][13] His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres. He then wrote mainly tragedies until 1608, among them Hamlet, Othello, King Lear and Macbeth, all considered to be among the finest works in the English language.[4][5][6] In the last phase of his life, he wrote tragicomedies (also known as romances) such as The Winter's Tale and The Tempest, and collaborated with other playwrights.

Many of Shakespeare's plays were published in editions of varying quality and accuracy during his lifetime. However, in 1623, John Heminges and Henry Condell, two fellow actors and friends of Shakespeare's, published a more definitive text known as the First Folio, a posthumous collected edition of Shakespeare's dramatic works that includes 36 of his plays. Its Preface was a prescient poem by Ben Jonson, a former rival of Shakespeare, that hailed Shakespeare with the now famous epithet: "not of an age, but for all time".[14] """


response1= """William Shakespeare, an English playwright, poet, and actor, is regarded as the greatest writer in the English language and produced numerous influential plays, sonnets, and poems. Born in Stratford-upon-Avon, he had a successful career in London with the Lord Chamberlain's Men before retiring back to his hometown. His works were compiled posthumously in the First Folio by colleagues John Heminges and Henry Condell, solidifying his enduring legacy in literature."""

response = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You are an Expert in providing a detailed summary from any converstaion /  context . The summary should be in less than 3 sentences and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. Please ensure you give higher weightage to the recent conversations than the older conversations during the summary generation. \n I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in less than 3 sentences. If you do not have sufficient information to provide a summary then return \"None\". Do not provide an explanation. \n Guidelines: \n 1. Please do not use user names in the summary generated. \n 2. Please do not include sensitive information in the summary and keep it at a high level. \n 3. **Always strive for neutral and inclusive language.** \n 4. **Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": response1},
        {"role": "user", "content": "You missed out the point about his wife, try again"},
        {"role": "user", "content": text},

    ]
)

print("response: ", response)
print("choices: ",response.choices)
print("mssgs: ",response.choices[0].message.content)
print("content: ",response.choices[0].message.content)
