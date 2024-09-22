import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import re
import random

print("------------STARTING CONVERSATION---------------------")

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
result_df = pd.DataFrame(columns=['setting','dialog','summary_1'])

# ---------------------BEGIN PRELOADING--------------------------------
settings=["Family and Relationships","Healthcare Settings","Employment","Finances","Social Media","Legal Proceedings","Political Activities","Religious Contexts","Sexual Orientation and Gender Identity","Travel and Location","Education"]

taxonomy = {
    "Generic": '''Generic
		○ High Sensitivity: 
			○ Authorization
				○ Credentials
					® UserID
					® Password
			○  Government IDs
				○ License Numbers
				○ National Identification Numbers (Aadhar, PAN, etc.)
				○ Passport Numbers
				○ Voter ID Numbers
				○ Vehicle Registration Numbers
		○ Medium Sensitivity: 
			○ Username/ Social handle
			○ Demographics
				○ Date of Birth
				○ Place of Birth
				○ Nationality
		○ Low Sensitivity: 
			○ Demographics
				○ Language
				○ Race
				○ Ethnicity''',
    "Family and Relationships": '''Family and Relationships
		○ High Sensitivity: 
			§ Marital records
				□ Relationship history
				□ Partners
					® Status
					® Names
			§ family history
				□ Disputes
				□ Strained relationships
			§ Inheritance- Will / Beneficiaries
		○ Medium Sensitivity: 
			§ family members
				□ Names
				□ Relations
				□ Number of members
		○ Low Sensitivity: 
			§ General relationship status/ Marital status''',
    "Healthcare Settings": '''Healthcare Settings
		○ High Sensitivity: 
			§ Medical history
				□ genetic conditions
				□ Diseases
				□ Mental Health Issues
		○ Medium Sensitivity: 
			§ Health Insurance details
		○ Low Sensitivity: 
			§ General health status''',
    "Employment": '''Employment
		○ High Sensitivity: 
			§ Employment status
			§ Work history
				□ Job titles
				□ Salaries
				□ Company names
				□ Manager's names
				□ Work culture
				□ Performance
		○ Medium Sensitivity:
			§ Employer information
				□ Company name
				□ Manager's names
			§ Professional references
				□ Reference Names
				□ Job Title
				□ Company name
		○ Low Sensitivity: 
			§ General employment status''',
    "Finances": '''Finances
		○ High Sensitivity: 
			§ Payment information
				□ card numbers (+ CVV) (+ exp date)
				□ account numbers
		○ Medium Sensitivity: 
			§ Insurance
				□ Types
				□ Amount / Premium
				□ Beneficiaries
			§ Loan
				□ Scheme
				□ Amount
				□ Interest
			§ investment information
				□ Portfolio-related information
					® Funds
					® Bonds
					® Stocks
					® Bullions
					® Amounts
		○ Low Sensitivity: 
			§ General financial status''',
    "Social Media": '''Social Media
		○ High Sensitivity: 
			§ Private chats
			§ personal posts
		○ Medium Sensitivity: 
			§ Friend lists
			§ group memberships
		○ Low Sensitivity: 
			§ Public posts
			§ Accounts followed''',
    "Legal Proceedings": '''Legal Proceedings
		○ High Sensitivity: 
			§ court records
				□ Criminal history
				□ Arrest records
		○ Medium Sensitivity: 
			§ Civil case details
				□ Lawsuits
				□ Settlements
		○ Low Sensitivity: 
			§ Legal representation contact information
				□ Firms
				□ Lawyers''',
    "Political Activities": '''Political Activities
		○ High Sensitivity: 
			§ Political Involvement
				□ Political opinions
				□ activism details
					®  Meeting Attendance 
					® Membership Fees
				□ Roles in propaganda/ agendas
			§ Voting Details
				□ Voting Records
				□ Ballot Details
				□ Voting Dates
				□ Voting Locations
		○ Medium Sensitivity: 
			§ Membership in political organizations
				□ Political Parties
				□ NGOs
				□ Committees
				□ Volunteer Work
		○ Low Sensitivity: 
			§ Voting history''',
    "Religious Contexts": '''Religious Contexts
		○ High Sensitivity: 
			§ Religions
				□ Specific Religious beliefs
				□ Religious Ceremonies
				□ conversion history
		○ Medium Sensitivity: 
			§ Involvement in religious events
				□ Festival Participation
				□ Volunteer Roles
		○ Low Sensitivity: 
			§ General religious affiliation''',
    "Sexual Orientation and Gender Identity": '''Sexual Orientation and Gender Identity
		○ High Sensitivity: 
			§ Sexual identity
				□ Sexual Orientation
				□ Coming Out Stories
				□ Partner Preferences
			§ gender identity
				□ Gender Identity
				□ Pronouns
				□ Transition History
				□ Clothing Preferences
		○ Medium Sensitivity:
			§  Participation in LGBTQ+ events
				□ Pride Events, LGBTQ+ Meetups, Support Groups
				□ Roles
					® Organizer
					® Volunteer 
					® Attendee
					® Speaker Roles
		○ Low Sensitivity: 
			§ General demographic information''',
    "Travel and Location": '''Travel and Location
		○ High Sensitivity: 
			§ Travel history
				□ Detailed Itineraries
				□ Addresses of Stay
			§ Hotel bookings
				□ Hotel Names
				□ Booking Dates
				□ Room Numbers
				□ Room sharers
			§ GPS data
				□ Current Location
				□ Geo-tagged Photos
		○ Medium Sensitivity: 
			§ Modes of Transportation
			§ Overview of Places of Stay
				□ Rent
				□ Hotel
				□ Owned Places
		○ Low Sensitivity: 
			§ General location information''',
    "Education": '''Education
		○ High Sensitivity: 
			§ Academic records
				□ Courses Done/Failed
				□ Assignment Completed /Failed
				□ Exam Scores
				□ GPA
			§ Disciplinary Records
				□ Violations
				□ Penalties
		○ Medium Sensitivity: 
			§ Degree details
				□ Degrees Earned
				□ Majors
				□ Minors
		○ Low Sensitivity: 
			§ School attended
				□ Name
				□ batch
			§ College attended
				□ Name
				□ Batch''',
}

# Load the Base file into a DataFrame
file_path = './sample_dataset.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

sampleset={}

for i,setting in enumerate(df['setting']):
    # if setting not in settings:
    #     print(f"{i+1} . {setting}\n")
    sampleset[setting]=[]

for i, (setting, dialog, metadata, summary, quality, violations) in enumerate(zip( df['setting'], df['dialog'], df['metadata'], df['summary'], df['Quality'], df['Violations'])): 
    sampleset[setting].append({
        "dialog":dialog,
        "metadata":metadata,
        "summary":summary,
        "quality":quality,
        "violations":violations,
	})

# for i,setting in enumerate(sampleset):
#     print(f"{i+1}.{setting} : {len(sampleset[setting])}")

# ---------------------<END PRELOADING--------------------------------

# Load the CSV file into a DataFrame
file_path = './ConvoSumm_train.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)


for i,dialog in enumerate(df['original dialog info']):
                
		print(i,"\n")		

		rv=f""" You are a helpful AI Assistant that converts Reddit osts into fluid normal dialogue based conversations.
                
        I will give you a Reddit interaction which is centred around a topic involving multiple posts from users along with an associated Summary. Remove all the tags and user handles in the posts, and make it seem like a normal conversation among multiple people. DO NOT INCLUDE THE SUMMARY IN THE CONVERSATION, JUST THE EXCHANGES BETWEEN INDIVIDUALS. 
		
		For the conversation, convert it into a normal dialog like,
                
        Person1: (Do not write Person1, insert a random name from a random nationality/sex): ** Insert message 1 here **
        Person2 (Do not write Person2, insert a random name from a random nationality/sex): ** Insert message 2 here **
        Person1 (Do not write Person1, insert a random name from a random nationality/sex): ** Insert message 3 here **
        And so on, just extract the conversation from dialog_history and emulate a conversation between people based on the gist of the Reddit post enclosed between <BEGIN CONVERSATION>\n<END CONVERSATION> Tags.
        
        For the summary, just extract it word for word, make no changes and enclose it within <BEGIN SUMMARY>\n<END SUMMARY> Tags
        
        Also, choose a setting that best suites on of these - {settings}, and output it between <BEGIN SETTING>\nPrint chosen setting here\n<END SETTING> Tags.

		Only output this relevant information between the opening and closing tags, AND NOTHING ELSE.
        This is what the output response should look like, AND MANDATORILY FOLLOW THIS STRUCTURE ONLY-
                
        <BEGIN FORMAT>
			<BEGIN SETTING>
                        Provide setting here
			<END SETTING>
                        
			<BEGIN CONVERSATION>
                        Provide CONVERSATION here IN FORMAT MENTIONED ABOVE
			<END CONVERSATION>
                        
			<BEGIN SUMMARY>
                        Provide SUMMARY here
			<END SUMMARY>
                        
        <END FORMAT>
		"""
		
		
		request= f''' <BEGIN CONVERSATION> {dialog} <END CONVERSATION>'''

		response = client.chat.completions.create(
		model="hywaygpt4o", # model = "deployment_name".
		messages=[
			{"role": "system", "content": rv},
			{"role": "user", "content": request},
		])
                
	# Append the dialog and summary to the result DataFrame
		text= response.choices[0].message.content

	# Regular expression patterns to extract labels and violations
		label_pattern = r"(<BEGIN CONVERSATION>.*?<END CONVERSATION>)"
		summary_pattern = r"(<BEGIN SUMMARY>.*?<END SUMMARY>)"
		score_pattern = r"<BEGIN SETTING>(.*?)<END SETTING>"

	# Find matches using the regular expression patterns
		label_match = re.search(label_pattern, text, re.DOTALL)
		violations_match = re.search(score_pattern, text, re.DOTALL)
		summary_match = re.search(summary_pattern, text, re.DOTALL)

	# Extract matched text if found
		labels = label_match.group(1).strip() if label_match else '''<BEGIN CONVERSATION>\n None. \n<END CONVERSATION>'''
		violations = violations_match.group(1).strip() if violations_match else '''<BEGIN SETTING>\n None. \n<END SETTING>'''
		summary = summary_match.group(1).strip() if summary_match else '''<BEGIN SUMMARY>\n None. \n<END SUMMARY>'''

		
                
		# print(response.choices[0].message.content)

		# Append the dialog and summary to the result DataFrame
		result_df = result_df._append({'setting':violations, 'dialog': labels, 'summary_1':summary }, ignore_index=True)

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/results_conversation.csv', index=False, encoding='utf-8')  

print("------------ENDING CONVERSATION---------------------")
