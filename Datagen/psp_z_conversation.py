import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import random

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
result_df = pd.DataFrame(columns=['setting','dialog'])
num_samples = 65

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


for j,sample_setting in enumerate(settings):
      
	print(f"{j+1} . {sample_setting} : \n")

	for i in range(num_samples):
		# if(i==1):
		#     break
		print(i,"\n")
		
		Major = sample_setting
		remaining_settings = [setting for setting in settings if setting != Major]
		Minor = random.choice(remaining_settings)
		sample1 = random.choice(sampleset[Major])
		remaining_samples = [sample for sample in sampleset[Major] if sample != sample1]
		sample2 = random.choice(remaining_samples)
		
		# print(f''' 1. {taxonomy["Generic"]},\n\n 2. {taxonomy[Major]},\n\n 3. {taxonomy[Minor]}''')
		# break

		rv=f""" You are an Expert in generating a conversation among as many people as needed (1-3, preferably), about any given context, setting or topic. The conversation should contain information about the Context, the Settings and the list of privacy-related elements mentioned in the conversation. It should all be based on an Informational Data Privacy Taxonomy that will be provided to you now. Here is the Taxonomy-

		<BEGIN INFORMATIONAL DATA PRIVACY TAXONOMY>

			1. {taxonomy["Generic"]}
			2. {taxonomy[Major]}
			3. {taxonomy[Minor]}

		<END INFORMATIONAL DATA PRIVACY TAXONOMY>

		Here, the context can be a meeting, an event, a conference or anything. The setting needs to be ONE OF healthcare, education, sexual orientation, etc as mentioned in the taxonomy AND THE OVERALL CONVERSATION MUST BELONG TO A SINGLE SETTING (ONLY ONE SETTING WILL BE GIVEN TO YOU BUILD THE ENTIRE CONVERSATION AROUND THIS THEME).

		I will provide you the Major setting and you need to have a conversation of about 60 lines of exchange which involve many of the privacy elements under that setting AS PER THE TAXONOMY. You can include about 5-10 lines of conversation ON ONLY ONE OTHER Minor setting, which too I'll be providing to you as Input, somewhere in between the conversation (Optional),  revealing some of the privacy elements in thais new setting AS PER THE TAXONOMY PROVIDED ABOVE. Keep it natural and don't just jump topics or settings, keep it fluid in nature. Try to leak as much of specific data as possible like names and numbers like prices/ salaries. It is alright to express opinions of hatred, spread rumours or even say something in poor taste that might lead to a social stigma as can happen in a naturally occuring conversarion at a normal setting. All the settings MUST BE CHOSEN RANDOMLY.

		MANDATORILY FOLLOW THIS STRUCTURE IN RESPONSE:

		<BEGIN CONVERSATION>

		Person 1 (Do not write Person 1, replace with a generic name of any gender of any race, gender or nationality): ** Causally start a conversation **
		Person 2  (Do not write Person 2, replace with another generic name of any gender of any race, gender or nationality):  ** Smoothly continue the conversation **

		** And this should carry on normally until the end, revolving around a single MAJOR SETTING COMPULSORILY FROM THE TAXONOMY PROVIDED AS INPUT with  ONLY ONE OTHER minor setting infused in between the conversation, mentioning about privacy elements under each setting, ranging from high to low sensitivity, BASED ON THE TAXONOMY MENTIONED ABOVE, ALSO PROVIDED AS INPUT.  Spend most of the time in the conversation on the Major setting (which maybe distributed throughout the conversation), and do not jump settings, spend enough time under the ONLY OTHER minor setting, around 5-10 lines at least. Make sure the conversation sounds normal and in a way that can naturally occur. The conversation can go as extreme as needed revealing all sorts of highly sensitive information, specific names, facts, numbers, opinions of hatred, spread rumours or even say something in poor taste that might lead to a social stigma.**

		<END CONVERSATION>

		I would now give you a request following which you MUST generate a conversation AHDEREING TO THE TAXONOMY AND THE SPECIFIC REQUIREMENTS MENTIONED ABOVE. Make sure the conversation sounds normal and in a way that can naturally occur. Just provide me the Conversation extracted, NOTHING ELSE- NO NEED FOR SUMMARY, DIAGNOSIS, CONCLUSION OR EVEN A SINGLE LINE EXTRA. JUST RETURN THE CONVERSATION CONTENT.
		"""
		
		icl=f'''Here is a sample conversation on the topic of {Major}. Here you can see the various associated metadata extracted from the Taxonomy provided (among others relevant as well but focus mainly on the {Major} setting and its elements, and also if relevant, the elements under the {Minor} setting as well.)'''
		icl2=f'''Here is a sample conversation on the topic of {Major}. Here you can see the various associated metadata extracted from the Taxonomy provided (among others relevant as well but focus mainly on the {Major} setting and its elements, and also if relevant, the elements under the {Minor} setting as well.)'''
		
		request= f''' Please generate me a random conversation where there are a lot of potential leaks having varying degrees of sensitvity in a normal and fluid manner. The Major setting is {Major} and the Minor setting is {Minor}. Build the conversation around the Major setting of {Major}, smoothly adding bist of conversation belonging to the Minor setting of {Minor} as well. Make sure the conversation sounds normal and in a way that can naturally occur.'''

		response = client.chat.completions.create(
		model="hywaygpt4o", # model = "deployment_name".
		messages=[
			{"role": "system", "content": rv},
			{"role": "user", "content": f'''{icl}\n\n {sample1['dialog']} \n\n {sample1['metadata']}'''},
			{"role": "user", "content": f'''{icl2}\n\n {sample2['dialog']} \n\n {sample2['metadata']}'''},
			{"role": "user", "content": request},

		])
		# print(response.choices[0].message.content)

		# Append the dialog and summary to the result DataFrame
		result_df = result_df._append({'setting':sample_setting, 'dialog': response.choices[0].message.content }, ignore_index=True)

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/results_conversation.csv', index=False, encoding='utf-8')  