import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import re

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    # api_version="2023-03-15-preview",
    api_version="2024-02-15-preview",
    azure_endpoint="https://hywayllm-gpt4.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

rv=""" You are an Expert in generating a privacy preserving summary of any converstaion based on an Informational Data Privacy. The privacy preservation assurance should all be based on an Informational Data Privacy Taxonomy that will be provided to you now. Here is the Taxonomy-

<BEGIN INFORMATIONAL DATA PRIVACY TAXONOMY>

	1. Generic
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
				○ Ethnicity
	2. Family and Relationships
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
			§ General relationship status/ Marital status
	3. Healthcare Settings
		○ High Sensitivity: 
			§ Medical history
				□ genetic conditions
				□ Diseases
				□ Mental Health Issues
		○ Medium Sensitivity: 
			§ Health Insurance details
		○ Low Sensitivity: 
			§ General health status
	4. Employment
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
			§ General employment status
	5. Finances
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
			§ General financial status
	6. Social Media
		○ High Sensitivity: 
			§ Private chats
			§ personal posts
		○ Medium Sensitivity: 
			§ Friend lists
			§ group memberships
		○ Low Sensitivity: 
			§ Public posts
			§ Accounts followed
	7. Legal Proceedings
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
				□ Lawyers
	8. Political Activities
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
			§ Voting history
	9. Religious Contexts
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
			§ General religious affiliation
	10. Sexual Orientation and Gender Identity
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
			§ General demographic information
	11. Travel and Location
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
			§ General location information
	12. Education
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
				□ Batch

<END INFORMATIONAL DATA PRIVACY TAXONOMY>

You will be given a summary and some privacy violations in it, you are too remove all the violations in it and others if exist, and provide a new privacy preserving summary. The Violations have information about potential high sensitivity content. It will be provided within <BEGIN VIOLATIONS>......<END VIOLATIONS> Tags. All the Privacy elements are mentioned AS PER THE TAXONOMY GIVEN ABOVE, and follow the structure as level0.level1.level2 and go as deep as needed to capture the position as per the taxonomy to the maximum extent possible. The Summary provided aims to provide a gist of the conversation and can be found within <BEGIN SUMMARY>......<END SUMMARY> Tags.  VIOLATIONS IS NOT A HARD LIST, IT IS MERELY TO BE USED AS A REFERENCE ALTHOUGH IT MIGHT NOT HAVE SOME ADDITIONAL VIOLATIONS NOT INCLUDED IN THE SUMMARY SO YOU MUST ALSO ENSURE THAT THOSE VIOLATIONS ARE REMOVED AND PROVIDE A PROPER TAXONOMY ADHEREING PRIVACY PRESERVING SUMMARY. You are to remove all the violations mentioned from the summary, and also any other additional sensitive left in the summary.

You are to check if any of the SPECIFIC INFORMATION like Names, Company names, Age, Salary, Disease Names, etc from the Metadata are leaked into the Summary, and if so they should be reported as a Violation. Note that merely mentioning someone has a disease, someone works at a company, someone has a loan, someone passed with good grades, soneon's father/mother/relatives, someone has a case against them isn't a violation, SPECIFIC DETAILS LIKE EXACT NAMES, LOCATIONS, DISEASE NAMES, FAMILY MEMBER NAMES, EXACT GRADES, EXACT SCORES, EXACT COMPANY NAMES, EXACT SALARIES, EXACT LOAN NUMBERS, EXACT CHARGES IN CASES OF COURTS, NUMBERS LIKE SALARIES, etc. need to be revealed directly for a violation. THERE SHOULD BE PROPER NOUNS OR HARD NUMBERS, MERELY USING COMMON NOUNS OR GENERIC ADJECTIVES DO NOT MAKE A VIOLATION. Any Generic information or vague words from whch even if we can indirectly get some information IS NOT A VIOLATION AS IT LEAKS NO SENSITIVE INFORMATION AND MUST NOT BE FLAGGED AS A VIOLATION.

The New Summary generated should be able to clearly explain the intent and the gist of the conversation, which ensuring there are none of the prior violations, nor any other violations regarding privacy and sensitive information, ALL BASED ON THE TAXONOMY PROVIDED EARLIER. 

Maintain the following Guidelines I'll provide below: 

<BEGIN GUIDELINES>
1. Do not use specific names, brands, facts and figures in the summary generated - unless absolutely necessary within the context and settings of the conversation
2. Do not include sensitive information in the summary and keep it at a high level. 
3. Always strive for neutral and inclusive language.
4. Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories.
5. If a certain piece of information is irrelevant based on the context and settings of the conversation, say in a casual discussion the names of individuals or specific prices/brands/comapnies/ figures, they can simple be masked with something generic/representative or omitted altogether, BUT ALL THIS SHOULD BE DONE IN ACCORDANCE WITH THE TAXONOMY PROVIDED.
6. ABSOLUTELY STICK TO THE TAXONOMY MENTIONED ABOVE AND DO NOT DEVIATE IN ANY WAY FROM IT.
7. REMOVE ALL THE VIOLATIONS MENTIONED UNDER VIOLATiONS TAG, AND ALSO ANY OTHERS IF ANY.
<END GUIDELINES>

MANDATORILY FOLLOW THIS STRUCTURE IN THE RESPONSE:

<BEGIN SUMMARY>
** Provide New Privacy Preserving Summary here without any of the Violations mentioned, and also other violations if any **
<END SUMMARY>

I would now give you a summary within <BEGIN SUMMARY>......<END SUMMARY> Tags and its violations within <BEGIN VIOLATIONS>......<END VIOLATIONS> Tags for which you MUST provide a response based on the structure provided above. Just provide me the Outcomes generated - NEW PRIVACY PRESERVING SUMMARY, NOTHING ELSE- NO NEED FOR DIAGNOSIS, CONCLUSION OR EVEN A SINGLE LINE EXTRA. JUST RETURN THE OUTCOMES CONTENT IN THE FORMAT ASKED.
"""

datasets_to_process = ['DialogSum', 'TweetSumm', 'ConvoSumm', 'SAMSum']

for dataset_name in datasets_to_process:
	dataset_path = f'./Datagen/datasets/{dataset_name}.csv'  # Assuming the datasets are in CSV format
	df = pd.read_csv(dataset_path)

	print(f"-------------- {dataset_name} STARTED ---------------------------")
	
	for i, (summary,label, violations) in enumerate(zip(df['summary'], df['Quality'], df['Violations'])): 
		
		# if(i==1):
		# 	break

		print(i,"\n")


		if "good" in label.lower():
			df.at[i, 'Corrected_Summary'] = summary

		else:
			# Create the prompt using both summary and metadata with triple quotes
			prompt = f'''
			Summary: {summary} 
			Violations: {violations}
			'''
			
			response = client.chat.completions.create(
			model="hywaygpt4o", # model = "deployment_name".
			messages=[
				{"role": "system", "content": rv},
				{"role": "user", "content": prompt},
			])

			df.at[i, 'Corrected_Summary'] = response.choices[0].message.content

	# Print the result DataFrame
	print(df)

	# Optionally, save the result DataFrame to a new CSV file
	df.to_csv(f'./datasets/corrected_{dataset_name}.csv', index=False, encoding='utf-8')  # Replace with the actual file path

	print(f"-------------- {dataset_name} COMPLETED ---------------------------")
