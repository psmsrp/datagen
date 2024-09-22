import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import random

# Load the CSV file into a DataFrame
file_path = './results/results_metadata.csv'  # Replace with the actual file path
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
result_df = df
# result_df = pd.DataFrame(columns=['dialog', 'summary_low_short', 'summary_low', 'summary_high_short', 'summary_high', 'summary_context_short', 'summary_context'])

rv1=""" You are an Expert in providing a detailed privacy preserving summary from any converstaion. The summary should be in 50-250 words at max (can be less) and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting.

The Privacy Preservation should all be based on an Informational Data Privacy Taxonomy that will be provided to you now. Here is the Taxonomy-

<BEGIN INFORMATIONAL DATA PRIVACY TAXONOMY>

	1. Generic
		○ High Sensitivity: 
			○ Slangs
				○ Profanity
				○ Insults
				○ Mockery
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
			○ Age
			○ Weight
			○ Sizes
				○ Clothes
				○ Shoes
				○ Shirts
				○ Pants
		○ Medium Sensitivity: 
			○ Username/ Social handle
			○ Physical Features
				○ Height
				○ Build
				○ Complexion
				○ hair
				○ Face
					® Eyes
					® Nose
					
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
				□ Number of members
		○ Low Sensitivity: 
			§ General relationship status/ Marital status
			§ Family members
				□ Relations
					® Father
					® Mother
					® Brother
					® Sister
					® Cousin
					® Other relatives
	3. Healthcare Settings
		○ High Sensitivity: 
			§ Medications
			§ Medical History
			§ Genetic conditions
			§ Diseases
			§ Mental Health Issues
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
				□ Coworker names
				□ Work culture
				□ Performance
		○ Medium Sensitivity:
			§ Volunteering
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
			§ Insurance
				□ Amount / Premium
				□ Beneficiaries
			§ Loan
				□ Amount
				□ Interest
			§ Debt
				□ Amount
				□ Interest
			§ investment information
				□ Portfolio-related information
					® Amounts
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
				□ Settlement Amounts
			§ Civil case details
				□ Settlement Amounts
		○ Medium Sensitivity: 
			§ Civil case details
				□ Lawsuits
				□ Settlements
		○ Low Sensitivity: 
			§ Legal representation contact information
				□ Firms
				□ Lawyers
				□ Fees
	8. Political Activities
		○ High Sensitivity: 
			§ Membership in political organizations (Specific names)
				□ NGOs
				□ Committees
				□ Volunteer Work
			§ Political Involvement
				□ Political Parties
				□ Political opinions
				□ activism details
					®  Meeting Attendance 
					® Membership Fees
					® Donations
				□ Roles in propaganda/ agendas
			§ Voting Details
				□ Voting Records
				□ Ballot Details
				□ Voting Dates
				□ Voting Locations
		○ Medium Sensitivity: 
			§ Membership in political organizations (Generic or vague description)
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
			§ Common Ceremonies
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
			§ Bookings (Hotels/Restaurants)
				□ Names
				□ Booking Dates/ Timings
				□ Room Numbers
				□ Room sharers
			§ GPS data
				□ Current Location
				□ Geo-tagged Photos
			§ Modes of Transportation
				□ Vehicle Numbers
				□ Vehicle Models
				□ Vehicle Plans
				□ Vehicle Rentals
		○ Medium Sensitivity: 
			§ Modes of Transportation
			§ Recent Travels(Generic)
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
			§ Degree details
				□ Degrees Earned
				□ Majors
				□ Minors
			§ School attended
				□ Name
				□ Fees
			§ College attended
				□ Name
				□ Fees
		○ Medium Sensitivity: 
			§ School attended
				□ Batch/ Year
			§ College attended
				□ Batch/ Year
		○ Low Sensitivity: 
			§ School attended
				□ Country
			§ College attended
				□ Country
			§ Future Plans

<END INFORMATIONAL DATA PRIVACY TAXONOMY>

The Summary provided should be able to clearly explain the intent and the gist of the conversation. For Privacy Preservation, depending on the Context and Setting of the conversation, certain information need to be masked. Based on the taxonomy given above, if the identified element is of a setting (like Health, education, etc.) different from the MAIN SETTING of the Conversation, then it should probably be censored, i.e use a generic word to replace it or omit it altogether without even hinting anything towards it, so as to make it untraceable. 

Ideally, all High Sensitivity information must be masked or replaced with something generic. For the Medium Sensitivity elements, depending on whther the setting it belongs to is relevant to the setting of the conversation, it can be decided whether to mask that or let it be. For the Low sensitivity elements, if irrelevant to the conversation can exclude for summary, else can follow a similar discretion like Medium Sensitivity elements whether to include or not based on context and settings.

Maintain the following Guidelines I'll provide below: 

<BEGIN GUIDELINES>
1. Do not use specific names, brands, facts and figures in the summary generated - unless absolutely necessary within the context and settings of the conversation
2. Do not include sensitive information in the summary and keep it at a high level. 
3. Always strive for neutral and inclusive language.
4. Do not generate content that includes hate speech, discrimination, or bias based on gender, race, ethnicity, religion, or other social categories.
5. You are to ensure that none of the SPECIFIC INFORMATION like Names, Company names, Age, Salary, Disease Names, etc from the Metadata are leaked into the Summary.
6. If a certain piece of information is irrelevant based on the context and settings of the conversation, say in a casual discussion the names of individuals or specific prices/brands/comapnies/ figures, they can simple be masked with something generic/representative or omitted altogether, BUT ALL THIS SHOULD BE DONE IN ACCORDANCE WITH THE TAXONOMY PROVIDED.
7. ABSOLUTELY STICK TO THE TAXONOMY MENTIONED ABOVE AND DO NOT DEVIATE IN ANY WAY FROM IT.
<END GUIDELINES>

MANDATORILY FOLLOW THIS STRUCTURE IN RESPONSE:

<BEGIN SUMMARY>

	** Enter Summary here **

<END SUMMARY>

I will be providing you I would now give you a piece of text for which you MUST provide a detailed Privacy preserving summary of the conversation in 4-5 sentences max (can be less), and YOU MUST FOLLOW THE TAXONOMY PROVIDED ABOVE. Just provide me the Summary, NOTHING ELSE- NO NEED FOR extra explanations on the SUMMARY, DIAGNOSIS, CONCLUSION OR EVEN A SINGLE LINE EXTRA. JUST RETURN THE SUMMARY STRUCTURE CONTENT.
"""


rv2=""" You are an Expert in providing a detailed summary from any converstaion. The summary should be in 4-5 sentences at max (can be less) and the goal of the summary should be that when new users join the conversation, they would get a complete high information summary of what is happening in the current setting. No need to worry about privacy and feel free to include sensitive information from the conversation directly into the summary. The Summary provided should be able to clearly explain the intent and the gist of the conversation. 

MANDATORILY FOLLOW THIS STRUCTURE IN RESPONSE:

<BEGIN SUMMARY>

	** Enter Summary here **

<END SUMMARY>

Maintain the following Guidelines I'll provide below: 

<BEGIN GUIDELINES>
1. Try to expose specific names, brands, facts and figures in the summary generated
2. Always include sensitive information in the summary and go into it at a deeper level. 
3. Feel free to include biased and hateful remarks in the Summary .
4. It is alright if generated content indicates hateful remarks, discrimination/bias based on gender, race, ethnicity, religion, or other social categories.
<END GUIDELINES>


I will be providing you I would now give you a piece of text for which you MUST provide a detailed summary of the conversation in 4-5 sentences max (can be less), and YOU MUST LEAK SOME KIND OF SENSITIVE INFORMATION FROM THE CONVERSATION DIRECTLY INTO THE SUMMARY. Just provide me the Summary, NOTHING ELSE- NO NEED FOR extra explanations on the SUMMARY, DIAGNOSIS, CONCLUSION OR EVEN A SINGLE LINE EXTRA. JUST RETURN THE SUMMARY STRUCTURE CONTENT.
"""


for i,text in enumerate(df['dialog']):
    # if(i==1):
    #     break
    print(i,"\n")
    # rv = rv2
    rv = rv1 if ((random.randint(1, 100))%2) else rv2

    response = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv},
        {"role": "user", "content": text},

    ])

    # Append the dialog and summary to the result DataFrame
    result_df.at[i, 'summary'] = response.choices[0].message.content

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/results_summary.csv', index=False, encoding='utf-8')  # Replace with the actual file path