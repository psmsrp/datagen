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

# ---------------------BEGIN PRELOADING--------------------------------
settings=["Family and Relationships","Healthcare Settings","Employment","Finances","Social Media","Legal Proceedings","Political Activities","Religious Contexts","Sexual Orientation and Gender Identity","Travel and Location","Education"]

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
file_path = './results/results_conversation.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Initialize a new DataFrame for the results
result_df = pd.DataFrame(columns=['setting','dialog', 'metadata'])
# result_df = pd.DataFrame(columns=['dialog', 'summary_low_short', 'summary_low', 'summary_high_short', 'summary_high', 'summary_context_short', 'summary_context'])

rv=""" You are an Expert in extracting metadata from any converstaion. The metadata should contain information about the Context, the Settings and the list of privacy-related elements mentioned in the conversation. It should all be based on an Informational Data Privacy Taxonomy that will be provided to you now. Here is the Taxonomy-

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

 
The Metadata should encompasss every single bit of relevant detail from the taxonomy. An element can belong to multiple categories under the taxonomy, you'll have to extract all such privacy elements, and label them using one or many applicable categories, to the maximum depth you can. The goal of the Metadata should be that when we see the conversation, we would get a complete high level information of what is happening in the current context, under the current setting. 

Here, the context can be a meeting, an event, a conference or anything. The setting can be generic, healthcare, education, etc as mentioned in the taxonomy, but it MUST BE A SINGLE SETTING. CHOOSE THE ONE SETTING (Given 12 are for a start, if none fit best only then specify another one) AND ONLY ONE THAT FITS CONVERSATION SETTING BEST. The privacy sensitive elements are what come under those, they can be as deep and specific as needed like family.high.marital_records.partners.names or can be as generic as employment.high.emplyment_status. 
 
When you extract Privacy sensitive elements for metadata, be sure to categorize them to the maximum possible extent in depth under the given taxonomy. It may also happen the same data falls under multiple such labels, in which case include them all. Whatever elements you mention under each setting, metion the category based on the taxonomy. MAKE SURE IT MUST COMPULSORILY FALL EXACTLY UNDER ONE OF THE LEVELS OF HEIRARCHY PROPOSED BY THE TAXONOMY LIKE family.high.marital_records.partners.names or employment.high.emplyment_status. IF at all, it falls outside the Taxonomy, you must then categorize it accordingly in a similar taxonomical fashion. If for a Setting there are no Privacy elements, or in case anywhere there a certain category isn't applicable, Do not mention NA or along the lines of no match found / None mentioned explicitly - Simply omit mentioning anything about that Setting or Category altogether. Do not mention the elements under the actual single Setting identified differently- just include it at the top of the same list, after which any other relevant elements under any other relevant settings is mentioned but do so STRICTLY ADHERING TO THE TAXONOMY FORMAT PROVIDED ABOVE. 

EXPLICITLY MENTION ONLY A SINGLE SETTING IDENTIFIED AS THE THEME OF THE CONVERSATION. All the other relevant settings can be followed along with privact elemments later- MANDATORILY FOLLOW THIS STRUCTURE IN RESPONSE:

<BEGIN METADATA>

Context: ** Enter Context here **
Setting: ** Enter ONLY THE IDENTIFIED SETTING here **
Elements: ** Make a list here of all the Privacy elements FOLLOWING THE ABOVE TAXONOMY AND IN THE SAME FORMAT MENTIONED EARLIER grouped under all the relevant/ applicable Settings, starting with the one identified as the Main Setting in the previous field above, eg: 
	1. SETTING IDENTIFIED ABOVE (Replace with actual Setting): 
		1.Taxonomical specific category:
        	Elements identified listed
        2.Taxonomical specific category:
        	Elements identified listed
    
    2. SETTING 2 (Replace with actual Setting): 
       	1.Taxonomical specific category:
        	Elements identified listed
        2.Taxonomical specific category:
        	Elements identified listed

			..... and so on **

<END METADATA>


I would now give you a piece of text for which you MUST extract a detailed metadata regarding the conversation, covering all the possible elements, and grouping them accordingly into one or many relevant categories COMPULSORILY ADHEREING TO THE TAXONOMY. Just provide me the Metadata extracted, NOTHING ELSE- NO NEED FOR SUMMARY, DIAGNOSIS, CONCLUSION OR EVEN A SINGLE LINE EXTRA. JUST RETURN THE METADATA CONTENT.
"""
print(len(df),len(df['setting']), len(df['dialog']))

for i,(Major,text) in enumerate(zip(df['setting'],df['dialog'])):
    
    print(i,"\n")
    sample1 = random.choice(sampleset[Major])
    remaining_samples = [sample for sample in sampleset[Major] if sample != sample1]
    sample2 = random.choice(remaining_samples)
    icl=f'''Here is a sample conversation on the topic of {Major}. Here you can see the various associated metadata extracted from the Taxonomy provided (among others relevant as well but focus mainly on the {Major} setting and its elements, and also if relevant, the elements under the other Minor settings as well.)'''
    icl2=f'''Here is a sample conversation on the topic of {Major}. Here you can see the various associated metadata extracted from the Taxonomy provided (among others relevant as well but focus mainly on the {Major} setting and its elements, and also if relevant, the elements under the other Minor settings as well.)'''

    response = client.chat.completions.create(
    model="hywaygpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": rv},
        {"role": "user", "content": f'''{icl}\n\n {sample1['dialog']} \n\n {sample1['metadata']}'''},
		{"role": "user", "content": f'''{icl2}\n\n {sample2['dialog']} \n\n {sample2['metadata']}'''},
        {"role": "user", "content": text},

    ])

    # Append the dialog and summary to the result DataFrame
    result_df = result_df._append({'setting': Major,'dialog': text, 'metadata' : response.choices[0].message.content }, ignore_index=True)

# Print the result DataFrame
print(result_df)

# Optionally, save the result DataFrame to a new CSV file
result_df.to_csv('./results/results_metadata.csv', index=False, encoding='utf-8')  