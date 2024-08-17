import msal
import requests

# Replace these values with your application's details
client_id = '91063160-d347-4cf7-9485-369bac23516b'
client_secret = 'w.N8Q~ndH1xkZLlPcadLFC8gwYSy0l8jG7DAsbHV'
tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47'
authority = f'https://login.microsoftonline.com/{tenant_id}'
scopes = ['https://graph.microsoft.com/.default']

# Create a confidential client application
app = msal.ConfidentialClientApplication(
    client_id,
    authority=authority,
    client_credential=client_secret
)

# Acquire a token
result = app.acquire_token_for_client(scopes=scopes)

if 'access_token' in result:
    access_token = result['access_token']
     # Replace this with the actual user's email
    user_email = 't-ppurkayast@microsoft.com'
    
    # Make an API call to fetch mails
    endpoint = f'https://graph.microsoft.com/v1.0/users/{user_email}/messages'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    response = requests.get(endpoint, headers=headers)

    if response.status_code == 200:
        mails = response.json()
        for mail in mails['value']:
            print(f"Subject: {mail['subject']}")
            print(f"From: {mail['from']['emailAddress']['address']}")
            print(f"Received: {mail['receivedDateTime']}")
            print()
    else:
        print(f'API call failed: {response.status_code} - {response.text}')
else:
    print('Failed to acquire token')



# import msal
# import requests

# # Replace these values with your application's details
# client_id = '91063160-d347-4cf7-9485-369bac23516b'
# client_secret = 'w.N8Q~ndH1xkZLlPcadLFC8gwYSy0l8jG7DAsbHV'
# tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47'
# authority = f'https://login.microsoftonline.com/{tenant_id}'
# scopes = ['User.Read', 'Mail.Read']

# # Create a public client application (for delegated authentication)
# app = msal.PublicClientApplication(client_id, authority=authority)

# # Obtain the authorization URL
# auth_url = app.get_authorization_request_url(scopes=scopes)

# # Print the authorization URL and follow the prompt to login and get the auth code
# print(f'Navigate to the following URL to authenticate and retrieve the authorization code:\n{auth_url}')

# # Paste the authorization code obtained from the redirect URL after authentication
# auth_code = input('Enter the authorization code: ').strip()

# # Acquire a token using the authorization code (delegated authentication)
# result = app.acquire_token_by_authorization_code(auth_code, scopes=scopes)

# if 'access_token' in result:
#     access_token = result['access_token']

#     # Make an API call to fetch mails for the authenticated user
#     endpoint = 'https://graph.microsoft.com/v1.0/me/messages'
#     headers = {
#         'Authorization': f'Bearer {access_token}',
#         'Accept': 'application/json'
#     }
#     response = requests.get(endpoint, headers=headers)

#     if response.status_code == 200:
#         mails = response.json()
#         for mail in mails['value']:
#             print(f"Subject: {mail['subject']}")
#             print(f"From: {mail['from']['emailAddress']['address']}")
#             print(f"Received: {mail['receivedDateTime']}")
#             print()
#     else:
#         print(f'API call failed: {response.status_code} - {response.text}')
# else:
#     print(f'Failed to acquire token: {result.get("error_description", "Unknown error")}')

