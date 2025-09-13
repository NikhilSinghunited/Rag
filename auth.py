import os
import base64
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Load .env
load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def get_inbox_emails(service, max_results=5):
    """Fetch emails from Gmail inbox"""
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])

    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = txt['payload']
        headers = payload.get("headers", [])
        
        subject = sender = None
        for h in headers:
            if h['name'] == 'Subject':
                subject = h['value']
            if h['name'] == 'From':
                sender = h['value']

        # Try to decode body (if exists)
        body = ""
        parts = payload.get("parts")
        if parts:
            data = parts[0]["body"].get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        emails.append({"from": sender, "subject": subject, "body": body})

    return emails


if __name__ == "__main__":
    svc = gmail_service()
    me = svc.users().getProfile(userId='me').execute()
    print("✅ Authenticated as:", me.get("emailAddress"))

    # Call the inbox fetch function
    inbox = get_inbox_emails(svc, max_results=5)
    for idx, mail in enumerate(inbox, 1):
        print(f"\n📧 Email {idx}")
        print("From:", mail["from"])
        print("Subject:", mail["subject"])
        print("Body preview:", mail["body"][:200])
