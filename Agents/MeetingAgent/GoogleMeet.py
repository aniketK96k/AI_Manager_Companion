import os
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly"
]


def authenticate():

    credentials = None

    if os.path.exists("token.json"):
        credentials = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            credentials = flow.run_local_server(
                port=0
            )

        with open("token.json", "w") as token:
            token.write(credentials.to_json())

    return credentials


def get_last_30_days():

    credentials = authenticate()

    service = build(
        "calendar",
        "v3",
        credentials=credentials
    )

    now = datetime.now(timezone.utc)

    thirty_days_ago = now - timedelta(days=30)

    events = []

    page_token = None

    while True:

        response = service.events().list(
            calendarId="primary",
            timeMin=thirty_days_ago.isoformat(),
            timeMax=now.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            pageToken=page_token
        ).execute()

        events.extend(
            response.get("items", [])
        )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return events


def main():

    events = get_last_30_days()

    print(
        f"\nFound {len(events)} calendar events\n"
    )

    for event in events:

        name = event.get(
            "summary",
            "No title"
        )

        start = event.get(
            "start",
            {}
        ).get(
            "dateTime",
            event.get("start", {}).get("date")
        )

        end = event.get(
            "end",
            {}
        ).get(
            "dateTime",
            event.get("end", {}).get("date")
        )

        meet_link = event.get(
            "hangoutLink"
        )

        print("=" * 70)

        print("Meeting:", name)
        print("Start:", start)
        print("End:", end)
        print("Google Meet:", meet_link)


if __name__ == "__main__":
    main()