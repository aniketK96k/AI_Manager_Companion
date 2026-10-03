import os
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from langchain_core.tools import tool


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

            credentials = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(credentials.to_json())

    return credentials


@tool
def get_last_30_days_meetings() -> list[dict]:
    """
    Get all meetings from the user's primary Google Calendar
    during the last 30 days.

    Returns the meeting name, start time, end time,
    Google Meet link, description, and location.
    """

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

        events.extend(response.get("items", []))

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    meetings = []

    for event in events:

        start_data = event.get("start", {})
        end_data = event.get("end", {})

        start = start_data.get(
            "dateTime",
            start_data.get("date")
        )

        end = end_data.get(
            "dateTime",
            end_data.get("date")
        )

        meetings.append({
            "id": event.get("id"),
            "name": event.get(
                "summary",
                "No title"
            ),
            "start": start,
            "end": end,
            "meet_link": event.get("hangoutLink"),
            "description": event.get("description"),
            "location": event.get("location")
        })

    return meetings