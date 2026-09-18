import os
from mcp.server.fastmcp import FastMCP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]

mcp = FastMCP("Google Calendar")


class CalendarService:

    def __init__(self):
        self.service = self._connect()

    def _connect(self):
        creds = None

        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file(
                "token.json", SCOPES
            )

        if not creds or not creds.valid:

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open("token.json", "w") as f:
                f.write(creds.to_json())

        return build("calendar", "v3", credentials=creds)

    def create_event(self, title, start_time, duration_minutes):
        from datetime import datetime, timedelta

        start = datetime.fromisoformat(start_time)
        end = start + timedelta(minutes=duration_minutes)

        event = {
            "summary": title,
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": "Asia/Kolkata"
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": "Asia/Kolkata"
            }
        }

        return self.service.events().insert(
            calendarId="primary",
            body=event
        ).execute()


calendar = CalendarService()


@mcp.tool()
def create_calendar_event(
    title: str,
    start_time: str,
    duration_minutes: int = 30
) -> str:
    """Create an event in Google Calendar."""

    event = calendar.create_event(
        title,
        start_time,
        duration_minutes
    )

    return (
        f"Event created successfully. "
        f"Title: {event['summary']}, "
        f"Event ID: {event['id']}"
    )


if __name__ == "__main__":
    mcp.run()