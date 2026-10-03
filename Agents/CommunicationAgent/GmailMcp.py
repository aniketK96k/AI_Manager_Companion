"""Reusable helpers for talking to Google's Gmail MCP server."""
import os
from contextlib import asynccontextmanager

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://gmailmcp.googleapis.com/mcp/v1"
PROJECT_ID = "meeting-ai-agent-509213"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]  # delete token.json after changing this list

CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"  # cached login so the browser doesn't open every run


def get_access_token() -> str:
    """Return a valid access token, reusing/refreshing the cached one if possible."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds.token


@asynccontextmanager
async def gmail_session():
    """Async context manager that yields an initialized MCP ClientSession."""
    http_client = httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {get_access_token()}",
            "x-goog-user-project": PROJECT_ID,
        },
        timeout=httpx.Timeout(30, read=300),
        follow_redirects=True,
    )
    async with http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def list_gmail_tools():
    """Return the list of tools exposed by the Gmail MCP server."""
    async with gmail_session() as session:
        result = await session.list_tools()
        return result.tools


async def call_gmail_tool(name: str, arguments: dict | None = None):
    """Call one Gmail MCP tool by name and return the result."""
    async with gmail_session() as session:
        return await session.call_tool(name, arguments or {})