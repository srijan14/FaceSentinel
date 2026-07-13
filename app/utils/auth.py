from fastapi import HTTPException, Depends, Header
from typing import Optional
from app.config import settings
from app.utils.exceptions import AuthenticationError, create_http_exception


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    """Verify the authorization token directly"""
    try:
        if not authorization:
            raise AuthenticationError("Missing authorization header")

        # Direct token comparison without Bearer prefix
        if authorization != settings.api_key:
            raise AuthenticationError("Invalid API key")

        return authorization

    except AuthenticationError as e:
        raise create_http_exception(e)
    except Exception:
        raise create_http_exception(AuthenticationError())
