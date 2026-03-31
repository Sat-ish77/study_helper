"""
backend/dependencies.py
Supabase JWT verification middleware.
Every protected route calls get_current_user() as a dependency.
The user_id extracted here is the same TEXT user_id used in all sh_ tables.
"""

import os
from fastapi import Header, HTTPException, status
from supabase import create_client

def get_supabase():
    """Module-level Supabase singleton — no st.cache_resource needed."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)

# Singleton instance
_supabase = None

def get_db():
    global _supabase
    if _supabase is None:
        _supabase = get_supabase()
    return _supabase


async def get_current_user(authorization: str = Header(...)) -> str:
    """
    Verifies Supabase JWT from Authorization: Bearer <token> header.
    Returns user_id as TEXT string.
    Raises 401 if token is invalid or missing.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use: Bearer <token>"
        )

    token = authorization.replace("Bearer ", "").strip()

    try:
        sb = get_db()
        # Verify JWT with Supabase — returns user data if valid
        response = sb.auth.get_user(token)
        if not response or not response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        # Return user_id as string — matches TEXT type in all sh_ tables
        return str(response.user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(e)}"
        )