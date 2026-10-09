from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from psycopg import Connection

from src.backend.core.config import SECRET_KEY, ALGORITHM
from src.backend.db.session import get_db
from src.backend.db.users import get_user_by_id
from src.genration_pipeline.pipeline import LegalSaathiPipeline

def get_pipeline(request: Request) -> LegalSaathiPipeline:
    return request.app.state.pipeline


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)],db: Annotated[Connection, Depends(get_db)],) -> dict:
    """ validate the token and user account return user info """
    error = HTTPException(status.HTTP_401_UNAUTHORIZED,"Could not validate credentials",headers={"WWW-Authenticate": "Bearer"},)

    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM],
            options={"require": ["sub", "exp"]},
        )
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise error

    user = get_user_by_id(db, user_id)

    if not user or not user["is_active"]:
        raise error
    return user




CurrentUser = Annotated[dict, Depends(get_current_user)]

def require_role(*roles: str):
    def checker(user: CurrentUser) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not enough permissions")
        return user
    return checker





