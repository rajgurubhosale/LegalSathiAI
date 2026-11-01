
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from psycopg import errors

from src.backend.api.dependencies import CurrentUser, DB
from src.backend.api.schemas import Token, UserCreate, UserOut
from src.backend.core.security import create_access_token, hash_password, verify_password
from src.backend.db.users import create_user, get_user_by_email


router = APIRouter(prefix="/auth", tags=["auth"])
Form =  Annotated[OAuth2PasswordRequestForm, Depends()]
@router.post("/register", response_model=UserOut, status_code=201)
async def register(data: UserCreate, db: DB):
    try:
        return await create_user(db,data.username, data.email, hash_password(data.password))

    except errors.UniqueViolation:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")


@router.post("/login", response_model=Token)
async def login(form:Form, db: DB):
    """ user login validation """
    user_dict = await get_user_by_email(db, form.username)      
    headers = {"WWW-Authenticate": "Bearer"}

    if not user_dict:
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers=headers)

    hashed_password = verify_password(form.password, user_dict['hashed_password'])
    if hashed_password is False:
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers=headers)

    if not user_dict["is_active"]:
        raise HTTPException(status_code=403, detail="This account is disabled.")

    return Token(access_token=create_access_token(user_dict["id"]))


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser):
    return user
