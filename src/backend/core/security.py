from datetime import datetime, timedelta, timezone
import jwt
from pwdlib import PasswordHash
from src.backend.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_MINUTES

_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """ hash password before saving it"""
    return _hasher.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    """ use for login to check password is correct"""
    return _hasher.verify(password, hashed)


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)