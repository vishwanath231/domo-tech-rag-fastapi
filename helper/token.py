from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "domo-rag"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=1)):
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token
