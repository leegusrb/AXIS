"""비밀번호 해시(표준 라이브러리 PBKDF2)와 JWT 토큰."""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi import HTTPException

_ITER = 200_000
_UNAUTHORIZED = "로그인이 필요해요"


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, _ITER)
    return f"pbkdf2_sha256${_ITER}${salt.hex()}${h.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        algo, it, salt, h = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    got = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it))
    return hmac.compare_digest(got.hex(), h)


def load_secret(var_dir: Path) -> str:
    """서버를 다시 켜도 토큰이 유효하도록 비밀키를 파일에 둔다."""
    p = Path(var_dir) / "jwt_secret"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(32))
        p.chmod(0o600)
    return p.read_text().strip()


def make_token(user_id: int, secret: str, days: int = 30) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=days)
    return jwt.encode({"sub": str(user_id), "exp": exp}, secret, algorithm="HS256")


def read_token(token: str, secret: str) -> int:
    try:
        return int(jwt.decode(token, secret, algorithms=["HS256"])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as e:
        raise HTTPException(401, _UNAUTHORIZED) from e


def bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, _UNAUTHORIZED)
    return authorization[len("Bearer "):]
