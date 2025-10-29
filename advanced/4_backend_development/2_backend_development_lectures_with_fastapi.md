---

## Module 4: Crafting Robust & Secure APIs

### 1) Data Validation (Never trust input)
Use Pydantic models to validate and document payloads. Validation happens before your handler runs.

```python
# validation_example.py
from fastapi import FastAPI
from pydantic import BaseModel, condecimal, constr

app = FastAPI(title="Validation Demo")

class ProductIn(BaseModel):
    name: constr(min_length=2, max_length=200)
    price: condecimal(max_digits=10, decimal_places=2)
    quantity: int = 0

@app.post("/validate")
def validate_product(p: ProductIn):
    return {"ok": True, "product": p}
```

### 2) Configuration & Secrets Management
Store secrets in environment variables; never commit them. Railway exposes env vars in the UI.

```python
# security_config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60

settings = Settings()  # reads from env (and .env in dev if configured)
```

### 3) Authentication vs Authorization
- **Authentication**: Who are you?
- **Authorization**: What can you do?

We’ll implement stateless auth with JWT.

### 4) Token-Based Authentication (JWT)
We’ll create a `users` table, register and login endpoints, and protected routes.

```python
# auth_models.py
from sqlalchemy import Integer, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

```python
# auth_schemas.py
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

```python
# auth_service.py
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt
from sqlalchemy.orm import Session
from auth_models import User
from auth_schemas import UserCreate
from security_config import settings

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes or settings.jwt_expires_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def register_user(db: Session, data: UserCreate) -> User:
    if db.query(User).filter(User.email == data.email).first():
        raise ValueError("Email already registered")
    user = User(email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate(db: Session, email: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.email == email).first()
    if user and verify_password(password, user.hashed_password):
        return user
    return None
```

```python
# auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import get_db
from auth_schemas import UserCreate, Token
from auth_service import register_user, authenticate, create_access_token
import jwt
from security_config import settings

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        user = register_user(db, payload)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    token = create_access_token(subject=user.email)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token(subject=user.email)
    return {"access_token": token, "token_type": "bearer"}

def get_current_user_email(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
```

Protect a route:
```python
# main_secure.py
from fastapi import FastAPI, Depends
from database import Base, engine
from auth_routes import router as auth_router, get_current_user_email

app = FastAPI(title="Secure API")
Base.metadata.create_all(bind=engine)
app.include_router(auth_router)

@app.get("/me")
def me(email: str = Depends(get_current_user_email)):
    return {"email": email}
```

Packages needed (add if missing):
```bash
pip install bcrypt PyJWT python-multipart
```

### 5) Error Handling
Create custom exceptions and handlers to return consistent error shapes.

```python
# errors.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class DomainError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

def install_error_handlers(app: FastAPI):
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
```

Use it in your app:
```python
# main_secure.py (excerpt)
from errors import install_error_handlers, DomainError

install_error_handlers(app)

@app.get("/boom")
def boom():
    raise DomainError("Something went wrong in business logic", 422)
```

---

## Module 5: Quality, Testing, and Advanced Topics

### 1) Software Architecture & Project Structure
Refactor into layers: `routers/`, `schemas/`, `models/`, `services/`, `core/`.

```bash
.
├─ app/
│  ├─ main.py
│  ├─ core/ (config, security, errors)
│  ├─ models/
│  ├─ schemas/
│  ├─ services/
│  └─ routers/
└─ tests/
```

Example service function:
```python
# services/products_service.py
from sqlalchemy.orm import Session
from models import Product
from schemas import ProductCreate

def create_product(db: Session, payload: ProductCreate) -> Product:
    if db.query(Product).filter(Product.name == payload.name).first():
        raise ValueError("Product name exists")
    p = Product(**payload.dict())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p
```

### 2) API Testing Strategies
- Unit tests: test services in isolation.
- Integration tests: spin up app with a test DB (separate schema/database) or mock DB.

Install testing tools:
```bash
pip install pytest httpx
```

Code example with FastAPI TestClient:
```python
# tests/test_products.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_create_and_list_products(monkeypatch):
    # In real integration tests, point DATABASE_URL to a test DB.
    # Here we assume an in-memory or dedicated test environment.
    payload = {"name": "TestItem", "price": "9.99", "quantity": 3}
    r = client.post("/products", json=payload)
    assert r.status_code in (200, 201)
    r2 = client.get("/products")
    assert r2.status_code == 200
    assert any(p["name"] == "TestItem" for p in r2.json())
```

Run tests:
```bash
pytest -q
```

### 3) Asynchronous & Background Tasks
Use background tasks for non-blocking work inside a request.

```python
# background_tasks_example.py
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def send_welcome_email(email: str):
    # call email provider API here
    print(f"Sent welcome email to {email}")

@app.post("/register")
def register_user(email: str, background: BackgroundTasks):
    # create user in DB here
    background.add_task(send_welcome_email, email)
    return {"ok": True}
```

When work grows, consider a worker (Celery + Redis) for reliability and retries.

### 4) Real-Time Communication with WebSockets
Use WebSockets for push-style, bidirectional communication.

```python
# websocket_example.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/notify")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass
```

Client test (browser console):
```javascript
const ws = new WebSocket("wss://YOUR-APP.onrailway.app/ws/notify");
ws.onmessage = (e) => console.log(e.data);
ws.onopen = () => ws.send("hello");
```

---

## Module 6: Final Capstone Project & Continuous Deployment on Railway

### 1) Project Scoping and Design
Build a simple blogging platform with users, posts, and comments.
- Users: register/login (JWT), list profile
- Posts: CRUD; owner = user; pagination
- Comments: CRUD; linked to posts and users

### 2) Full Application Implementation & Incremental Deployment
Workflow:
- Implement feature → push to GitHub → Railway deploys automatically
- Maintain DB migrations with Alembic for each schema change
- Use feature branches and PRs for reviews (optional)

### 3) What the Final Codebase Demonstrates
- **Authentication**: JWT-based login and protected routes
- **Relational Modeling**: Users–Posts–Comments with foreign keys
- **Validation & Errors**: Pydantic models and custom handlers
- **Structure**: Routers, services, schemas, models, core modules
- **CD**: GitHub → Railway push-to-deploy pipeline

### 4) Continuous Deployment Setup on Railway
- Connect GitHub repo to Railway service.
- Configure start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Environment variables: `DATABASE_URL`, `JWT_SECRET`, etc.
- Create migrations per feature:
```bash
alembic revision --autogenerate -m "add comments"
alembic upgrade head
```

### 5) Production Monitoring (Basics)
In Railway UI:
- **Logs**: View live application logs for errors and requests.
- **Metrics**: Check CPU/memory usage to right-size your service.
- **Env Vars**: Rotate secrets safely; no code changes required.

Practical:
- Trigger a deployment and watch logs during rollout.
- Validate health endpoints and DB connectivity post-deploy.

This final capstone consolidates all modules into a continuously deployed, monitored API on Railway, ready for real use and extension.

### Backend Development Lectures with FastAPI (Modules 1–3 + Capstone 1)

This curriculum is hands-on and cloud-first. Each module includes short theory, practical steps, and copy-pasteable code you can run locally and deploy to the cloud (Railway + PostgreSQL).

---
