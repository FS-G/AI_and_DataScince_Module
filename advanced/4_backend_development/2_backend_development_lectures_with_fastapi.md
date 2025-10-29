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

## Module 1: The Foundations of Web Communication & Cloud Setup

### 1) Client–Server Model (What actually happens?)
- **Client**: A browser/mobile app/another server makes a request.
- **Server**: Your FastAPI app receives the request, runs code, and returns a response.
- **Stateless**: Each request is independent; servers don’t remember previous requests unless you store state (DB, cache, JWTs, etc.).

Visual: Client → Internet → Web Server (Uvicorn/Gunicorn) → App (FastAPI) → DB (PostgreSQL) → App → Client.

### 2) Communication Protocols Overview
- **TCP**: Reliable, ordered, connection-oriented. Used under HTTP/HTTPS.
- **UDP**: Fast, unreliable, used for real-time streaming/gaming.
- **HTTP/HTTPS**: Application protocol of the web; resources, methods, headers.
- **WebSockets**: Full-duplex, persistent connection for realtime apps.
- **gRPC**: High-performance RPC over HTTP/2 with protobuf schemas.

### 3) Deep Dive into HTTP/HTTPS
- **Methods**: GET (read), POST (create), PUT/PATCH (update), DELETE (remove)
- **Status Codes**: 200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 422 Unprocessable Entity, 500 Internal Server Error
- **Headers**: `Content-Type`, `Authorization`, `Accept`, `Cache-Control`
- **Bodies**: Typically JSON for APIs

Quick demo using curl:
```bash
curl -i https://httpbin.org/get
curl -i -X POST https://httpbin.org/post \
  -H "Content-Type: application/json" \
  -d '{"message":"hello"}'
```

### 4) APIs & REST (Principles)
- **Resource-oriented**: `/products`, `/orders`, `/employees`
- **Stateless**: Server doesn’t store client session state between requests
- **Uniform interface**: Predictable URLs, methods, and status codes
- **Representation**: JSON by default; use schemas for validation

### 5) Setting Up Your Cloud-Native Environment (Railway)
Prereqs: A GitHub account.

- Create an account on Railway (`https://railway.app`).
- Create a new project.
- Add a **PostgreSQL** service.
- Note the connection string (will be provided as `DATABASE_URL`).
- Install Railway CLI:
```bash
npm i -g @railway/cli
railway login
```

### 6) Local Development Environment
Install Python 3.10+ and Git.

```bash
# On Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install fastapi uvicorn[standard] sqlalchemy psycopg[binary] pydantic pydantic-settings alembic python-dotenv
```

Recommended files to start with:
```bash
echo "fastapi\nuvicorn[standard]\nsqlalchemy\npsycopg[binary]\npydantic\npydantic-settings\nalembic\npython-dotenv" > requirements.txt
echo "__pycache__/\n.venv/\n.env" > .gitignore
```

### 7) Version Control & Initial Deployment Prep
```bash
git init
git add .
git commit -m "chore: initial setup"
gh repo create yourname/fastapi-starter --public --source=. --remote=origin --push
```

Link to Railway and set env later after we have an app. Railway supports push-to-deploy from GitHub.

---

## Module 2: Building Your First Web Server

### 1) Web Server vs Web Framework
- **Web server** (Uvicorn/Gunicorn): Listens on a port, speaks HTTP protocol.
- **Framework** (FastAPI): Routing, validation, dependency injection, docs.

### 2) Routing: Map URLs to Code
Create `main.py` with a minimal FastAPI app and one route.

```python
# main.py
from fastapi import FastAPI

app = FastAPI(title="Hello FastAPI")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is running!"}
```

Run locally:
```bash
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000 and http://localhost:8000/docs
```

### 3) Handling Requests: Path, Query, Headers, Body

```python
# app_requests.py (examples; you can merge into main.py if preferred)
from typing import Optional
from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI(title="Request Handling Examples")

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

@app.get("/items/{item_id}")
def get_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "query": q}

@app.get("/headers")
def read_headers(user_agent: Optional[str] = Header(None)):
    return {"user_agent": user_agent}

@app.post("/items")
def create_item(item: Item):
    return {"status": "created", "item": item}
```

Key ideas:
- Type hints drive validation and docs.
- Path params: in the route template.
- Query params: regular function params with defaults.
- Headers: use `Header`.
- Body: Pydantic models.

### 4) Sending Responses (JSON by default)
Returning a `dict` automatically becomes JSON.

```python
@app.get("/health")
def health():
    return {"ok": True}
```

### 5) Deploying This Minimal App to Railway
Add a start command. Railway detects Python; you can set a start command in the service settings or add a `Procfile`.

```bash
echo "web: uvicorn main:app --host 0.0.0.0 --port $PORT" > Procfile
```

Push to GitHub, then in Railway:
- Create a new service from your GitHub repo.
- Set the start command or use `Procfile`.
- Deploy. Visit the provided URL.

---

## Module 3: Cloud Database Integration with PostgreSQL

### 1) RDBMS Fundamentals (PostgreSQL)
- **Schemas, Tables, Columns, Rows**
- **Primary Keys, Foreign Keys**
- **Relationships**: one-to-many, many-to-many

### 2) Load DATABASE_URL securely with Pydantic Settings
Create `.env` locally (do not commit) and set on Railway as an environment variable.

```
# .env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME
```

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", env_file_encoding="utf-8")

settings = Settings()
```

### 3) SQLAlchemy Engine, Session, and Models

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

```python
# models.py
from sqlalchemy import Column, Integer, String, Numeric, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from database import Base

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
```

```python
# schemas.py
from pydantic import BaseModel, condecimal

class ProductCreate(BaseModel):
    name: str
    price: condecimal(max_digits=10, decimal_places=2)
    in_stock: bool = True

class ProductRead(BaseModel):
    id: int
    name: str
    price: condecimal(max_digits=10, decimal_places=2)
    in_stock: bool

    class Config:
        from_attributes = True
```

### 4) CRUD Endpoints (Live DB)

```python
# main.py (CRUD edition)
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from models import Product
from schemas import ProductCreate, ProductRead

app = FastAPI(title="Products API")
Base.metadata.create_all(bind=engine)

@app.post("/products", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    exists = db.query(Product).filter(Product.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Product with this name already exists")
    product = Product(name=payload.name, price=payload.price, in_stock=payload.in_stock)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product

@app.get("/products/{product_id}", response_model=ProductRead)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/products", response_model=list[ProductRead])
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).order_by(Product.id).all()

@app.put("/products/{product_id}", response_model=ProductRead)
def update_product(product_id: int, payload: ProductCreate, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    product.name = payload.name
    product.price = payload.price
    product.in_stock = payload.in_stock
    db.commit()
    db.refresh(product)
    return product

@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    db.delete(product)
    db.commit()
    return None
```

Test locally:
```bash
uvicorn main:app --reload
curl -X POST http://localhost:8000/products -H "Content-Type: application/json" -d '{"name":"Pen","price":"1.50","in_stock":true}'
curl http://localhost:8000/products
```

### 5) Database Migrations with Alembic
Initialize Alembic and configure it to use your SQLAlchemy `Base` metadata.

```bash
alembic init alembic
```

Edit `alembic.ini` to use `sqlalchemy.url` via env variable (optional), and `alembic/env.py` to target your metadata:

```python
# alembic/env.py (excerpt)
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from database import Base
from models import Product

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline():
    url = os.getenv("DATABASE_URL")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        {"sqlalchemy.url": os.getenv("DATABASE_URL")}, prefix="", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create and apply migrations:
```bash
alembic revision --autogenerate -m "create products table"
alembic upgrade head
```

### 6) Deploy with Railway (DB + App)
- Set `DATABASE_URL` in Railway service settings (copy from the Railway Postgres plugin).
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Run Alembic migrations in a Railway shell or via a deploy hook.

Optional `Procfile`:
```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Capstone Project 1: Mini E‑Commerce Inventory API (FastAPI + PostgreSQL)

Goal: Build a small, production-like API with proper DB models and CRUD. The app tracks products and employees and allows simple inventory operations.

### Features
- Manage products: create/list/get/update/delete
- Manage employees: create/list/get/update/delete
- Track stock levels on products (simple integer quantity)

### Data Model
- `Product`: `id`, `name`, `price`, `quantity` (integer stock), `is_active`
- `Employee`: `id`, `full_name`, `email`, `is_active`

### Suggested Project Structure
```
.
├─ main.py
├─ config.py
├─ database.py
├─ models.py
├─ schemas.py
├─ routers/
│  ├─ products.py
│  └─ employees.py
├─ alembic/
├─ requirements.txt
└─ Procfile
```

### Code: Models and Schemas
```python
# models.py
from sqlalchemy import Integer, String, Boolean, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from database import Base

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

```python
# schemas.py
from pydantic import BaseModel, EmailStr, condecimal

class ProductBase(BaseModel):
    name: str
    price: condecimal(max_digits=10, decimal_places=2)
    quantity: int = 0
    is_active: bool = True

class ProductCreate(ProductBase):
    pass

class ProductRead(ProductBase):
    id: int
    class Config:
        from_attributes = True

class EmployeeBase(BaseModel):
    full_name: str
    email: EmailStr
    is_active: bool = True

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeRead(EmployeeBase):
    id: int
    class Config:
        from_attributes = True
```

### Code: Routers
```python
# routers/products.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Product
from schemas import ProductCreate, ProductRead

router = APIRouter(prefix="/products", tags=["products"])

@router.post("", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    if db.query(Product).filter(Product.name == payload.name).first():
        raise HTTPException(400, "Product name already exists")
    product = Product(**payload.dict())
    db.add(product)
    db.commit()
    db.refresh(product)
    return product

@router.get("/{product_id}", response_model=ProductRead)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    return product

@router.get("", response_model=list[ProductRead])
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).order_by(Product.id).all()

@router.put("/{product_id}", response_model=ProductRead)
def update_product(product_id: int, payload: ProductCreate, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    for field, value in payload.dict().items():
        setattr(product, field, value)
    db.commit()
    db.refresh(product)
    return product

@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    db.delete(product)
    db.commit()
    return None
```

```python
# routers/employees.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Employee
from schemas import EmployeeCreate, EmployeeRead

router = APIRouter(prefix="/employees", tags=["employees"])

@router.post("", response_model=EmployeeRead, status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)):
    if db.query(Employee).filter(Employee.email == payload.email).first():
        raise HTTPException(400, "Email already exists")
    employee = Employee(**payload.dict())
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee

@router.get("/{employee_id}", response_model=EmployeeRead)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    return emp

@router.get("", response_model=list[EmployeeRead])
def list_employees(db: Session = Depends(get_db)):
    return db.query(Employee).order_by(Employee.id).all()

@router.put("/{employee_id}", response_model=EmployeeRead)
def update_employee(employee_id: int, payload: EmployeeCreate, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    for field, value in payload.dict().items():
        setattr(emp, field, value)
    db.commit()
    db.refresh(emp)
    return emp

@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    db.delete(emp)
    db.commit()
    return None
```

### Code: App Entrypoint
```python
# main.py
from fastapi import FastAPI
from database import Base, engine
from routers import products, employees

app = FastAPI(title="Mini E‑Commerce Inventory API")
Base.metadata.create_all(bind=engine)

app.include_router(products.router)
app.include_router(employees.router)

@app.get("/health")
def health():
    return {"ok": True}
```

### Migrations and Deployment
1) Initialize Alembic and generate migrations after defining models
```bash
alembic init alembic
alembic revision --autogenerate -m "init products and employees"
alembic upgrade head
```

2) Railway Deployment
- Add your repo as a Railway service.
- Set env var `DATABASE_URL` from the Railway PostgreSQL plugin.
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` (via service settings or `Procfile`).
- Run Alembic migrations in a Railway shell.

### Quick Test Commands
```bash
# Products
curl -X POST $URL/products -H "Content-Type: application/json" -d '{"name":"Notebook","price":"3.99","quantity":25}'
curl $URL/products

# Employees
curl -X POST $URL/employees -H "Content-Type: application/json" -d '{"full_name":"Jane Doe","email":"jane@example.com"}'
curl $URL/employees
```

That’s it—by the end of Module 3 and this capstone, you will have a real, cloud-deployed FastAPI + PostgreSQL application with migrations, clean routing, and production-ready structure.

