# ============================================
# COMPLETE FASTAPI BLOG API WITH DETAILED EXPLANATIONS
# ============================================
# Install: pip install fastapi sqlalchemy asyncpg aioredis python-jose passlib python-multipart uvicorn

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload, joinedload
from sqlalchemy import String, ForeignKey, select, func
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Optional
from datetime import datetime, timedelta
from redis import asyncio as aioredis
import json
from passlib.context import CryptContext
from jose import JWTError, jwt

# ============================================
# 1. DATABASE MODELS (SQLAlchemy 2.0 Style)
# ============================================
# SQLAlchemy 2.0 uses "Mapped" type hints for better type safety and IDE support

class Base(DeclarativeBase):
    """
    Base class for all database models.
    DeclarativeBase is SQLAlchemy 2.0's new way to define models.
    All models inherit from this to get ORM capabilities.
    """
    pass

class User(Base):
    """
    User model - represents users table in database.
    Uses SQLAlchemy 2.0's Mapped[] type hints for type safety.
    """
    __tablename__ = "users"  # Explicit table name in database
    
    # Primary key column - auto-incrementing integer
    # mapped_column() replaces old Column() syntax
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Email field with constraints
    # String(255) = varchar(255) in database
    # unique=True creates a unique constraint
    # index=True creates a database index for faster lookups
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    
    # Username field
    username: Mapped[str] = mapped_column(String(50), unique=True)
    
    # Password is hashed, never store plain text!
    hashed_password: Mapped[str] = mapped_column(String(255))
    
    # Timestamp with default value
    # default=datetime.utcnow sets current time on insert
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    # RELATIONSHIPS - Define how models connect
    # back_populates creates bidirectional relationship
    # lazy="selectin" prevents N+1 by eager loading in separate query
    # List["Post"] means one user has many posts
    posts: Mapped[List["Post"]] = relationship(back_populates="author", lazy="selectin")
    comments: Mapped[List["Comment"]] = relationship(back_populates="user", lazy="selectin")

class Post(Base):
    """Post model - represents blog posts"""
    __tablename__ = "posts"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str]  # TEXT type in database (no length limit)
    
    # Foreign key - links to users.id
    # ForeignKey creates database constraint for referential integrity
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    # default=0 sets initial value on insert
    view_count: Mapped[int] = mapped_column(default=0)
    
    # relationship() links to User model (not just the ID)
    # back_populates must match the field name in User model
    author: Mapped["User"] = relationship(back_populates="posts")
    
    # One post has many comments
    comments: Mapped[List["Comment"]] = relationship(back_populates="post", lazy="selectin")

class Comment(Base):
    """Comment model - user comments on posts"""
    __tablename__ = "comments"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str]
    
    # Two foreign keys - comment belongs to both a post and a user
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    # Relationships to both Post and User
    post: Mapped["Post"] = relationship(back_populates="comments")
    user: Mapped["User"] = relationship(back_populates="comments")

# ============================================
# 2. PYDANTIC SCHEMAS (Request/Response Models)
# ============================================
# Pydantic models validate incoming data and serialize outgoing data
# Think of these as DRF Serializers

class UserBase(BaseModel):
    """
    Base user schema with common fields.
    EmailStr validates email format automatically.
    """
    email: EmailStr  # Validates email format
    username: str

class UserCreate(UserBase):
    """
    Schema for user registration.
    Inherits email and username from UserBase.
    """
    password: str  # Only needed for creation, not returned

class UserResponse(UserBase):
    """
    Schema for returning user data.
    Never return passwords to clients!
    """
    id: int
    created_at: datetime
    
    # ConfigDict replaces old Config class
    # from_attributes=True allows creating from ORM models (sqlalchemy objects)
    # This is like DRF's 'fields = '__all__'
    model_config = ConfigDict(from_attributes=True)

class CommentResponse(BaseModel):
    """Response schema for comments with user info"""
    id: int
    content: str
    user_id: int
    username: str  # Extra field from JOIN - not in database model
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PostBase(BaseModel):
    """Base post fields"""
    title: str
    content: str

class PostCreate(PostBase):
    """Schema for creating posts - just title and content"""
    pass

class PostResponse(PostBase):
    """
    Schema for returning posts in list view.
    Includes computed/joined fields that don't exist in model.
    """
    id: int
    author_id: int
    author_name: str  # Joined from User table
    created_at: datetime
    view_count: int
    comment_count: int  # Aggregated count from subquery
    model_config = ConfigDict(from_attributes=True)

class PostDetailResponse(PostResponse):
    """
    Extended schema for single post detail view.
    Includes nested comments list.
    """
    comments: List[CommentResponse] = []  # Nested list of comments

# ============================================
# 3. DATABASE SETUP & CONNECTION POOLING
# ============================================

# Connection string format: dialect+driver://user:password@host/database
# asyncpg is the fastest PostgreSQL driver for async Python
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/fastapi_blog"

# create_async_engine creates connection pool
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log all SQL queries (set False in production for performance)
    
    # CONNECTION POOLING - Reuses database connections instead of creating new ones
    pool_size=20,  # Keep 20 connections open at all times
    max_overflow=0,  # Don't create temporary connections beyond pool_size
    
    # pool_pre_ping=True tests connections before using them
    # Prevents errors from stale connections (connection might die while idle)
    pool_pre_ping=True,
)

# Session factory - creates new database sessions
# Like Django's connection, but for async
async_session_maker = async_sessionmaker(
    engine,  # Use our engine
    class_=AsyncSession,  # Return AsyncSession objects
    expire_on_commit=False  # Don't expire objects after commit (allows access after commit)
)

# ============================================
# 4. DEPENDENCY INJECTION (FastAPI's Superpower)
# ============================================
# Dependencies are reusable functions that FastAPI calls automatically
# Similar to Django middleware but more flexible

async def get_db():
    """
    Database session dependency.
    FastAPI calls this for every request that needs a database.
    
    Why use 'async with'?
    - Ensures session is always closed, even if error occurs
    - Prevents connection leaks
    
    Why 'yield' instead of 'return'?
    - Code after yield runs AFTER the request completes
    - Guarantees cleanup happens
    """
    async with async_session_maker() as session:
        try:
            yield session  # Request handler uses session here
        finally:
            await session.close()  # Always close, even on errors

# Global Redis connection (singleton pattern)
redis_client = None

async def get_redis():
    """
    Redis connection dependency.
    Reuses same connection instead of creating new one each time.
    
    Why global?
    - Redis connections are expensive to create
    - One connection can handle many concurrent requests
    """
    global redis_client
    if redis_client is None:
        # Create connection pool to Redis
        redis_client = await aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",  # Store strings, not bytes
            decode_responses=True  # Automatically decode responses to strings
        )
    return redis_client

# ============================================
# 5. AUTHENTICATION SETUP (JWT Tokens)
# ============================================
# JWT = JSON Web Token - secure way to identify users without sessions

# CRITICAL: Change this in production! Generate with: openssl rand -hex 32
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"  # Hashing algorithm for JWT
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Tokens expire after 30 minutes

# Password hashing context - uses bcrypt algorithm
# bcrypt is slow by design - prevents brute force attacks
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer tells FastAPI where to look for the token
# tokenUrl="token" means the login endpoint is at /token
# This creates the "lock" icon in Swagger docs
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    """
    Check if plain password matches hashed password.
    Never store plain passwords - always hash them!
    
    How bcrypt works:
    1. Takes plain password + salt
    2. Runs it through hash function many times (slow)
    3. Result is impossible to reverse
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password for storing in database"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT token with expiration.
    
    JWT contains:
    - Payload (user data)
    - Expiration time
    - Signature (proves it wasn't tampered with)
    
    Why JWT instead of sessions?
    - Stateless - server doesn't store anything
    - Scalable - works across multiple servers
    - Self-contained - all info in the token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})  # Add expiration to payload
    
    # encode() creates signed token - only SECRET_KEY holder can create valid tokens
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    token: str = Depends(oauth2_scheme),  # Get token from Authorization header
    db: AsyncSession = Depends(get_db)  # Get database session
) -> User:
    """
    Dependency that validates JWT token and returns current user.
    
    Use this in any endpoint that requires authentication:
    user = Depends(get_current_user)
    
    How it works:
    1. Extract token from Authorization header
    2. Verify signature and expiration
    3. Get user_id from token
    4. Query user from database
    5. Return user object
    """
    # Standard error for authentication failures
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},  # Tells client to send bearer token
    )
    
    try:
        # Decode and verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")  # "sub" = subject (user id)
        if user_id is None:
            raise credentials_exception
    except JWTError:
        # Token is invalid, expired, or tampered with
        raise credentials_exception
    
    # Fetch user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None:
        # Token is valid but user doesn't exist (deleted?)
        raise credentials_exception
    
    return user

# ============================================
# 6. CACHING HELPERS (Redis)
# ============================================
# Cache reduces database load by storing frequently accessed data in memory

async def get_cached_or_fetch(
    cache_key: str,  # Unique key for this data (e.g., "posts:list:0:10")
    fetch_func,  # Async function to call if cache miss
    redis: aioredis.Redis,
    ttl: int = 300  # Time to live in seconds (5 minutes default)
):
    """
    Generic caching pattern - check cache first, fetch if missing.
    
    Cache-aside pattern:
    1. Check Redis for cached data
    2. If found (cache hit), return it
    3. If not found (cache miss), fetch from database
    4. Store in Redis for next time
    5. Return data
    
    Why TTL?
    - Data eventually becomes stale
    - TTL ensures cache refreshes periodically
    - Balances performance vs freshness
    """
    # Try to get from cache
    cached = await redis.get(cache_key)
    if cached:
        # Cache hit! Parse JSON and return
        return json.loads(cached)
    
    # Cache miss - fetch from database
    data = await fetch_func()
    
    # Store in cache with expiration
    # default=str handles datetime serialization
    await redis.setex(cache_key, ttl, json.dumps(data, default=str))
    
    return data

async def invalidate_cache(pattern: str, redis: aioredis.Redis):
    """
    Delete cache keys matching a pattern.
    
    When to invalidate?
    - After creating/updating/deleting data
    - Ensures users don't see stale data
    
    Example: invalidate_cache("posts:*", redis)
    - Deletes all keys starting with "posts:"
    """
    keys = await redis.keys(pattern)  # Find matching keys
    if keys:
        await redis.delete(*keys)  # Delete all at once

# ============================================
# 7. FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Blog API",  # Shows in Swagger docs
    description="FastAPI example with caching, N+1 prevention, and async patterns",
    version="1.0.0"
)

# Lifecycle events - run code at startup/shutdown

@app.on_event("startup")
async def startup():
    """
    Run when server starts.
    Creates all database tables if they don't exist.
    
    In production:
    - Use Alembic for migrations instead
    - Don't create tables automatically
    """
    async with engine.begin() as conn:
        # run_sync runs synchronous code in async context
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("shutdown")
async def shutdown():
    """
    Run when server stops.
    Cleanly closes all connections.
    
    Why important?
    - Prevents "connection already closed" errors
    - Ensures data is flushed to disk
    """
    await engine.dispose()  # Close all database connections
    if redis_client:
        await redis_client.close()  # Close Redis connection

# ============================================
# 8. AUTHENTICATION ENDPOINTS
# ============================================

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user: UserCreate,  # Request body validated by Pydantic
    db: AsyncSession = Depends(get_db)  # Injected database session
):
    """
    Register a new user.
    
    Flow:
    1. Validate email format (Pydantic does this)
    2. Check if email already exists
    3. Hash the password
    4. Save to database
    5. Return user data (without password!)
    """
    # Check if user already exists
    # select() creates a SELECT query
    result = await db.execute(
        select(User).where(User.email == user.email)
    )
    
    # scalar_one_or_none() returns one object or None
    # Throws error if multiple found (shouldn't happen with unique constraint)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user instance
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=get_password_hash(user.password)  # Hash password!
    )
    
    # Add to session (like staging changes)
    db.add(db_user)
    
    # Commit to database (actually save)
    await db.commit()
    
    # Refresh to get auto-generated fields (id, created_at)
    await db.refresh(db_user)
    
    # FastAPI automatically converts to JSON using UserResponse schema
    return db_user

@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),  # Standard OAuth2 form
    db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint - returns JWT token.
    
    OAuth2PasswordRequestForm expects:
    - username (we use email)
    - password
    
    Returns:
    - access_token: JWT token
    - token_type: "bearer"
    """
    # Find user by email (OAuth2 calls it "username")
    result = await db.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()
    
    # Verify user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create JWT token with user id
    # Token payload: {"sub": "123"}
    # "sub" is standard JWT claim for subject (user id)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Return in OAuth2 format
    return {"access_token": access_token, "token_type": "bearer"}

# ============================================
# 9. POST ENDPOINTS (N+1 Prevention & Caching)
# ============================================

@app.get("/posts", response_model=List[PostResponse])
async def get_posts(
    skip: int = 0,  # Offset for pagination
    limit: int = 10,  # Items per page
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis)
):
    """
    Get paginated list of posts with author names and comment counts.
    
    OPTIMIZATIONS:
    1. Redis caching (5 min TTL) - reduces DB load
    2. JOIN for author name - prevents N+1 query
    3. Subquery for comment count - avoids separate query per post
    
    Without optimization:
    - 1 query for posts
    - 10 queries for authors (N+1 problem!)
    - 10 queries for comment counts
    - Total: 21 queries
    
    With optimization:
    - 1 query with joins
    - Total: 1 query! 🚀
    """
    cache_key = f"posts:list:{skip}:{limit}"  # Unique key per page
    
    async def fetch_posts():
        """
        Fetch posts with all data in ONE query.
        This function only runs on cache miss.
        """
        
        # STEP 1: Create subquery for comment counts
        # This is like a CTE (Common Table Expression)
        comment_count_subq = (
            select(
                Comment.post_id,  # Group by post
                func.count(Comment.id).label("count")  # Count comments
            )
            .group_by(Comment.post_id)
            .subquery()  # Convert to subquery for joining
        )
        
        # STEP 2: Main query with joins
        stmt = (
            select(
                Post,  # All post columns
                User.username.label("author_name"),  # Join author name
                func.coalesce(comment_count_subq.c.count, 0).label("comment_count")  # Join count
            )
            # JOIN posts and users - gets author name
            .join(User, Post.author_id == User.id)
            
            # LEFT JOIN comment counts - some posts might have 0 comments
            # outerjoin = LEFT JOIN (includes posts with no comments)
            .outerjoin(comment_count_subq, Post.id == comment_count_subq.c.post_id)
            
            # Pagination
            .offset(skip)  # Skip first N records
            .limit(limit)  # Return only N records
            
            # Sort newest first
            .order_by(Post.created_at.desc())
        )
        
        # Execute query
        result = await db.execute(stmt)
        
        # Build response dictionaries
        # result returns tuples: (post, author_name, comment_count)
        posts_data = []
        for post, author_name, comment_count in result:
            posts_data.append({
                "id": post.id,
                "title": post.title,
                "content": post.content,
                "author_id": post.author_id,
                "author_name": author_name,  # From JOIN
                "created_at": post.created_at,
                "view_count": post.view_count,
                "comment_count": comment_count  # From subquery
            })
        
        return posts_data
    
    # Use caching helper - checks Redis first
    return await get_cached_or_fetch(cache_key, fetch_posts, redis, ttl=300)

@app.get("/posts/{post_id}", response_model=PostDetailResponse)
async def get_post(
    post_id: int,  # From URL path
    background_tasks: BackgroundTasks,  # For async tasks
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis)
):
    """
    Get single post with all comments and user info.
    
    OPTIMIZATIONS:
    1. Eager loading with options() - loads relationships in 2 queries instead of N+1
    2. Background task for view count - doesn't slow down response
    3. Redis caching - fast subsequent requests
    
    How eager loading works:
    - joinedload: Uses JOIN (one query, but duplicates rows)
    - selectinload: Uses WHERE IN (two queries, no duplicates)
    
    Without eager loading:
    - 1 query for post
    - 1 query for author
    - N queries for comments (one per comment!)
    - N queries for comment users
    
    With eager loading:
    - 1 query for post + author (joinedload)
    - 1 query for all comments (selectinload)
    - 1 query for all comment users (selectinload)
    - Total: 3 queries regardless of N! 🎯
    """
    cache_key = f"post:detail:{post_id}"
    
    async def fetch_post():
        """Fetch post with all related data using eager loading"""
        
        stmt = (
            select(Post)
            .options(
                # joinedload uses SQL JOIN - good for one-to-one
                # Loads author in same query as post
                joinedload(Post.author),
                
                # selectinload uses WHERE IN - good for one-to-many
                # First selectinload gets all comments for this post
                # Second selectinload gets all users for those comments
                # Chaining: Post -> Comments -> Comment Users
                selectinload(Post.comments).selectinload(Comment.user)
            )
            .where(Post.id == post_id)
        )
        
        result = await db.execute(stmt)
        post = result.scalar_one_or_none()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Build nested response structure
        # All data is already loaded - no more queries!
        return {
            "id": post.id,
            "title": post.title,
            "content": post.content,
            "author_id": post.author_id,
            "author_name": post.author.username,  # No query - already loaded!
            "created_at": post.created_at,
            "view_count": post.view_count,
            "comment_count": len(post.comments),
            "comments": [
                {
                    "id": c.id,
                    "content": c.content,
                    "user_id": c.user_id,
                    "username": c.user.username,  # No query - already loaded!
                    "created_at": c.created_at
                }
                for c in post.comments
            ]
        }
    
    post_data = await get_cached_or_fetch(cache_key, fetch_post, redis, ttl=300)
    
    # Increment view count in background
    # add_task runs after response is sent - doesn't slow down user
    background_tasks.add_task(increment_view_count, post_id, db, redis)
    
    return post_data

async def increment_view_count(post_id: int, db: AsyncSession, redis: aioredis.Redis):
    """
    Background task to increment view count.
    
    Why background task?
    - Doesn't slow down response to user
    - View count isn't critical - can happen async
    - User gets fast response
    
    Why new session?
    - Original session is closed after response sent
    - Need new session for background work
    """
    async with async_session_maker() as session:
        result = await session.execute(select(Post).where(Post.id == post_id))
        post = result.scalar_one_or_none()
        if post:
            post.view_count += 1
            await session.commit()
            
            # Invalidate cache so next request gets updated count
            await invalidate_cache(f"post:detail:{post_id}", redis)

@app.post("/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post: PostCreate,  # Request body
    current_user: User = Depends(get_current_user),  # Requires authentication!
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis)
):
    """
    Create a new post.
    
    Requires authentication - only logged in users can post.
    Depends(get_current_user) validates JWT and returns user.
    """
    # Create post with current user as author
    db_post = Post(
        title=post.title,
        content=post.content,
        author_id=current_user.id  # From JWT token
    )
    
    db.add(db_post)
    await db.commit()
    await db.refresh(db_post)  # Get auto-generated id
    
    # Invalidate list cache - new post should appear
    # Pattern "posts:list:*" matches all pagination pages
    await invalidate_cache("posts:list:*", redis)
    
    return {
        "id": db_post.id,
        "title": db_post.title,
        "content": db_post.content,
        "author_id": db_post.author_id,
        "author_name": current_user.username,
        "created_at": db_post.created_at,
        "view_count": db_post.view_count,
        "comment_count": 0
    }

@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: int,
    current_user: User = Depends(get_current_user),  # Must be logged in
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis)
):
    """
    Delete a post.
    
    Authorization check: only author can delete their own posts.
    """
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check if current user is the author
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.delete(post)
    await db.commit()
    
    # Invalidate all related caches
    await invalidate_cache("posts:*", redis)  # List cache
    await invalidate_cache(f"post:detail:{post_id}", redis)  # Detail cache
    
    # 204 No Content - successful deletion returns nothing
    return None

# ============================================
# 10. COMMENT ENDPOINTS
# ============================================

@app.post("/posts/{post_id}/comments", status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: int,
    content: str,  # Query parameter (could also be in request body)
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis)
):
    """Add a comment to a post"""
    
    # Verify post exists
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    comment = Comment(
        content=content,
        post_id=post_id,
        user_id=current_user.id
    )
    db.add(comment)
    await db.commit()
    
    # Invalidate caches - comment count changed
    await invalidate_cache(f"post:detail:{post_id}", redis)
    await invalidate_cache("posts:list:*", redis)  # List shows comment counts
    
    return {"message": "Comment created", "comment_id": comment.id}

# ============================================
# 11. STATS ENDPOINT (Parallel Queries)
# ============================================

@app.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis