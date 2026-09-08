"""
Database setup for the minimal HRMS.

Uses SQLite via SQLAlchemy. The whole database is a single file (hrms.db)
created next to this project. Because we go through SQLAlchemy's ORM layer,
switching to PostgreSQL later is a one-line change to DATABASE_URL.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite file lives at the project root. check_same_thread=False is required
# because FastAPI may touch the connection from different threads.
DATABASE_URL = "sqlite:///./hrms.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# All ORM models inherit from this Base.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency. Yields a database session and guarantees it is
    closed after the request, even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
