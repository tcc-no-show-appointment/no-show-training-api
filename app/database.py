from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
from app.config import config

# Construct connection string
if config.DB_CONNECTION_STRING:
    # Use provided connection string with encryption settings
    params = urllib.parse.quote_plus(config.DB_CONNECTION_STRING)
else:
    # Fall back to constructing from individual components
    params = urllib.parse.quote_plus(
        f"DRIVER={config.DB_DRIVER};"
        f"SERVER={config.DB_SERVER};"
        f"DATABASE={config.DB_NAME};"
        f"UID={config.DB_USER};"
        f"PWD={config.DB_PASSWORD}"
    )

# Use mssql+pyodbc
SQLALCHEMY_DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"

# Create engine
# fast_executemany=True is recommended for Azure SQL
# Added pool settings and connection arguments for better reliability
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_size=5,         # Maximum number of connections to keep in pool
    max_overflow=10,     # Maximum overflow connections
    connect_args={
        "timeout": 30,   # Connection timeout in seconds
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
