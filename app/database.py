from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
from app.config import config

# Construct connection string
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
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
