"""
Database connection management.
Async SQLAlchemy with PostgreSQL.
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core import config

engine = create_async_engine(
    config.DATABASE_URL,
    echo=config.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """Dependency: yields an async database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables and auto-add new columns to existing tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Auto-add new columns that create_all can't add to existing tables
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'ai_comparisons' AND column_name = 'object_category'
                ) THEN
                    ALTER TABLE ai_comparisons ADD COLUMN object_category VARCHAR(50) NULL;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'ai_comparisons' AND column_name = 'dino_similarity'
                ) THEN
                    ALTER TABLE ai_comparisons ADD COLUMN dino_similarity FLOAT NULL;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'ai_comparisons' AND column_name = 'dino_patch_changed_fraction'
                ) THEN
                    ALTER TABLE ai_comparisons ADD COLUMN dino_patch_changed_fraction FLOAT NULL;
                    ALTER TABLE ai_comparisons ADD COLUMN dino_patch_max_region FLOAT NULL;
                    ALTER TABLE ai_comparisons ADD COLUMN dino_patch_mean FLOAT NULL;
                END IF;
            END $$;
        """))


async def close_db():
    """Dispose engine on shutdown."""
    await engine.dispose()
