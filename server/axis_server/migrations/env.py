from alembic import context
from sqlalchemy import create_engine

url = context.config.get_main_option("sqlalchemy.url")
engine = create_engine(url)
with engine.connect() as conn:
    context.configure(connection=conn, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
