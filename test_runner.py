import asyncio

from sqlmodel import Session, select

from app.database import engine
from app.models import Chunk, QAPair
from app.worker.runner import run_job
from app.database import create_db_and_tables

create_db_and_tables()

async def main():

    # Find a job that already has chunks
    with Session(engine) as s:
        chunk = s.exec(select(Chunk)).first()

    if chunk is None:
        print("No chunks in database.")
        return

    print(f"Running job {chunk.job_id}")

    await run_job(chunk.job_id)

    with Session(engine) as s:
        qa = s.exec(
            select(QAPair)
            .where(QAPair.job_id == chunk.job_id)
        ).first()

    if qa is None:
        print("❌ No QA pair generated.")
        return

    print("\nGenerated QA\n")

    print("Question:")
    print(qa.question)

    print("\nAnswer:")
    print(qa.answer)

    print("\nVerified:", qa.quote_verified)


if __name__ == "__main__":
    asyncio.run(main())