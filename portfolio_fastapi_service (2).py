
"""
Notes API (FastAPI)
-------------------
A tiny service for creating and listing notes using SQLite. It keeps things simple:
a title, a body, and optional tags. Good for demos or as a starting point for a
slightly larger project.

Start:
    uvicorn portfolio_fastapi_service:app --reload

Dependencies:
    fastapi, uvicorn, sqlalchemy, pydantic
"""
from __future__ import annotations

import logging
from typing import Optional, List

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, constr
from sqlalchemy import Column, Integer, String, create_engine, select, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("notes")

# Database
DB_URL = "sqlite:///./notes.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class NoteORM(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(150), nullable=False)
    body = Column(Text, nullable=False)
    tags = Column(String(200), default="")

Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Schemas
TagStr = constr(strip_whitespace=True, min_length=1, max_length=30)

class NoteIn(BaseModel):
    title: constr(strip_whitespace=True, min_length=1, max_length=150)
    body: constr(strip_whitespace=True, min_length=1, max_length=10_000)
    tags: list[TagStr] = Field(default_factory=list)

class NoteOut(NoteIn):
    id: int
    class Config:
        from_attributes = True

# App
app = FastAPI(title="Notes API", version="1.0.0")

@app.get("/health")
def health() -> dict:
    return {"ok": True}

@app.post("/notes", response_model=NoteOut, status_code=status.HTTP_201_CREATED)
def create_note(payload: NoteIn, db: Session = Depends(get_db)) -> NoteOut:
    note = NoteORM(title=payload.title, body=payload.body, tags=",".join(payload.tags))
    db.add(note)
    db.commit()
    db.refresh(note)
    log.info("created note %s", note.id)
    return NoteOut.model_validate({**payload.model_dump(), "id": note.id})

@app.get("/notes/{note_id}", response_model=NoteOut)
def get_note(note_id: int, db: Session = Depends(get_db)) -> NoteOut:
    note = db.get(NoteORM, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteOut.model_validate({
        "id": note.id,
        "title": note.title,
        "body": note.body,
        "tags": [t for t in note.tags.split(",") if t],
    })

@app.get("/notes", response_model=List[NoteOut])
def list_notes(
    q: Optional[str] = Query(None, description="Search in title/body"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[NoteOut]:
    items = db.execute(select(NoteORM)).scalars().all()

    def matches(n: NoteORM) -> bool:
        if q and (q.lower() not in n.title.lower() and q.lower() not in n.body.lower()):
            return False
        if tag and tag not in (n.tags.split(",") if n.tags else []):
            return False
        return True

    filtered = [n for n in items if matches(n)]
    window = filtered[offset: offset + limit]
    return [NoteOut.model_validate({
        "id": n.id, "title": n.title, "body": n.body,
        "tags": [t for t in n.tags.split(",") if t],
    }) for n in window]

@app.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: int, db: Session = Depends(get_db)) -> None:
    note = db.get(NoteORM, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(note)
    db.commit()
    log.info("deleted note %s", note_id)
    return None
