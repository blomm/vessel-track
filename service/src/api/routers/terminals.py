from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from src.database.connection import get_db
from src.database.models import Terminal
from src.schemas.terminal import TerminalResponse, TerminalCreate

router = APIRouter()


@router.get("", response_model=List[TerminalResponse])
async def list_terminals(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    region: str = Query(None, description="Filter by region"),
    terminal_type: str = Query(None, description="Filter by terminal type (export/import)"),
    db: AsyncSession = Depends(get_db)
):
    """List all terminals with optional filtering"""
    query = select(Terminal)

    if region:
        query = query.where(Terminal.region == region)
    if terminal_type:
        query = query.where(Terminal.terminal_type == terminal_type)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    terminals = result.scalars().all()
    return terminals


@router.get("/{terminal_id}", response_model=TerminalResponse)
async def get_terminal(
    terminal_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific terminal"""
    result = await db.execute(
        select(Terminal).where(Terminal.id == terminal_id)
    )
    terminal = result.scalar_one_or_none()

    if not terminal:
        raise HTTPException(status_code=404, detail=f"Terminal {terminal_id} not found")

    return terminal


@router.post("", response_model=TerminalResponse)
async def create_terminal(
    terminal: TerminalCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new terminal"""
    db_terminal = Terminal(**terminal.model_dump())
    db.add(db_terminal)
    await db.commit()
    await db.refresh(db_terminal)
    return db_terminal
