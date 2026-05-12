import asyncio
import time
from typing import Callable

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from . import crud
from .database import SessionLocal, get_db
from .logger import logger


router = APIRouter(tags=["counts"])


# ---------------------------------------------------------------------------
# 8 individual count endpoints (modular design)
# ---------------------------------------------------------------------------

@router.get("/customers/count")
def customers_count(db: Session = Depends(get_db)):
    logger.info("GET /customers/count")
    n = crud.count_customers(db)
    logger.info(f"GET /customers/count -> {n}")
    return {"customers": n}


@router.get("/orders/count")
def orders_count(db: Session = Depends(get_db)):
    logger.info("GET /orders/count")
    n = crud.count_orders(db)
    logger.info(f"GET /orders/count -> {n}")
    return {"orders": n}


@router.get("/products/count")
def products_count(db: Session = Depends(get_db)):
    logger.info("GET /products/count")
    n = crud.count_products(db)
    logger.info(f"GET /products/count -> {n}")
    return {"products": n}


@router.get("/employees/count")
def employees_count(db: Session = Depends(get_db)):
    logger.info("GET /employees/count")
    n = crud.count_employees(db)
    logger.info(f"GET /employees/count -> {n}")
    return {"employees": n}


@router.get("/offices/count")
def offices_count(db: Session = Depends(get_db)):
    logger.info("GET /offices/count")
    n = crud.count_offices(db)
    logger.info(f"GET /offices/count -> {n}")
    return {"offices": n}


@router.get("/payments/count")
def payments_count(db: Session = Depends(get_db)):
    logger.info("GET /payments/count")
    n = crud.count_payments(db)
    logger.info(f"GET /payments/count -> {n}")
    return {"payments": n}


@router.get("/orderdetails/count")
def orderdetails_count(db: Session = Depends(get_db)):
    logger.info("GET /orderdetails/count")
    n = crud.count_orderdetails(db)
    logger.info(f"GET /orderdetails/count -> {n}")
    return {"orderdetails": n}


@router.get("/productlines/count")
def productlines_count(db: Session = Depends(get_db)):
    logger.info("GET /productlines/count")
    n = crud.count_productlines(db)
    logger.info(f"GET /productlines/count -> {n}")
    return {"productlines": n}


# ---------------------------------------------------------------------------
# Aggregated endpoint: runs all 8 counts concurrently via asyncio.gather()
# ---------------------------------------------------------------------------

async def _run_count(count_fn: Callable[[Session], int]) -> int:
    """Run a sync count function in a worker thread.

    Each thread gets its own DB session because SQLAlchemy sessions are
    NOT thread-safe. asyncio.to_thread() lets gather() fan multiple
    blocking queries across threads in real parallel.
    """
    def work():
        db = SessionLocal()
        try:
            return count_fn(db)
        finally:
            db.close()
    return await asyncio.to_thread(work)


@router.get("/overall_counts")
async def overall_counts():
    logger.info("GET /overall_counts: scheduling 8 concurrent count tasks")
    start = time.perf_counter()

    customers, orders, products, employees, offices, payments, orderdetails, productlines = (
        await asyncio.gather(
            _run_count(crud.count_customers),
            _run_count(crud.count_orders),
            _run_count(crud.count_products),
            _run_count(crud.count_employees),
            _run_count(crud.count_offices),
            _run_count(crud.count_payments),
            _run_count(crud.count_orderdetails),
            _run_count(crud.count_productlines),
        )
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"GET /overall_counts: gather() completed in {elapsed_ms:.2f} ms"
    )

    return {
        "customers": customers,
        "orders": orders,
        "products": products,
        "employees": employees,
        "offices": offices,
        "payments": payments,
        "orderdetails": orderdetails,
        "productlines": productlines,
        "_elapsed_ms": round(elapsed_ms, 2),
    }
