from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from . import crud, schemas
from .database import get_db
from .logger import logger


router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("/", response_model=List[schemas.CustomerOut])
def list_customers(
    skip: int = Query(0, ge=0, description="Rows to skip for pagination"),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
    db: Session = Depends(get_db),
):
    logger.info(f"GET /customers (skip={skip}, limit={limit})")
    customers = crud.get_customers(db, skip=skip, limit=limit)
    logger.info(f"GET /customers -> {len(customers)} rows")
    return customers


@router.get("/{customer_number}", response_model=schemas.CustomerOut)
def read_customer(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"GET /customers/{customer_number}")
    customer = crud.get_customer(db, customer_number)
    if customer is None:
        logger.warning(f"Customer not found: {customer_number}")
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_number} not found",
        )
    return customer


@router.post(
    "/",
    response_model=schemas.CustomerOut,
    status_code=status.HTTP_201_CREATED,
)
def create_customer(
    payload: schemas.CustomerCreate,
    db: Session = Depends(get_db),
):
    logger.info(f"POST /customers (customerNumber={payload.customerNumber})")
    if crud.get_customer(db, payload.customerNumber) is not None:
        logger.warning(f"Conflict: customer {payload.customerNumber} already exists")
        raise HTTPException(
            status_code=409,
            detail=f"Customer {payload.customerNumber} already exists",
        )
    created = crud.create_customer(db, payload)
    if created is None:
        raise HTTPException(
            status_code=400,
            detail="Could not create customer (integrity error)",
        )
    return created


@router.put("/{customer_number}", response_model=schemas.CustomerOut)
def update_customer(
    customer_number: int,
    payload: schemas.CustomerUpdate,
    db: Session = Depends(get_db),
):
    logger.info(f"PUT /customers/{customer_number}")
    updated = crud.update_customer(db, customer_number, payload)
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_number} not found",
        )
    return updated


@router.delete("/{customer_number}", status_code=status.HTTP_204_NO_CONTENT)
def delete_customer(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"DELETE /customers/{customer_number}")
    try:
        deleted = crud.delete_customer(db, customer_number)
    except IntegrityError:
        db.rollback()
        logger.warning(
            f"Cannot delete customer {customer_number}: has related orders/payments"
        )
        raise HTTPException(
            status_code=409,
            detail="Cannot delete: customer has related orders or payments",
        )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_number} not found",
        )
    return None


@router.get("/{customer_number}/orders", response_model=List[schemas.OrderOut])
def list_customer_orders(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"GET /customers/{customer_number}/orders")
    if crud.get_customer(db, customer_number) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_number} not found",
        )
    return crud.get_customer_orders(db, customer_number)


@router.get("/{customer_number}/payments", response_model=List[schemas.PaymentOut])
def list_customer_payments(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"GET /customers/{customer_number}/payments")
    if crud.get_customer(db, customer_number) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_number} not found",
        )
    return crud.get_customer_payments(db, customer_number)
