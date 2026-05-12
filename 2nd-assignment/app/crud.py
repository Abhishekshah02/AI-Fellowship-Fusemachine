from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from . import models, schemas
from .logger import logger


# ---------------------------------------------------------------------------
# Customer CRUD
# ---------------------------------------------------------------------------

def get_customers(db: Session, skip: int = 0, limit: int = 100) -> List[models.Customer]:
    logger.info(f"CRUD: list customers (skip={skip}, limit={limit})")
    return (
        db.query(models.Customer)
        .order_by(models.Customer.customerNumber)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_customer(db: Session, customer_number: int) -> Optional[models.Customer]:
    logger.info(f"CRUD: fetch customer {customer_number}")
    return (
        db.query(models.Customer)
        .filter(models.Customer.customerNumber == customer_number)
        .first()
    )


def create_customer(db: Session, payload: schemas.CustomerCreate) -> Optional[models.Customer]:
    logger.info(f"CRUD: create customer {payload.customerNumber}")
    new_customer = models.Customer(**payload.model_dump())
    db.add(new_customer)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        logger.error(f"CRUD: integrity error creating customer: {exc.orig}")
        return None
    db.refresh(new_customer)
    logger.info(f"CRUD: customer {new_customer.customerNumber} created")
    return new_customer


def update_customer(
    db: Session,
    customer_number: int,
    payload: schemas.CustomerUpdate,
) -> Optional[models.Customer]:
    logger.info(f"CRUD: update customer {customer_number}")
    customer = get_customer(db, customer_number)
    if customer is None:
        logger.warning(f"CRUD: customer {customer_number} not found for update")
        return None

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(customer, key, value)

    db.commit()
    db.refresh(customer)
    logger.info(f"CRUD: customer {customer_number} updated ({len(update_data)} fields)")
    return customer


def delete_customer(db: Session, customer_number: int) -> bool:
    logger.info(f"CRUD: delete customer {customer_number}")
    customer = get_customer(db, customer_number)
    if customer is None:
        logger.warning(f"CRUD: customer {customer_number} not found for delete")
        return False

    db.delete(customer)
    db.commit()
    logger.info(f"CRUD: customer {customer_number} deleted")
    return True


def get_customer_orders(db: Session, customer_number: int) -> List[models.Order]:
    logger.info(f"CRUD: fetch orders for customer {customer_number}")
    return (
        db.query(models.Order)
        .filter(models.Order.customerNumber == customer_number)
        .order_by(models.Order.orderDate.desc())
        .all()
    )


def get_customer_payments(db: Session, customer_number: int) -> List[models.Payment]:
    logger.info(f"CRUD: fetch payments for customer {customer_number}")
    return (
        db.query(models.Payment)
        .filter(models.Payment.customerNumber == customer_number)
        .order_by(models.Payment.paymentDate.desc())
        .all()
    )


# ---------------------------------------------------------------------------
# Task 3: count functions (one per table)
# Returns 0 instead of crashing if the table is empty.
# ---------------------------------------------------------------------------

def _count_table(db: Session, table: str) -> int:
    logger.info(f"CRUD: count {table} - start")
    result = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    n = result or 0
    logger.info(f"CRUD: count {table} = {n}")
    return n


def count_customers(db: Session) -> int:
    return _count_table(db, "customers")


def count_orders(db: Session) -> int:
    return _count_table(db, "orders")


def count_products(db: Session) -> int:
    return _count_table(db, "products")


def count_employees(db: Session) -> int:
    return _count_table(db, "employees")


def count_offices(db: Session) -> int:
    return _count_table(db, "offices")


def count_payments(db: Session) -> int:
    return _count_table(db, "payments")


def count_orderdetails(db: Session) -> int:
    return _count_table(db, "orderdetails")


def count_productlines(db: Session) -> int:
    return _count_table(db, "productlines")
