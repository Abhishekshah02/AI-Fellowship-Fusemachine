from sqlalchemy import Column, Date, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Customer(Base):
    __tablename__ = "customers"

    customerNumber = Column(Integer, primary_key=True, index=True)
    customerName = Column(String(50), nullable=False)
    contactLastName = Column(String(50), nullable=False)
    contactFirstName = Column(String(50), nullable=False)
    phone = Column(String(50), nullable=False)
    addressLine1 = Column(String(50), nullable=False)
    addressLine2 = Column(String(50), nullable=True)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=True)
    postalCode = Column(String(15), nullable=True)
    country = Column(String(50), nullable=False)
    salesRepEmployeeNumber = Column(Integer, nullable=True)
    creditLimit = Column(Numeric(10, 2), nullable=True)

    orders = relationship("Order", back_populates="customer")
    payments = relationship("Payment", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    orderNumber = Column(Integer, primary_key=True, index=True)
    orderDate = Column(Date, nullable=False)
    requiredDate = Column(Date, nullable=False)
    shippedDate = Column(Date, nullable=True)
    status = Column(String(15), nullable=False)
    comments = Column(Text, nullable=True)
    customerNumber = Column(
        Integer,
        ForeignKey("customers.customerNumber"),
        nullable=False,
    )

    customer = relationship("Customer", back_populates="orders")


class Payment(Base):
    __tablename__ = "payments"

    customerNumber = Column(
        Integer,
        ForeignKey("customers.customerNumber"),
        primary_key=True,
    )
    checkNumber = Column(String(50), primary_key=True)
    paymentDate = Column(Date, nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)

    customer = relationship("Customer", back_populates="payments")
