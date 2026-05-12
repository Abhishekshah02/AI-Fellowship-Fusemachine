from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PaymentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customerNumber: int
    checkNumber: str
    paymentDate: date
    amount: Decimal


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    orderNumber: int
    orderDate: date
    requiredDate: date
    shippedDate: Optional[date] = None
    status: str
    comments: Optional[str] = None
    customerNumber: int


class CustomerBase(BaseModel):
    customerName: str = Field(..., max_length=50)
    contactLastName: str = Field(..., max_length=50)
    contactFirstName: str = Field(..., max_length=50)
    phone: str = Field(..., max_length=50)
    addressLine1: str = Field(..., max_length=50)
    addressLine2: Optional[str] = Field(None, max_length=50)
    city: str = Field(..., max_length=50)
    state: Optional[str] = Field(None, max_length=50)
    postalCode: Optional[str] = Field(None, max_length=15)
    country: str = Field(..., max_length=50)
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[Decimal] = None


class CustomerCreate(CustomerBase):
    customerNumber: int


class CustomerUpdate(BaseModel):
    customerName: Optional[str] = Field(None, max_length=50)
    contactLastName: Optional[str] = Field(None, max_length=50)
    contactFirstName: Optional[str] = Field(None, max_length=50)
    phone: Optional[str] = Field(None, max_length=50)
    addressLine1: Optional[str] = Field(None, max_length=50)
    addressLine2: Optional[str] = Field(None, max_length=50)
    city: Optional[str] = Field(None, max_length=50)
    state: Optional[str] = Field(None, max_length=50)
    postalCode: Optional[str] = Field(None, max_length=15)
    country: Optional[str] = Field(None, max_length=50)
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[Decimal] = None


class CustomerOut(CustomerBase):
    model_config = ConfigDict(from_attributes=True)
    customerNumber: int
