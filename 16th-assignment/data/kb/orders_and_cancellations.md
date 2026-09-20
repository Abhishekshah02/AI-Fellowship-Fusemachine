# Orders and Cancellations

## Order lifecycle
`PENDING` -> `PROCESSING` -> `SHIPPED` -> `DELIVERED`, with `CANCELLED` reachable
from `PENDING` and `PROCESSING` only.

## Cancelling an order
Customers can cancel from **My Orders** for free while the order is `PENDING` or
`PROCESSING`. After `SHIPPED`, the order cannot be cancelled; it must be refused at
the door or returned under the 30-day return policy.

## Modifying an order
Item quantity and size can be changed while `PENDING`. After that the order must be
cancelled and re-placed; there is no charge for doing so.

## Cancellation refunds
A cancelled order releases the payment authorisation immediately. Card issuers may
still show the pending charge for up to 5 business days; this is not a real charge.

## Failed orders
An order that fails payment authorisation stays `PENDING` for 60 minutes and is then
cancelled automatically. Stock is not held after that point.
