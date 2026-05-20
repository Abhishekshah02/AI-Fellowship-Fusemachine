"""Frozen schema description used inside LLM prompts.

We keep this hand-curated rather than introspecting the live DB at request time:
it keeps prompts deterministic, cheap, and the LLM can reason about column
semantics (e.g. that `MSRP` is the retail price while `buyPrice` is wholesale).
"""

SCHEMA_DESCRIPTION = """\
PostgreSQL database: classicmodels (8 tables).
All identifiers are quoted with double quotes and are case-sensitive (camelCase).

Table: productlines
  "productLine"        VARCHAR(50) PRIMARY KEY    -- e.g. 'Classic Cars', 'Motorcycles'
  "textDescription"    VARCHAR(4000)
  "htmlDescription"    TEXT
  "image"              BYTEA

Table: products
  "productCode"        VARCHAR(15) PRIMARY KEY
  "productName"        VARCHAR(70)
  "productLine"        VARCHAR(50)  REFERENCES productlines("productLine")
  "productScale"       VARCHAR(10)
  "productVendor"      VARCHAR(50)
  "productDescription" TEXT
  "quantityInStock"    INTEGER
  "buyPrice"           NUMERIC(10,2)  -- wholesale cost
  "MSRP"               NUMERIC(10,2)  -- retail price

Table: offices
  "officeCode"   VARCHAR(10) PRIMARY KEY
  "city"         VARCHAR(50)
  "phone"        VARCHAR(50)
  "addressLine1" VARCHAR(50)
  "addressLine2" VARCHAR(50)
  "state"        VARCHAR(50)
  "country"      VARCHAR(50)
  "postalCode"   VARCHAR(15)
  "territory"    VARCHAR(10)

Table: employees
  "employeeNumber" INTEGER PRIMARY KEY
  "lastName"       VARCHAR(50)
  "firstName"      VARCHAR(50)
  "extension"      VARCHAR(10)
  "email"          VARCHAR(100)
  "officeCode"     VARCHAR(10)  REFERENCES offices("officeCode")
  "reportsTo"      INTEGER      REFERENCES employees("employeeNumber")  -- manager
  "jobTitle"       VARCHAR(50)

Table: customers
  "customerNumber"         INTEGER PRIMARY KEY
  "customerName"           VARCHAR(50)
  "contactLastName"        VARCHAR(50)
  "contactFirstName"       VARCHAR(50)
  "phone"                  VARCHAR(50)
  "addressLine1"           VARCHAR(50)
  "addressLine2"           VARCHAR(50)
  "city"                   VARCHAR(50)
  "state"                  VARCHAR(50)
  "postalCode"             VARCHAR(15)
  "country"                VARCHAR(50)
  "salesRepEmployeeNumber" INTEGER  REFERENCES employees("employeeNumber")
  "creditLimit"            NUMERIC(10,2)

Table: payments  (composite PK: customerNumber + checkNumber)
  "customerNumber" INTEGER  REFERENCES customers("customerNumber")
  "checkNumber"    VARCHAR(50)
  "paymentDate"    DATE
  "amount"         NUMERIC(10,2)

Table: orders
  "orderNumber"    INTEGER PRIMARY KEY
  "orderDate"      DATE
  "requiredDate"   DATE
  "shippedDate"    DATE        -- NULL while in-process / cancelled
  "status"         VARCHAR(15) -- 'Shipped','Resolved','Cancelled','On Hold','Disputed','In Process'
  "comments"       TEXT
  "customerNumber" INTEGER REFERENCES customers("customerNumber")

Table: orderdetails  (composite PK: orderNumber + productCode)
  "orderNumber"     INTEGER     REFERENCES orders("orderNumber")
  "productCode"     VARCHAR(15) REFERENCES products("productCode")
  "quantityOrdered" INTEGER
  "priceEach"       NUMERIC(10,2)
  "orderLineNumber" SMALLINT

Common joins:
  orders        o JOIN customers     c USING ("customerNumber")
  payments      p JOIN customers     c USING ("customerNumber")
  orderdetails od JOIN orders        o USING ("orderNumber")
  orderdetails od JOIN products      p USING ("productCode")
  products      p JOIN productlines pl USING ("productLine")
  employees     e JOIN offices       o USING ("officeCode")
  employees     e JOIN employees     m ON e."reportsTo" = m."employeeNumber"
  customers     c JOIN employees    sr ON c."salesRepEmployeeNumber" = sr."employeeNumber"
"""

# Used by the rule-based fallback decomposer to resolve table/column mentions.
TABLE_COLUMNS = {
    "productlines": ["productLine", "textDescription", "htmlDescription", "image"],
    "products": [
        "productCode", "productName", "productLine", "productScale", "productVendor",
        "productDescription", "quantityInStock", "buyPrice", "MSRP",
    ],
    "offices": [
        "officeCode", "city", "phone", "addressLine1", "addressLine2",
        "state", "country", "postalCode", "territory",
    ],
    "employees": [
        "employeeNumber", "lastName", "firstName", "extension", "email",
        "officeCode", "reportsTo", "jobTitle",
    ],
    "customers": [
        "customerNumber", "customerName", "contactLastName", "contactFirstName",
        "phone", "addressLine1", "addressLine2", "city", "state", "postalCode",
        "country", "salesRepEmployeeNumber", "creditLimit",
    ],
    "payments": ["customerNumber", "checkNumber", "paymentDate", "amount"],
    "orders": [
        "orderNumber", "orderDate", "requiredDate", "shippedDate",
        "status", "comments", "customerNumber",
    ],
    "orderdetails": [
        "orderNumber", "productCode", "quantityOrdered", "priceEach", "orderLineNumber",
    ],
}
