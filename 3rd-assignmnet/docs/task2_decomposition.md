# Task 2 — Query Understanding (Decomposition)

For every benchmark question, the structured breakdown the agent must
produce **before** generating SQL. Each item lists:

* **Intent** — what is being asked, as a verb phrase
* **Tables** — the tables required
* **Columns** — the columns projected or aggregated on
* **Filters** — WHERE conditions (empty if none implied)
* **Joins** — join conditions, written as `tableA.col = tableB.col`
* **Aggregations / Group by** — listed when the question implies them

This is exactly the JSON shape the agent's `decompose()` step produces at
runtime — see `app/decomposer.py` and `prompts/decompose.txt`.

---

## Simple SELECTs (Q01–Q20)

### Q01 — List all products
- **Intent:** Retrieve all product rows
- **Tables:** products
- **Columns:** all columns of products
- **Filters:** none
- **Joins:** none

### Q02 — Get all customers
- **Intent:** Retrieve all customer rows
- **Tables:** customers
- **Columns:** all columns of customers
- **Filters:** none
- **Joins:** none

### Q03 — Show all orders
- **Intent:** Retrieve all order rows
- **Tables:** orders
- **Columns:** all columns of orders
- **Filters:** none
- **Joins:** none

### Q04 — List all employees
- **Intent:** Retrieve all employee rows
- **Tables:** employees
- **Columns:** all columns of employees
- **Filters:** none
- **Joins:** none

### Q05 — Get all offices
- **Intent:** Retrieve all office rows
- **Tables:** offices
- **Columns:** all columns of offices
- **Filters:** none
- **Joins:** none

### Q06 — Show all product lines
- **Intent:** Retrieve all product-line rows
- **Tables:** productlines
- **Columns:** all columns of productlines
- **Filters:** none
- **Joins:** none

### Q07 — List all payments
- **Intent:** Retrieve all payment rows
- **Tables:** payments
- **Columns:** all columns of payments
- **Filters:** none
- **Joins:** none

### Q08 — Get product names and prices
- **Intent:** Retrieve product names with their retail price
- **Tables:** products
- **Columns:** products.productName, products.MSRP
- **Filters:** none
- **Joins:** none
- **Note:** "price" is ambiguous; we choose MSRP (retail) over buyPrice (wholesale)

### Q09 — Get customer names and cities
- **Intent:** Retrieve customer names with their cities
- **Tables:** customers
- **Columns:** customers.customerName, customers.city
- **Filters:** none
- **Joins:** none

### Q10 — List employee first and last names
- **Intent:** Retrieve employee names
- **Tables:** employees
- **Columns:** employees.firstName, employees.lastName
- **Filters:** none
- **Joins:** none

### Q11 — Get all order dates
- **Intent:** Retrieve order placement dates
- **Tables:** orders
- **Columns:** orders.orderDate
- **Filters:** none
- **Joins:** none

### Q12 — Show product vendor list
- **Intent:** List distinct vendors that supply products
- **Tables:** products
- **Columns:** products.productVendor
- **Filters:** none (DISTINCT implied)
- **Joins:** none

### Q13 — Get all product codes
- **Intent:** Retrieve product identifiers
- **Tables:** products
- **Columns:** products.productCode
- **Filters:** none
- **Joins:** none

### Q14 — List all countries from offices
- **Intent:** List distinct countries where the company has offices
- **Tables:** offices
- **Columns:** offices.country
- **Filters:** none (DISTINCT implied)
- **Joins:** none

### Q15 — Show all order statuses
- **Intent:** List the distinct order status values that exist
- **Tables:** orders
- **Columns:** orders.status
- **Filters:** none (DISTINCT implied)
- **Joins:** none

### Q16 — Get all payment amounts
- **Intent:** Retrieve every payment amount
- **Tables:** payments
- **Columns:** payments.amount
- **Filters:** none
- **Joins:** none

### Q17 — List all job titles
- **Intent:** List the distinct job titles employees hold
- **Tables:** employees
- **Columns:** employees.jobTitle
- **Filters:** none (DISTINCT implied)
- **Joins:** none

### Q18 — Get customer phone numbers
- **Intent:** Retrieve customer names with their phone numbers
- **Tables:** customers
- **Columns:** customers.customerName, customers.phone
- **Filters:** none
- **Joins:** none

### Q19 — Show product MSRP values
- **Intent:** Retrieve product names with their MSRP
- **Tables:** products
- **Columns:** products.productName, products.MSRP
- **Filters:** none
- **Joins:** none

### Q20 — List order numbers
- **Intent:** Retrieve order identifiers
- **Tables:** orders
- **Columns:** orders.orderNumber
- **Filters:** none
- **Joins:** none

---

## JOIN questions (Q21–Q30)

### Q21 — Get orders with customer names
- **Intent:** Retrieve orders annotated with the customer name
- **Tables:** orders, customers
- **Columns:** orders.orderNumber, customers.customerName
- **Filters:** none
- **Joins:** orders.customerNumber = customers.customerNumber

### Q22 — Get employees with office city
- **Intent:** Retrieve employees annotated with their office city
- **Tables:** employees, offices
- **Columns:** employees.firstName, employees.lastName, offices.city
- **Filters:** none
- **Joins:** employees.officeCode = offices.officeCode

### Q23 — Get payments with customer names
- **Intent:** Retrieve payments annotated with the paying customer's name
- **Tables:** payments, customers
- **Columns:** customers.customerName, payments.checkNumber, payments.amount
- **Filters:** none
- **Joins:** payments.customerNumber = customers.customerNumber

### Q24 — Get order details with product names
- **Intent:** Retrieve order-line items annotated with the product name
- **Tables:** orderdetails, products
- **Columns:** orderdetails.orderNumber, products.productName, orderdetails.quantityOrdered
- **Filters:** none
- **Joins:** orderdetails.productCode = products.productCode

### Q25 — Get products with product line description
- **Intent:** Retrieve products annotated with their line's text description
- **Tables:** products, productlines
- **Columns:** products.productName, productlines.textDescription
- **Filters:** none
- **Joins:** products.productLine = productlines.productLine

### Q26 — Get customers with sales rep names
- **Intent:** Retrieve customers annotated with their assigned sales rep
- **Tables:** customers, employees
- **Columns:** customers.customerName, employees.firstName, employees.lastName
- **Filters:** none (LEFT JOIN tolerated for customers without a rep)
- **Joins:** customers.salesRepEmployeeNumber = employees.employeeNumber

### Q27 — Get orders with customer city
- **Intent:** Retrieve orders annotated with the customer's city
- **Tables:** orders, customers
- **Columns:** orders.orderNumber, customers.city
- **Filters:** none
- **Joins:** orders.customerNumber = customers.customerNumber

### Q28 — Get employees and their manager
- **Intent:** Retrieve each employee with the name of their manager
- **Tables:** employees (self-joined as employees and managers)
- **Columns:** employee.firstName, employee.lastName, manager.firstName, manager.lastName
- **Filters:** none (LEFT JOIN — top of hierarchy has no manager)
- **Joins:** employee.reportsTo = manager.employeeNumber

### Q29 — Get orderdetails with product vendor
- **Intent:** Retrieve order-line items annotated with the product's vendor
- **Tables:** orderdetails, products
- **Columns:** orderdetails.orderNumber, products.productVendor, orderdetails.quantityOrdered
- **Filters:** none
- **Joins:** orderdetails.productCode = products.productCode

### Q30 — Get payments with customer country
- **Intent:** Retrieve payments annotated with the paying customer's country
- **Tables:** payments, customers
- **Columns:** customers.customerName, customers.country, payments.amount
- **Filters:** none
- **Joins:** payments.customerNumber = customers.customerNumber

---

## Group-by aggregations (Q31–Q40)

### Q31 — Count customers per country
- **Intent:** Count customers grouped by country
- **Tables:** customers
- **Columns:** customers.country
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)
- **Group by:** customers.country

### Q32 — Total payments per customer
- **Intent:** Sum payment amounts grouped by customer
- **Tables:** payments
- **Columns:** payments.customerNumber
- **Filters:** none
- **Joins:** none
- **Aggregations:** SUM(payments.amount)
- **Group by:** payments.customerNumber

### Q33 — Number of orders per status
- **Intent:** Count orders grouped by status
- **Tables:** orders
- **Columns:** orders.status
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)
- **Group by:** orders.status

### Q34 — Products per product line
- **Intent:** Count products grouped by product line
- **Tables:** products
- **Columns:** products.productLine
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)
- **Group by:** products.productLine

### Q35 — Employees per office
- **Intent:** Count employees grouped by office
- **Tables:** employees
- **Columns:** employees.officeCode
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)
- **Group by:** employees.officeCode

### Q36 — Total stock per product vendor
- **Intent:** Sum quantity in stock grouped by vendor
- **Tables:** products
- **Columns:** products.productVendor
- **Filters:** none
- **Joins:** none
- **Aggregations:** SUM(products.quantityInStock)
- **Group by:** products.productVendor

### Q37 — Average buy price per product line
- **Intent:** Average wholesale price grouped by product line
- **Tables:** products
- **Columns:** products.productLine
- **Filters:** none
- **Joins:** none
- **Aggregations:** AVG(products.buyPrice)
- **Group by:** products.productLine

### Q38 — Orders per customer
- **Intent:** Count orders grouped by customer
- **Tables:** orders
- **Columns:** orders.customerNumber
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)
- **Group by:** orders.customerNumber

### Q39 — Max MSRP per product line
- **Intent:** Highest MSRP grouped by product line
- **Tables:** products
- **Columns:** products.productLine
- **Filters:** none
- **Joins:** none
- **Aggregations:** MAX(products.MSRP)
- **Group by:** products.productLine

### Q40 — Min buy price per vendor
- **Intent:** Lowest wholesale price grouped by vendor
- **Tables:** products
- **Columns:** products.productVendor
- **Filters:** none
- **Joins:** none
- **Aggregations:** MIN(products.buyPrice)
- **Group by:** products.productVendor

---

## Scalar aggregates (Q41–Q50)

### Q41 — Total number of customers
- **Intent:** Count all customer rows
- **Tables:** customers
- **Columns:** none (just count)
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)

### Q42 — Total number of products
- **Intent:** Count all product rows
- **Tables:** products
- **Columns:** none
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)

### Q43 — Total revenue from payments
- **Intent:** Sum every payment amount
- **Tables:** payments
- **Columns:** payments.amount
- **Filters:** none
- **Joins:** none
- **Aggregations:** SUM(payments.amount)

### Q44 — Average product price
- **Intent:** Average retail price across products
- **Tables:** products
- **Columns:** products.MSRP
- **Filters:** none
- **Joins:** none
- **Aggregations:** AVG(products.MSRP)
- **Note:** "price" mapped to MSRP (retail) by convention

### Q45 — Max payment amount
- **Intent:** Largest single payment
- **Tables:** payments
- **Columns:** payments.amount
- **Filters:** none
- **Joins:** none
- **Aggregations:** MAX(payments.amount)

### Q46 — Min payment amount
- **Intent:** Smallest single payment
- **Tables:** payments
- **Columns:** payments.amount
- **Filters:** none
- **Joins:** none
- **Aggregations:** MIN(payments.amount)

### Q47 — Count total orders
- **Intent:** Count all order rows
- **Tables:** orders
- **Columns:** none
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)

### Q48 — Total quantity in stock
- **Intent:** Sum of quantity in stock across all products
- **Tables:** products
- **Columns:** products.quantityInStock
- **Filters:** none
- **Joins:** none
- **Aggregations:** SUM(products.quantityInStock)

### Q49 — Average MSRP
- **Intent:** Average MSRP across all products
- **Tables:** products
- **Columns:** products.MSRP
- **Filters:** none
- **Joins:** none
- **Aggregations:** AVG(products.MSRP)

### Q50 — Number of employees
- **Intent:** Count all employee rows
- **Tables:** employees
- **Columns:** none
- **Filters:** none
- **Joins:** none
- **Aggregations:** COUNT(*)
