# Europe Odoo Integration API

> REST API middleware built with FastAPI that bridges external systems (Node-RED, custom clients) with the Odoo 16 ERP platform, orchestrating a complete B2B automotive sales workflow.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [Phase 1 — Setup & References](#phase-1--setup--references)
  - [Phase 2 — Sales Process](#phase-2--sales-process)
  - [Phase 3 — Delivery Management](#phase-3--delivery-management)
  - [Phase 4 — Invoicing](#phase-4--invoicing)
- [Data Models](#data-models)
- [Node-RED Integration](#node-red-integration)
- [Shipping Pricing Logic](#shipping-pricing-logic)
- [Security](#security)
- [Author](#author)

---

## Overview

**Europe Odoo Integration API** is a production-ready FastAPI middleware that exposes a clean HTTP interface on top of Odoo's XML-RPC protocol. It eliminates the complexity of dealing with low-level RPC calls for external automation systems, providing a well-structured REST API that covers the full lifecycle of a B2B vehicle sale:

```
Customer Lookup → Multi-Vehicle Quotation → Delivery Scheduling → Invoice & PDF
```

Designed and built for the Europe car dealership operations workflow at HEC Lausanne.

---

## Features

- **Customer Management** — List, filter (individual/company), and retrieve detailed customer records
- **Product Catalog** — Browse vehicles (`CAR-` prefix) and services (`SERV-` prefix) from Odoo
- **Multi-Vehicle Quotations** — Create complex quotations with per-vehicle service associations
- **Order Lifecycle** — Confirm, cancel, and query sale orders
- **Delivery Management** — Set shipping method (home delivery or pickup) with automatic cost calculation
- **Delivery Scheduling** — Assign delivery dates in multiple formats (`DD/MM/YYYY` and `YYYY-MM-DD`)
- **Invoicing** — Create, post, and download customer invoices as PDF
- **Node-RED Compatible** — All endpoints support a `?format=nodered` query parameter for seamless Node-RED integration
- **Interactive Docs** — Auto-generated Swagger UI at `/docs` and ReDoc at `/redoc`

---

## Tech Stack

| Layer              | Technology              |
|--------------------|-------------------------|
| Language           | Python 3.10+            |
| API Framework      | FastAPI                 |
| Data Validation    | Pydantic                |
| ERP Backend        | Odoo 16 (XML-RPC)       |
| ASGI Server        | Uvicorn                 |
| PDF Generation     | Odoo Report Engine      |
| SSL / Certificates | certifi                 |

---

## Project Structure

```
Europe_Odoo/
├── my_odoo_api.py      # Main application — all endpoints and business logic
├── .gitignore
└── README.md
```

The project follows a single-module architecture for simplicity and portability.

---

## Getting Started

### Prerequisites

- Python **3.10** or higher
- Access to an Odoo 16 instance
- `pip` or a compatible package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/marcelo-fg/europe_odoo.git
cd europe_odoo

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install fastapi uvicorn pydantic certifi
```

### Configuration

> **Security notice:** Never commit credentials to version control.

Create a `.env` file at the project root (recommended with `python-dotenv`):

```env
ODOO_URL=https://your-instance.odoo.com
ODOO_DB=your-database
ODOO_USER=your@email.com
ODOO_PW=your-api-key
```

Then update `my_odoo_api.py` to load these values:

```python
from dotenv import load_dotenv
import os

load_dotenv()

URL  = os.getenv("ODOO_URL")
DB   = os.getenv("ODOO_DB")
USER = os.getenv("ODOO_USER")
PW   = os.getenv("ODOO_PW")
```

### Running the Server

```bash
uvicorn my_odoo_api:app --reload
```

The API will be available at:

| Interface  | URL                           |
|------------|-------------------------------|
| API Base   | `http://127.0.0.1:8000`       |
| Swagger UI | `http://127.0.0.1:8000/docs`  |
| ReDoc      | `http://127.0.0.1:8000/redoc` |

---

## API Reference

### Phase 1 — Setup & References

| Method | Endpoint                    | Description                                            |
|--------|-----------------------------|--------------------------------------------------------|
| `GET`  | `/get-status`               | Check Odoo connectivity and server version             |
| `GET`  | `/customers/list`           | List all customers (optional `?type=` and `?format=`)  |
| `GET`  | `/customers/by-type/{type}` | Filter customers by `individual` or `company`          |
| `GET`  | `/customers/{id}`           | Get full details for a specific customer               |
| `GET`  | `/products/list`            | List all products (filterable: `cars`, `services`)     |

**Query Parameters — `/customers/list`:**

| Parameter | Values                  | Description                                |
|-----------|-------------------------|--------------------------------------------|
| `type`    | `individual`, `company` | Filter by customer type                    |
| `format`  | `nodered`               | Return Node-RED compatible response format |

---

### Phase 2 — Sales Process

| Method | Endpoint                    | Description                                               |
|--------|-----------------------------|-----------------------------------------------------------|
| `POST` | `/saleorders/quotation`     | Create a multi-vehicle quotation with per-vehicle services |
| `POST` | `/saleorders/{id}/confirm`  | Confirm a draft quotation → sale order                    |
| `POST` | `/saleorders/{id}/cancel`   | Cancel a draft or confirmed order                         |
| `GET`  | `/saleorders/{customer_id}` | List all sale orders for a given customer                 |

**Example — Create Quotation Payload:**

```json
{
  "customer_id": 42,
  "vehicles": [
    {
      "product_id": 101,
      "quantity": 1,
      "services": [
        { "product_id": 201, "quantity": 1 },
        { "product_id": 202, "quantity": 2, "apply_to_quantity": true }
      ]
    },
    {
      "product_id": 103,
      "quantity": 2,
      "services": []
    }
  ]
}
```

---

### Phase 3 — Delivery Management

| Method | Endpoint                            | Description                                               |
|--------|-------------------------------------|-----------------------------------------------------------|
| `POST` | `/saleorders/{id}/shipping-method`  | Set shipping method and calculate cost                    |
| `POST` | `/saleorders/delivery-date`         | Schedule a delivery date for an order                     |
| `GET`  | `/saleorders/{id}/delivery-summary` | Get complete delivery info (carrier, cost, date, status)  |

**Shipping Method Payload:**

```json
{
  "order_id": 55,
  "method": "home_delivery"
}
```

Accepted values for `method`: `home_delivery`, `pickup`

---

### Phase 4 — Invoicing

| Method | Endpoint                       | Description                           |
|--------|--------------------------------|---------------------------------------|
| `POST` | `/saleorders/{id}/invoice`     | Create and post a customer invoice    |
| `GET`  | `/invoices/list`               | List all posted customer invoices     |
| `GET`  | `/saleorders/{id}/invoice/pdf` | Download the invoice as a PDF file    |

---

## Data Models

### `ServiceLine`

```python
class ServiceLine(BaseModel):
    product_id: int           # Must be a SERV- prefixed product
    quantity: float           # Must be > 0
    apply_to_quantity: bool   # If True, multiplied by parent vehicle quantity
```

### `VehicleLine`

```python
class VehicleLine(BaseModel):
    product_id: int              # Must be a CAR- prefixed product
    quantity: float              # Must be > 0
    services: list[ServiceLine]  # Associated services (can be empty)
```

### `QuotationRequest`

```python
class QuotationRequest(BaseModel):
    customer_id: int
    vehicles: list[VehicleLine]
```

---

## Node-RED Integration

All endpoints support the `?format=nodered` query parameter. When enabled, responses are wrapped in a Node-RED-compatible envelope:

```json
{
  "payload": { "..." : "..." },
  "statusCode": 200,
  "topic": "odoo-response"
}
```

**Example Node-RED HTTP Request node configuration:**

```
Method : GET
URL    : http://localhost:8000/customers/list?format=nodered
```

---

## Shipping Pricing Logic

| Method        | Base Price  | Weight Surcharge | Result  |
|---------------|-------------|------------------|---------|
| Home Delivery | CHF 800.00  | CHF 1.00 / kg    | Dynamic |
| Pickup        | CHF 0.00    | None             | Free    |

The total weight is computed from the confirmed order lines in Odoo. The shipping line is automatically added to the sale order upon calling `/shipping-method`.

---

## Security

| Status | Item                                                     |
|--------|----------------------------------------------------------|
| ⚠️     | Hardcoded credentials in source — migrate to `.env`      |
| ⚠️     | No authentication on API endpoints — restrict to LAN/VPN |
| ⚠️     | No rate limiting — consider adding in production         |
| ✅     | SSL/TLS enforced for all Odoo connections via `certifi`  |
| ✅     | Session-based PDF download (no credential reuse in links)|
| ✅     | Input validation on all endpoints via Pydantic           |

> For production use, secure all endpoints behind an API gateway or VPN, and replace hardcoded credentials with environment variables.

---

## Author

**Marcelo Ferreira Goncalves**  
HEC Lausanne — Information Systems

---

*Built on FastAPI + Odoo 16 XML-RPC*
```
