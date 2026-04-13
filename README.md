# Europe Odoo Integration

> A production-ready REST API built with **FastAPI**, acting as a middleware layer between external systems (e.g. Node-RED) and the **Odoo ERP** platform. Developed as part of a real-world integration project for Europe car dealership operations.
>
> ---
>
> ## Overview
>
> This project exposes a clean HTTP API on top of Odoo's XML-RPC interface, enabling external tools to interact with the ERP system without needing to deal with Odoo's low-level RPC protocol. It orchestrates a complete sales workflow - from customer listing and quotation creation, to invoicing and PDF report generation.
>
> **Key integration areas:**
> - Customer & Product catalog management
> - - Multi-vehicle quotation creation
>   - - Shipping method selection with dynamic pricing
>     - - Delivery date scheduling
>       - - Invoice creation, validation & PDF download
>        
>         - ---
>
> ## Tech Stack
>
> | Layer | Technology |
> |---|---|
> | Language | Python 3.10+ |
> | API Framework | FastAPI |
> | Data Validation | Pydantic |
> | ERP Backend | Odoo 16 (XML-RPC) |
> | HTTP Client | xmlrpc.client, urllib, certifi |
> | PDF Generation | Odoo Report Engine (session-based) |
> | Server | Uvicorn (ASGI) |
>
> ---
>
> ## Features
>
> - **Full Sales Workflow**: Covers the entire order lifecycle from quotation to invoicing
> - - **Multi-Vehicle Quotation**: Create complex quotations with multiple vehicles, each with associated services
>   - - **Dynamic Shipping Pricing**: Automated calculation - CHF 800 base fee + CHF 1.00/kg
>     - - **Dual Date Format Support**: Accepts both DD/MM/YYYY and YYYY-MM-DD for delivery scheduling
>       - - **Node-RED Compatible Output**: Endpoints support a ?format=nodered query parameter for seamless Node-RED integration
>         - - **PDF Invoice Download**: Session-authenticated Odoo report generation returned as a binary stream
>           - - **Order Management**: Cancel, confirm, and summarize sale orders programmatically
>            
>             - ---
>
> ## API Reference
>
> ### System
>
> | Method | Endpoint | Description |
> |---|---|---|
> | GET | /get-status | Check Odoo connectivity and retrieve server version |
>
> ### Customers
>
> | Method | Endpoint | Description |
> |---|---|---|
> | GET | /customers/list | List all customers (supports ?format=nodered&customer_type=...) |
> | GET | /customers/by-type/{type} | Filter customers by type (individual / company) |
> | GET | /customers/{id} | Get detailed info for a specific customer |
>
> ### Products
>
> | Method | Endpoint | Description |
> |---|---|---|
> | GET | /products/list | List products (supports ?filter_cars=true and ?filter_services=true) |
>
> > Products use prefix conventions: CAR- for vehicles, SERV- for services.
> >
> > ### Sale Orders
> >
> > | Method | Endpoint | Description |
> > |---|---|---|
> > | POST | /saleorders/quotation | Create a multi-vehicle quotation |
> > | GET | /saleorders/{id} | List all sale orders for a customer |
> > | POST | /saleorders/{id}/confirm | Confirm a quotation -> Sale Order |
> > | POST | /saleorders/{id}/cancel | Cancel a draft or confirmed sale order |
> > | POST | /saleorders/{id}/shipping | Set shipping method and calculate cost |
> > | POST | /saleorders/delivery-date | Update the scheduled delivery date |
> > | GET | /saleorders/{id}/summary | Get a full summary (carrier, cost, date, status, totals) |
> > | POST | /saleorders/{id}/invoice | Create and validate a customer invoice |
> > | GET | /saleorders/{id}/pdf | Download the invoice as a PDF file |
> >
> > ### Invoices
> >
> > | Method | Endpoint | Description |
> > |---|---|---|
> > | GET | /invoices/list | List all posted customer invoices |
> >
> > ---
> >
> > ## Shipping Pricing Logic
> >
> > When the shipping method is set to **Home Delivery**, the pricing is calculated as follows:
> >
> > ```
> > Total Shipping Cost = CHF 800 (base) + (weight_kg x CHF 1.00)
> > ```
> >
> > **Pickup** orders have no shipping cost applied.
> >
> > ---
> >
> > ## Quotation Payload Example
> >
> > ```json
> > {
> >   "customer_id": 42,
> >   "vehicles": [
> >     {
> >       "product_id": 101,
> >       "quantity": 1,
> >       "services": [
> >         { "product_id": 205, "quantity": 1 }
> >       ]
> >     }
> >   ]
> > }
> > ```
> >
> > ---
> >
> > ## Installation
> >
> > ### Prerequisites
> > - Python 3.10+
> > - - Access to an Odoo 16 instance
> >   - - pip
> >    
> >     - ### Setup
> >    
> >     - 1. Clone the repository:
> >       2. ```bash
> >          git clone https://github.com/marcelo-fg/Europe_Odoo.git
> >          cd Europe_Odoo
> >          ```
> >
> > 2. Create and activate a virtual environment:
> > 3. ```bash
> >    python -m venv venv
> >    source venv/bin/activate  # On Windows: venv\Scripts\activate
> >    ```
> >
> > 3. Install dependencies:
> > 4. ```bash
> >    pip install fastapi uvicorn pydantic certifi
> >    ```
> >
> > 4. Configure your Odoo credentials in my_odoo_api.py:
> > 5. ```python
> >    URL = "https://your-odoo-instance.odoo.com"
> >    DB  = "your-database-name"
> >    USER = "your-email@domain.com"
> >    PW  = "your-password"
> >    ```
> >
> > ---
> >
> > ## Running the API
> >
> > Start the development server:
> > ```bash
> > uvicorn my_odoo_api:app --reload
> > ```
> >
> > The interactive API documentation (Swagger UI) will be available at:
> > ```
> > http://127.0.0.1:8000/docs
> > ```
> >
> > ---
> >
> > ## Project Structure
> >
> > ```
> > Europe_Odoo/
> > |-- my_odoo_api.py     # Main FastAPI application - all endpoints and business logic
> > |-- .gitignore         # Python & environment exclusions
> > |-- README.md          # Project documentation
> > ```
> >
> > ---
> >
> > ## Node-RED Integration
> >
> > Several endpoints support a ?format=nodered query parameter. When enabled, the response is restructured to match the payload format expected by Node-RED function nodes, allowing direct integration with visual automation flows.
> >
> > Example:
> > ```
> > GET /customers/list?format=nodered
> > ```
> >
> > ---
> >
> > ## Odoo Configuration Requirements
> >
> > For the integration to work correctly, ensure your Odoo instance is configured with:
> >
> > - **Product naming conventions**: Vehicle products prefixed with CAR-, service products with SERV-
> > - - **Delivery carrier**: A carrier named "Home Delivery" must exist in Odoo
> >   - - **Payment method**: An Odoo payment wizard (sale.advance.payment.inv) must be available on the instance
> >     - - **Stock module**: The inventory/stock module must be enabled for delivery scheduling
> >      
> >       - ---
> >
> > ## Security Note
> >
> > The current implementation stores Odoo credentials directly in the source code. For production deployments, credentials should be moved to environment variables using a .env file and a library such as python-dotenv.
> >
> > Recommended approach:
> > ```bash
> > pip install python-dotenv
> > ```
> > ```python
> > from dotenv import load_dotenv
> > import os
> >
> > load_dotenv()
> > URL  = os.getenv("ODOO_URL")
> > DB   = os.getenv("ODOO_DB")
> > USER = os.getenv("ODOO_USER")
> > PW   = os.getenv("ODOO_PASSWORD")
> > ```
> >
> > ---
> >
> > ## Author
> >
> > **Marcelo Ferreira Goncalves
> > ---
> >
> > ## Odoo Configuration Requirements
> >
> > For the integration to work correctly, ensure your Odoo instance is configured with:
> >
> > - **Product naming conventions**: Vehicle products prefixed with CAR-, service products with SERV-
> > - - **Delivery carrier**: A carrier named "Home Delivery" must exist in Odoo
> >   - - **Payment method**: An Odoo payment wizard (sale.advance.payment.inv) must be available on the instance
> >     - - **Stock module**: The inventory/stock module must be enabled for delivery scheduling
> >      
> >       - ---
> >
> > ## Security Note
> >
> > The current implementation stores Odoo credentials directly in the source code. For production deployments, credentials should be moved to environment variables using a .env file and a library such as python-dotenv.
> >
> > Recommended approach:
> > ```bash
> > pip install python-dotenv
> > ```
> > ```python
> > from dotenv import load_dotenv
> > import os
> >
> > load_dotenv()
> > URL  = os.getenv("ODOO_URL")
> > DB   = os.getenv("ODOO_DB")
> > USER = os.getenv("ODOO_USER")
> > PW   = os.getenv("ODOO_PASSWORD")
> > ```
> >
> > ---
> >
> > ## Author
> >
> > **Marcelo Ferreira Goncalves**
> > HEC Lausanne - Information Systems
> > [GitHub](https://github.com/marcelo-fg)
> >
