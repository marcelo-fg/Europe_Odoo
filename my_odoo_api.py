"""FastAPI endpoints exposing the Odoo sales workflow."""

import base64
from datetime import datetime, timezone
import ssl
import xmlrpc.client

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

try:
    import certifi
except ImportError as exc:  # pragma: no cover - documented dependency
    raise ImportError("Missing dependency 'certifi'. Install it with 'pip install certifi'.") from exc


# ---------------------------------------------------------------------------
# Odoo connection setup
# ---------------------------------------------------------------------------
URL = "https://edu-heclausanne-europe.odoo.com"
DB = "edu-heclausanne-europe"
USER = "marcelo.ferreiragoncalves@unil.ch"
PW = "buqsog-1bykzu-fuftYh"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def connect_odoo(hosturl: str, db: str, user: str, pw: str):
    """Authenticate against Odoo and return (uid, common) proxy."""
    common = xmlrpc.client.ServerProxy(
        f"{hosturl}/xmlrpc/2/common",
        context=SSL_CONTEXT,
        allow_none=True,
    )
    uid = common.authenticate(db, user, pw, {})
    if not uid:
        raise ConnectionError("Invalid Odoo credentials")
    return uid, common


UID, _COMMON = connect_odoo(URL, DB, USER, PW)


class OrderLine(BaseModel):
    product_id: int
    quantity: float


class QuotationRequest(BaseModel):
    customer_id: int
    order_lines: list[OrderLine]


class DeliveryProposal(BaseModel):
    new_date: datetime


app = FastAPI()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _models():
    return xmlrpc.client.ServerProxy(
        f"{URL}/xmlrpc/2/object",
        context=SSL_CONTEXT,
        allow_none=True,
    )


def _success(message: str, data: object | None = None):
    payload = {"status": "success", "message": message}
    if data is not None:
        payload["data"] = data
    return payload


def _http_error(status_code: int, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"status": "error", "message": message})

def _nodered_response(status_type: str, message: str, data=None, notification: str = None):
    """
    Format standardisé pour Node-RED.
    
    Args:
        status_type: "success", "error", "warning"
        message: Message technique
        data: Données (optionnel)
        notification: Message pour ui_notification (optionnel)
    """
    response = {
        "status": status_type,
        "message": message,
    }
    
    if data is not None:
        response["data"] = data
    
    if notification:
        response["notification"] = notification
    elif status_type == "success":
        response["notification"] = f"✓ {message}"
    elif status_type == "error":
        response["notification"] = f"✗ {message}"
    elif status_type == "warning":
        response["notification"] = f"⚠ {message}"
    
    return response


def _ensure_positive(value: int | float, name: str) -> None:
    if value <= 0:
        _http_error(status.HTTP_400_BAD_REQUEST, f"{name} must be a positive value")



def _fetch_single_record(models_proxy, model: str, record_id: int, *, fields: list[str], not_found_detail: str):
    try:
        records = models_proxy.execute_kw(
            DB,
            UID,
            PW,
            model,
            "read",
            [[record_id]],
            {"fields": fields},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while reading {model}: {fault_string}")
    except Exception as err:  # pragma: no cover - network failure
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while reading {model}: {err}")

    if not records:
        _http_error(status.HTTP_404_NOT_FOUND, not_found_detail)

    return records[0] or {}


def _extract_portal_path(action) -> str | None:
    if isinstance(action, dict):
        for key in ("url", "target", "portal_url", "access_url"):
            value = action.get(key)
            if isinstance(value, str) and value:
                return value
    elif isinstance(action, str):
        return action
    elif isinstance(action, list) and action:
        first = action[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return _extract_portal_path(first)
    return None


def _build_portal_url(path: str) -> str:
    cleaned = (path or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith(("http://", "https://")):
        return cleaned
    base_url = URL.rstrip("/")
    return f"{base_url}/{cleaned.lstrip('/')}"


def _resolve_portal_url(
    models_proxy,
    *,
    model: str,
    record_id: int,
    candidate_methods: list[str],
    fallback_paths: list[str],
    error_label: str,
) -> str:
    for method_name in candidate_methods:
        try:
            action = models_proxy.execute_kw(
                DB,
                UID,
                PW,
                model,
                method_name,
                [[record_id]],
            )
        except xmlrpc.client.Fault as fault:
            fault_string = getattr(fault, "faultString", str(fault))
            if "does not exist" in fault_string.lower():
                continue
            _http_error(
                status.HTTP_502_BAD_GATEWAY,
                f"Odoo error while preparing {error_label}: {fault_string}",
            )
        except Exception:
            continue

        portal_path = _extract_portal_path(action)
        if portal_path:
            return _build_portal_url(portal_path)

    for path in fallback_paths:
        if isinstance(path, str) and path.strip():
            return _build_portal_url(path)

    _http_error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        f"Odoo did not return a portal link for {error_label}",
    )


# ---------------------------------------------------------------------------
# Phase 1 – Setup and References
# ---------------------------------------------------------------------------
@app.get("/get-status")
def get_status():
    """Return the current Odoo connection status."""
    try:
        uid, common = connect_odoo(URL, DB, USER, PW)
        version = common.version()
    except Exception as err:  # pragma: no cover - network failure
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo connection failed: {err}")

    info = {
        "user_id": uid,
        "database": DB,
        "server_version": version.get("server_version"),
    }
    return _success("Odoo instance reachable", info)



@app.get("/customers/list")
def list_customers(format: str = "standard", customer_type: str = "all"):
    """
    List customers available for sales documents.
    
    Args:
        format: "standard" (default) or "nodered" 
        customer_type: "all", "individual", "company"
    """
    models = _models()
    
    # Validation du customer_type
    valid_types = {"all", "individual", "company"}
    customer_type = customer_type.lower().strip()
    
    if customer_type not in valid_types:
        if format == "nodered":
            return _nodered_response(
                "error",
                f"Invalid customer_type: '{customer_type}'",
                notification="✗ Invalid type. Use 'all', 'individual' or 'company'"
            )
        _http_error(status.HTTP_400_BAD_REQUEST, "customer_type must be 'all', 'individual' or 'company'")
    
    # Appel Odoo
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "res.partner",
            "search_read",
            [[("customer_rank", ">", 0)]],
            {"fields": ["id", "name", "is_company", "email", "phone", "city", "country_id"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        if format == "nodered":
            return _nodered_response("error", f"Odoo error: {fault_string}")
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing customers: {fault_string}")
    except Exception as err:
        if format == "nodered":
            return _nodered_response("error", str(err))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing customers: {err}")

    # Transformation des données
    customers = []
    for record in records:
        customer_type_record = "company" if record.get("is_company") else "individual"
        
        # Filtrage selon le type demandé
        if customer_type != "all" and customer_type_record != customer_type:
            continue
        
        # Extraction du nom du pays
        country_id = record.get("country_id")
        country_name = country_id[1] if country_id and isinstance(country_id, list) else ""
        
        customers.append({
            "id": record.get("id"),
            "name": record.get("name"),
            "customer_type": customer_type_record,
            "email": record.get("email") or "",
            "phone": record.get("phone") or "",
            "city": record.get("city") or "",
            "country": country_name
        })

    # Format pour Node-RED
    if format == "nodered":
        if not customers:
            return _nodered_response(
                "warning",
                f"No customers found (filter: {customer_type})",
                data=[]
            )
        
        # Retourner directement le tableau pour ui_table
        return customers
    
    # Format standard (avec wrapper)
    return _success("Customers retrieved", {"customers": customers})



@app.get("/customers/by-type/{customer_type}")
def list_customers_by_type(customer_type: str, format: str = "standard"):
    """List customers filtered by type (individual or company)."""
    valid_types = {"individual", "company"}
    if customer_type not in valid_types:
        _http_error(status.HTTP_400_BAD_REQUEST, "customer_type must be 'individual' or 'company'")

    models = _models()
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "res.partner",
            "search_read",
            [[("customer_rank", ">", 0)]],
            {"fields": ["id", "name", "is_company"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing customers: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing customers: {err}")

    customers = []
    for record in records:
        customer_type_record = "company" if record.get("is_company") else "individual"
        if customer_type_record == customer_type:
            customers.append(
                {
                    "id": record.get("id"),
                    "name": record.get("name"),
                    "customer_type": customer_type_record,
                }
            )

    # Format pour Node-RED (direct table)
    if format == "nodered":
        # Retourner directement le tableau pour ui_table
        return customers
    
    # Format standard (avec wrapper)
    return _success(f"Customers of type '{customer_type}' retrieved", {"customers": customers})


@app.get("/products/list")
def list_products(format: str = "standard", filter_cars: bool = True, filter_services: bool = False):
    """
    List salable products.
    
    Args:
        format: "standard" or "nodered"
        filter_cars: If True, include cars (default_code starts with CAR-)
        filter_services: If True, include services (default_code starts with SRV-)
    """
    models = _models()
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "product.product",
            "search_read",
            [[]],
            {"fields": ["id", "name", "default_code", "list_price", "categ_id", "type"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        if format == "nodered":
            return {"status": "error", "message": f"Odoo error: {fault_string}"}
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing products: {fault_string}")
    except Exception as err:
        if format == "nodered":
            return {"status": "error", "message": str(err)}
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing products: {err}")

    products = []
    for record in records:
        default_code = record.get("default_code", "")
        
        # Filtrage selon le type demandé
        is_car = default_code.startswith("CAR-") if default_code else False
        is_service = default_code.startswith("SRV-") if default_code else False
        
        if filter_cars and not filter_services:
            if not is_car:
                continue
        elif filter_services and not filter_cars:
            if not is_service:
                continue
        
        # Extraction de la catégorie
        categ = record.get("categ_id")
        categ_name = categ[1] if (categ and len(categ) > 1) else ""
        
        products.append({
            "id": record.get("id"),
            "name": record.get("name"),
            "code": default_code,
            "price": record.get("list_price", 0),
            "category": categ_name,
            "type": record.get("type", "product")
        })

    # Format pour Node-RED
    if format == "nodered":
        if not products:
            return {"status": "warning", "message": "No products found", "data": []}
        
        # Format pour dropdowns avec prix formaté
        return [
            {
                "id": p["id"],
                "code": p["code"],
                "name": p["name"],
                "price": p["price"],
                "price_formatted": f"CHF {p['price']:,.0f}".replace(",", "'"),
                "category": p["category"],
                "display_label": f"{p['name']} - {p['code']} (CHF {p['price']:,.0f})".replace(",", "'")
            }
            for p in products
        ]
    
    # Format standard
    return _success("Products retrieved", {"products": products})


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    """Retrieve a single customer details."""
    _ensure_positive(customer_id, "customer_id")
    models = _models()
    record = _fetch_single_record(
        models,
        "res.partner",
        customer_id,
        fields=["name", "email", "city", "country_id", "comment"],
        not_found_detail=f"Customer {customer_id} not found",
    )
    return _success("Customer retrieved", record)

# ---------------------------------------------------------------------------
# Phase 2 – Sales Process
# ---------------------------------------------------------------------------
from typing import Optional

@app.post("/saleorders/quotation")
def create_quotation(
    format: str = "standard",
    customer_id: Optional[int] = None,
    vehicle_id: Optional[int] = None,
    quantity: int = 1,
    payload: Optional[QuotationRequest] = None
):
    """
    Create a draft sale quotation.
    
    Two modes:
    1. JSON body: {"customer_id": 89, "order_lines": [...]}
    2. Query params: ?customer_id=89&vehicle_id=184&quantity=1
    """
    
    # Mode query params (Node-RED sans code)
    if customer_id is not None and vehicle_id is not None:
        order_lines = [OrderLine(product_id=vehicle_id, quantity=quantity)]
        payload = QuotationRequest(customer_id=customer_id, order_lines=order_lines)
    
    # Validation
    if payload is None:
        error_msg = "Missing required parameters. Provide either JSON body or query params (customer_id, vehicle_id)"
        if format == "nodered":
            return {
                "status": "error",
                "message": error_msg,
                "notification": "✗ Missing parameters"
            }
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate positive IDs
    try:
        _ensure_positive(payload.customer_id, "customer_id")
    except HTTPException as e:
        if format == "nodered":
            return {"status": "error", "message": str(e.detail), "notification": "✗ Invalid customer_id"}
        raise
    
    if not payload.order_lines:
        if format == "nodered":
            return {"status": "error", "message": "No products selected", "notification": "✗ Please select at least one product"}
        raise HTTPException(status_code=400, detail="order_lines must contain at least one product")

    for line in payload.order_lines:
        try:
            _ensure_positive(line.product_id, "product_id")
            _ensure_positive(line.quantity, "quantity")
        except HTTPException as e:
            if format == "nodered":
                return {"status": "error", "message": str(e.detail), "notification": "✗ Invalid product or quantity"}
            raise

    models = _models()
    
    try:
        # Check customer exists
        customer_exists = models.execute_kw(
            DB, UID, PW,
            "res.partner",
            "search",
            [[("id", "=", payload.customer_id)]],
            {"limit": 1},
        )
        if not customer_exists:
            msg = f"Customer {payload.customer_id} not found"
            if format == "nodered":
                return {"status": "error", "message": msg, "notification": f"✗ {msg}"}
            raise HTTPException(status_code=404, detail=msg)

        # Check products exist
        product_ids = [line.product_id for line in payload.order_lines]
        products_found = models.execute_kw(
            DB, UID, PW,
            "product.product",
            "search",
            [[("id", "in", product_ids)]],
        )
        missing_ids = sorted(set(product_ids) - set(products_found))
        if missing_ids:
            msg = f"Products not found: {', '.join(map(str, missing_ids))}"
            if format == "nodered":
                return {"status": "error", "message": msg, "notification": "✗ Products not found"}
            raise HTTPException(status_code=404, detail=msg)

        # Create quotation
        order_line_entries = [
            (0, 0, {"product_id": line.product_id, "product_uom_qty": line.quantity})
            for line in payload.order_lines
        ]

        quotation_id = models.execute_kw(
            DB, UID, PW,
            "sale.order",
            "create",
            [{"partner_id": payload.customer_id, "order_line": order_line_entries}],
        )
        
        # Get quotation info
        quotation_info = models.execute_kw(
            DB, UID, PW,
            "sale.order",
            "read",
            [[quotation_id]],
            {"fields": ["name", "state", "amount_total"]},
        )
        
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        if format == "nodered":
            return {"status": "error", "message": f"Odoo error: {fault_string}", "notification": "✗ Odoo error"}
        raise HTTPException(status_code=502, detail=f"Odoo error: {fault_string}")
        
    except Exception as err:
        if format == "nodered":
            return {"status": "error", "message": str(err), "notification": f"✗ Error: {err}"}
        raise HTTPException(status_code=502, detail=f"Error: {err}")

    info = quotation_info[0] if quotation_info else {}
    
    # Format Node-RED
    if format == "nodered":
        amount = info.get("amount_total", 0)
        order_name = info.get("name", "")
        amount_formatted = f"CHF {amount:,.0f}".replace(",", "'")
        
        return {
            "status": "success",
            "message": f"Quotation {order_name} created",
            "order_id": quotation_id,
            "order_name": order_name,
            "state": info.get("state"),
            "amount_total": amount,
            "amount_formatted": amount_formatted,
            "line_count": len(payload.order_lines),
            "notification": f"✓ Quotation {order_name} created! Total: {amount_formatted}"
        }
    
    # Format standard
    return {
        "status": "success",
        "message": "Quotation created",
        "data": {
            "order_id": quotation_id,
            "name": info.get("name"),
            "state": info.get("state"),
            "amount_total": info.get("amount_total"),
            "line_count": len(payload.order_lines),
        }
    }

@app.post("/saleorders/{order_id}/cancel")
def cancel_sale_order(order_id: int):
    """Cancel a draft or confirmed sale order."""
    _ensure_positive(order_id, "order_id")
    models = _models()
    order = _fetch_single_record(
        models,
        "sale.order",
        order_id,
        fields=["name", "state"],
        not_found_detail=f"Sale order {order_id} not found",
    )

    state = order.get("state")
    if state == "cancel":
        return _success("Sale order already cancelled", {"order_id": order_id, "state": state})

    if state not in {"draft", "sent", "sale", "done"}:
        _http_error(status.HTTP_400_BAD_REQUEST, f"Sale order {order_id} cannot be cancelled from state '{state}'")

    try:
        models.execute_kw(
            DB,
            UID,
            PW,
            "sale.order",
            "action_cancel",
            [[order_id]],
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while cancelling sale order: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while cancelling sale order: {err}")

    return _success("Sale order cancelled", {"order_id": order_id, "state": "cancel"})


@app.post("/saleorders/{order_id}/confirm")
def confirm_sale_order(order_id: int):
    """Confirm a quotation so it becomes a sale order."""
    _ensure_positive(order_id, "order_id")
    models = _models()
    try:
        models.execute_kw(
            DB,
            UID,
            PW,
            "sale.order",
            "action_confirm",
            [[order_id]],
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while confirming sale order: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while confirming sale order: {err}")

    return _success("Sale order confirmed", {"order_id": order_id})




@app.get("/saleorders/{customer_id}")
def list_customer_sale_orders(customer_id: int):
    """List sale orders belonging to a specific customer."""
    _ensure_positive(customer_id, "customer_id")
    models = _models()
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "sale.order",
            "search_read",
            [[("partner_id", "=", customer_id)]],
            {"fields": ["id", "name", "state", "amount_total", "create_date"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing sale orders: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing sale orders: {err}")

    return _success("Sale orders retrieved", {"orders": records})


# ---------------------------------------------------------------------------
# Phase 3 – Delivery Management
# ---------------------------------------------------------------------------
@app.get("/saleorders/{order_id}/delivery-options")
def list_delivery_options(order_id: int):
    """List delivery methods that can be applied to a sale order."""
    _ensure_positive(order_id, "order_id")
    models = _models()
    try:
        carriers = models.execute_kw(
            DB,
            UID,
            PW,
            "delivery.carrier",
            "search_read",
            [[]],
            {"fields": ["id", "name", "fixed_price", "delivery_type", "product_id"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing delivery options: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing delivery options: {err}")

    options = [
        {
            "id": carrier.get("id"),
            "name": carrier.get("name"),
            "price": carrier.get("fixed_price", 0.0),
            "type": carrier.get("delivery_type"),
            "product": carrier.get("product_id")[1] if carrier.get("product_id") else None,
        }
        for carrier in carriers
    ]
    return _success("Delivery options retrieved", {"order_id": order_id, "delivery_methods": options})


@app.post("/deliveries/{picking_id}/propose-date")
def propose_delivery_date(picking_id: int, payload: DeliveryProposal):
    """Propose a new delivery date for a picking."""
    _ensure_positive(picking_id, "picking_id")
    models = _models()
    picking = _fetch_single_record(
        models,
        "stock.picking",
        picking_id,
        fields=["scheduled_date", "state"],
        not_found_detail=f"Delivery picking {picking_id} not found",
    )

    state = picking.get("state")
    if state not in {"assigned", "waiting", "confirmed"}:
        _http_error(
            status.HTTP_400_BAD_REQUEST,
            f"Delivery {picking_id} cannot be rescheduled from state '{state}'",
        )

    new_dt = payload.new_date
    if new_dt.tzinfo is not None:
        new_dt_utc = new_dt.astimezone(timezone.utc)
        new_dt_naive = new_dt_utc.replace(tzinfo=None)
        if new_dt_utc <= datetime.now(timezone.utc):
            _http_error(status.HTTP_400_BAD_REQUEST, "new_date must be set in the future")
    else:
        new_dt_naive = new_dt
        if new_dt_naive <= datetime.utcnow():
            _http_error(status.HTTP_400_BAD_REQUEST, "new_date must be set in the future")

    new_date_str = new_dt_naive.strftime("%Y-%m-%d %H:%M:%S")
    try:
        models.execute_kw(
            DB,
            UID,
            PW,
            "stock.picking",
            "write",
            [[picking_id], {"scheduled_date": new_date_str}],
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while proposing delivery date: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while proposing delivery date: {err}")

    data = {
        "picking_id": picking_id,
        "old_date": picking.get("scheduled_date"),
        "new_date": new_date_str,
        "state": state,
    }
    return _success("Delivery date proposed", data)


@app.post("/deliveries/{picking_id}/validate")
def validate_delivery(picking_id: int):
    """Validate a delivery picking."""
    _ensure_positive(picking_id, "picking_id")
    models = _models()
    picking = _fetch_single_record(
        models,
        "stock.picking",
        picking_id,
        fields=["scheduled_date", "state"],
        not_found_detail=f"Delivery picking {picking_id} not found",
    )

    state = picking.get("state")
    if state not in {"assigned", "waiting", "confirmed"}:
        _http_error(
            status.HTTP_400_BAD_REQUEST,
            f"Delivery {picking_id} cannot be validated from state '{state}'",
        )

    try:
        result = models.execute_kw(
            DB,
            UID,
            PW,
            "stock.picking",
            "button_validate",
            [[picking_id]],
        )
        refreshed = models.execute_kw(
            DB,
            UID,
            PW,
            "stock.picking",
            "read",
            [[picking_id]],
            {"fields": ["scheduled_date", "state"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while validating delivery: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while validating delivery: {err}")

    updated = refreshed[0] if refreshed else {}
    data = {
        "picking_id": picking_id,
        "state": updated.get("state", state),
        "scheduled_date": updated.get("scheduled_date", picking.get("scheduled_date")),
        "result": result,
    }
    return _success("Delivery validated", data)


@app.get("/saleorders/{order_id}/deliveries")
def list_sale_order_deliveries(order_id: int):
    """List deliveries associated with a specific sale order."""
    _ensure_positive(order_id, "order_id")
    models = _models()
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "stock.picking",
            "search_read",
            [[("sale_id", "=", order_id)]],
            {"fields": ["id", "name", "scheduled_date", "state"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing deliveries: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing deliveries: {err}")

    return _success("Deliveries retrieved", {"deliveries": records})


@app.get("/deliveries/list")
def list_deliveries():
    """List deliveries with their current status."""
    models = _models()
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "stock.picking",
            "search_read",
            [[]],
            {"fields": ["id", "name", "scheduled_date", "state"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing deliveries: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing deliveries: {err}")

    return _success("Deliveries retrieved", {"deliveries": records})


# ---------------------------------------------------------------------------
# Phase 4 – Invoicing
# ---------------------------------------------------------------------------
@app.post("/saleorders/{order_id}/invoice", status_code=status.HTTP_201_CREATED)
def create_invoice(order_id: int):
    """Create and post a customer invoice for a confirmed sale order."""
    _ensure_positive(order_id, "order_id")
    models = _models()
    order = _fetch_single_record(
        models,
        "sale.order",
        order_id,
        fields=["name", "state"],
        not_found_detail=f"Sale order {order_id} not found",
    )

    if order.get("state") != "sale":
        _http_error(
            status.HTTP_400_BAD_REQUEST,
            f"Sale order {order_id} must be confirmed (state 'sale') before invoicing",
        )

    try:
        wizard_id = models.execute_kw(
            DB,
            UID,
            PW,
            "sale.advance.payment.inv",
            "create",
            [
                {
                    "advance_payment_method": "delivered",
                    "sale_order_ids": [(6, 0, [order_id])],
                }
            ],
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while preparing invoice wizard: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while preparing invoice wizard: {err}")

    wizard_context = {
        "active_model": "sale.order",
        "active_id": order_id,
        "active_ids": [order_id],
        "default_move_type": "out_invoice",
        "open_invoices": False,
    }
    action_result = None
    marshal_warning = None

    try:
        action_result = models.execute_kw(
            DB,
            UID,
            PW,
            "sale.advance.payment.inv",
            "create_invoices",
            [[wizard_id]],
            {"context": wizard_context},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        if "cannot marshal none" in fault_string.lower():
            marshal_warning = fault_string
        else:
            _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while generating invoice: {fault_string}")
    except Exception as err:
        message = str(err)
        if "cannot marshal none" in message.lower():
            marshal_warning = message
        else:
            _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while generating invoice: {err}")

    def _coerce_ids(result) -> list[int]:
        if isinstance(result, list):
            return [int(item) for item in result if isinstance(item, int)]
        if isinstance(result, int):
            return [result]
        if isinstance(result, dict):
            for key in ("res_ids", "res_id"):
                value = result.get(key)
                if isinstance(value, list):
                    return [int(item) for item in value if isinstance(item, int)]
                if isinstance(value, int):
                    return [value]
        return []

    invoice_ids = _coerce_ids(action_result) if action_result else []
    if not invoice_ids:
        try:
            invoice_ids = models.execute_kw(
                DB,
                UID,
                PW,
                "account.move",
                "search",
                [[("invoice_origin", "=", order.get("name"))]],
                {"order": "create_date desc", "limit": 1},
            )
        except Exception as err:
            _http_error(
                status.HTTP_502_BAD_GATEWAY,
                f"Invoice created but could not be located: {err}",
            )

    if not invoice_ids:
        base_message = f"Odoo did not return an invoice for sale order {order_id}"
        if marshal_warning:
            base_message = f"{base_message}. Original message: {marshal_warning}"
        _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, base_message)

    invoice_id = int(invoice_ids[0])
    try:
        models.execute_kw(
            DB,
            UID,
            PW,
            "account.move",
            "action_post",
            [[invoice_id]],
        )
        invoice_data = models.execute_kw(
            DB,
            UID,
            PW,
            "account.move",
            "read",
            [[invoice_id]],
            {"fields": ["name", "amount_total", "state"]},
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while posting invoice: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while posting invoice: {err}")

    invoice_record = invoice_data[0] if invoice_data else {}
    if invoice_record.get("state") != "posted":
        _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Invoice {invoice_id} is in state '{invoice_record.get('state')}', expected 'posted'",
        )

    data = {
        "invoice_id": invoice_id,
        "invoice_number": invoice_record.get("name"),
        "amount_total": invoice_record.get("amount_total"),
        "state": invoice_record.get("state"),
    }
    return _success("Invoice created and posted", data)


@app.get("/invoices/list")
def list_invoices():
    """List all customer invoices."""
    models = _models()
    try:
        records = models.execute_kw(
            DB,
            UID,
            PW,
            "account.move",
            "search_read",
            [[("move_type", "=", "out_invoice")]],
            {
                "fields": [
                    "id",
                    "name",
                    "state",
                    "amount_total",
                    "invoice_date",
                    "partner_id",
                    "invoice_origin",
                ],
                "order": "invoice_date desc",
            },
        )
    except xmlrpc.client.Fault as fault:
        fault_string = getattr(fault, "faultString", str(fault))
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing invoices: {fault_string}")
    except Exception as err:
        _http_error(status.HTTP_502_BAD_GATEWAY, f"Odoo error while listing invoices: {err}")

    # Transform data for better readability
    invoices = []
    for record in records:
        partner = record.get("partner_id")
        customer_name = partner[1] if partner and isinstance(partner, list) else ""
        
        invoices.append({
            "id": record.get("id"),
            "invoice_number": record.get("name"),
            "state": record.get("state"),
            "amount_total": record.get("amount_total"),
            "invoice_date": record.get("invoice_date"),
            "customer_name": customer_name,
            "sale_order": record.get("invoice_origin"),
        })

    return _success("Invoices retrieved", {"invoices": invoices})


@app.get("/invoices/{invoice_id}/pdf")
def download_invoice_pdf(invoice_id: int):
    """Download the PDF file of a posted invoice."""
    _ensure_positive(invoice_id, "invoice_id")
    models = _models()
    
    # Load invoice and verify it exists and is posted
    invoice = _fetch_single_record(
        models,
        "account.move",
        invoice_id,
        fields=["id", "name", "state"],
        not_found_detail=f"Invoice {invoice_id} not found",
    )
    
    invoice_state = invoice.get("state")
    if invoice_state != "posted":
        _http_error(
            status.HTTP_400_BAD_REQUEST,
            f"Invoice must be posted before downloading PDF. Current state: '{invoice_state}'",
        )
    
    invoice_number = invoice.get("name", f"invoice_{invoice_id}")
    
    # Use Odoo's HTTP API with session authentication
    import urllib.request
    import urllib.parse
    import json
    from http.cookiejar import CookieJar
    
    try:
        # Create cookie jar for session management
        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar),
            urllib.request.HTTPSHandler(context=SSL_CONTEXT)
        )
        
        # Step 1: Authenticate and get session cookie
        auth_url = f"{URL}/web/session/authenticate"
        auth_data = {
            "jsonrpc": "2.0",
            "params": {
                "db": DB,
                "login": USER,
                "password": PW,
            }
        }
        
        auth_request = urllib.request.Request(
            auth_url,
            data=json.dumps(auth_data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        with opener.open(auth_request) as auth_response:
            auth_result = json.loads(auth_response.read().decode('utf-8'))
            if 'error' in auth_result:
                _http_error(
                    status.HTTP_502_BAD_GATEWAY,
                    f"Odoo authentication failed: {auth_result['error']}"
                )
        
        # Step 2: Download PDF using the authenticated session
        report_url = f"{URL}/report/pdf/account.report_invoice/{invoice_id}"
        pdf_request = urllib.request.Request(report_url)
        
        with opener.open(pdf_request) as pdf_response:
            pdf_bytes = pdf_response.read()
            
            # Check if we got HTML (login page) instead of PDF
            if pdf_bytes.startswith(b'<!DOCTYPE') or pdf_bytes.startswith(b'<html'):
                _http_error(
                    status.HTTP_502_BAD_GATEWAY,
                    "Received HTML instead of PDF - authentication may have failed"
                )
            
            # Verify it's a valid PDF
            if not pdf_bytes.startswith(b'%PDF'):
                _http_error(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "Odoo returned invalid PDF content"
                )
            
            if len(pdf_bytes) < 100:
                _http_error(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "Odoo returned empty PDF content"
                )
        
    except urllib.error.HTTPError as err:
        error_body = err.read().decode('utf-8', errors='ignore') if err.fp else ""
        if err.code == 404:
            _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Invoice report not found: {error_body[:200]}"
            )
        elif err.code == 401 or err.code == 403:
            _http_error(
                status.HTTP_502_BAD_GATEWAY,
                f"Authentication failed with Odoo: {error_body[:200]}"
            )
        else:
            _http_error(
                status.HTTP_502_BAD_GATEWAY,
                f"Odoo error ({err.code}): {error_body[:200]}"
            )
    except Exception as err:
        _http_error(
            status.HTTP_502_BAD_GATEWAY,
            f"Error downloading PDF: {str(err)}"
        )
    
    # Return PDF as downloadable file
    filename = f"invoice_{invoice_number}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Run the server
# Command to run: `fastapi dev my_odoo_api.py`