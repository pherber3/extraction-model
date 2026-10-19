"""Synthetic typed-extraction documents (invoices, receipts, support emails, shipping notices).

Each underlying record is rendered through one randomly chosen surface template
(label synonyms, line/sentence permutation, date/money formats vary per document).
Every example is validated before it is returned:

- each ``extract`` span ``state[s:e]`` normalizes (``extraction.normalize``) to the gold value,
- each ``null`` field's hypothetical value is not mentioned anywhere in the document,
- each ``compute`` field's value is derived exactly from values printed in the document.

Output follows the shared JSONL example contract (see ``generate``).
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from extraction.normalize import find_dates, normalize_value, parse_number, text_equal

DOC_TYPES: tuple[str, ...] = ("invoice", "receipt", "support_email", "shipping_notice")
_ID_PREFIX = {"invoice": "inv", "receipt": "rcp", "support_email": "eml", "shipping_notice": "shp"}

P_NULL = 0.28
P_COMPUTE = 0.25
MIN_FIELDS, MAX_FIELDS = 4, 9

# --------------------------------------------------------------------------- vocab

FIRST = [
    "Dana", "Marcus", "Priya", "Tomás", "Aiko", "Fatima", "Liam", "Sofia", "Chen", "Olu",
    "Hannah", "Diego", "Ingrid", "Rahul", "Maya", "Kwame", "Elena", "Jonas", "Leila", "Owen",
    "Grace", "Mateo", "Yuki", "Noor", "Felix", "Amara", "Ravi", "Clara", "Samir", "Beatriz",
]
LAST = [
    "Whitfield", "Okafor", "Nakamura", "Patel", "García", "Lindqvist", "Moreau", "Kowalski",
    "Haddad", "Brennan", "Achebe", "Rossi", "Schmidt", "Delgado", "Fischer", "Novak", "Iyer",
    "Murphy", "Sato", "Andersen", "Mbeki", "Costa", "Larsen", "Kaplan", "Dubois", "Reyes",
    "Hughes", "Zhang", "Petrov", "Quinn",
]
VENDORS = [
    "Northwind Supply Co.", "Brightline Design LLC", "Kestrel Logistics", "Ostara Software GmbH",
    "Harbor & Pine Consulting", "Bluefin Office Solutions", "Meridian Print Works",
    "Cobalt Data Services", "Tallgrass Facilities Ltd", "Ironwood Legal Partners",
    "Juniper Analytics Inc.", "Summit Electrical", "Lumen Creative Studio", "Granite Peak IT",
    "Willow & Ash Catering",
]
CLIENTS = [
    "Acme Robotics", "Fairview Dental Group", "Pinecrest Academy", "Redwood Property Management",
    "Sable & Finch Architects", "Orchard Street Bakery", "Tidewater Marine",
    "Lakeside Veterinary Clinic", "Copperleaf Hotels", "Starling Media",
]
INVOICE_ITEMS = [
    "Website maintenance", "Consulting hours", "Logo redesign", "Cloud hosting, standard plan",
    "On-site IT support", "Printed brochures", "Annual software license", "Data migration",
    "Office chairs", "Toner cartridges", "Network audit", "Catering for team event",
    "Copywriting services", "Security camera installation", "Payroll processing", "SEO audit",
    "Monthly bookkeeping", "Conference room rental", "Laptop docking stations",
    "Translation services",
]
MERCHANTS = [
    "Blue Heron Coffee", "Greenleaf Market", "Hardware Hub", "Sunrise Bakery", "Metro Pharmacy",
    "Copper Kettle Café", "Fresh Fields Grocery", "Pages & Co. Books", "Tidal Taqueria",
    "Northside Deli", "Paws Pet Supply", "Lantern Noodle Bar",
]
RECEIPT_ITEMS = [
    "Oat milk latte", "Blueberry muffin", "Bananas", "Sourdough loaf", "AA batteries",
    "Paint roller", "Ibuprofen", "Greek yogurt", "Avocados", "Dog treats", "Spiral notebook",
    "Iced tea", "Chicken burrito", "Chips & salsa", "Masking tape", "Olive oil",
    "Paperback novel", "Hand soap", "Ramen bowl", "Sparkling water",
]
PRODUCTS = [
    "Pixel tablet", "Nimbus air purifier", "Vertex standing desk", "Echo Pro headphones",
    "Halo smart speaker", "Strata espresso machine", "Orbit robot vacuum", "Lumen desk lamp",
    "Glide office chair", "Trek carry-on suitcase", "Canvas e-reader", "Pulse fitness watch",
    "Atlas mechanical keyboard", "Breeze tower fan",
]
RETAILERS = [
    "Brightcart", "Northbay Outfitters", "Gadgetry", "HomeNest", "Peak & Trail", "Loop Electronics",
    "Maple Street Goods", "Urbanist Supply",
]
DAMAGES = [
    "cracked screen", "dented corner", "scratched casing", "missing power cable",
    "torn packaging", "water damage", "broken hinge", "shattered glass panel", "bent frame",
    "loose buttons", "dead pixels", "stained fabric",
]
RESOLUTIONS = ["refund", "full refund", "replacement", "exchange", "store credit", "partial refund"]
CITIES = [
    "Portland, OR", "Austin, TX", "Denver, CO", "Columbus, OH", "Raleigh, NC", "Madison, WI",
    "Tucson, AZ", "Boise, ID", "Leeds", "Bristol", "Manchester", "Lyon", "Rotterdam",
    "Toronto, ON", "Vancouver, BC",
]
STREETS = ["Main", "Oak", "Harbor", "Elm", "Cedar", "Mill", "Station", "Bridge", "Park", "Lake"]
STREET_KINDS = ["St", "Ave", "Road", "Blvd", "Lane"]
CARDS = ["Visa", "Mastercard", "Amex", "Discover", "Debit"]

_MON_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MON_FULL = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --------------------------------------------------------------------------- field catalog

_MONEY = {"decimals": 2}
_DAYS = {"unit": "days", "integer": True, "decimals": 0}
_COUNT = {"integer": True, "decimals": 0}
_LIST = {"items": "text"}

# name -> (type, static extras); money fields get the document currency as `unit` at render time.
CATALOG: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {
    "invoice": {
        "invoice_number": ("text", {}),
        "vendor_name": ("text", {}),
        "vendor_email": ("text", {}),
        "customer_name": ("text", {}),
        "issue_date": ("date", {}),
        "due_date": ("date", {}),
        "days_until_due": ("duration", _DAYS),
        "po_number": ("text", {}),
        "subtotal": ("number", _MONEY),
        "discount": ("number", _MONEY),
        "tax": ("number", _MONEY),
        "shipping": ("number", _MONEY),
        "total": ("number", _MONEY),
        "line_items": ("list", _LIST),
        "item_count": ("number", _COUNT),
    },
    "receipt": {
        "merchant_name": ("text", {}),
        "receipt_number": ("text", {}),
        "transaction_date": ("date", {}),
        "transaction_time": ("time", {}),
        "items": ("list", _LIST),
        "subtotal": ("number", _MONEY),
        "tax": ("number", _MONEY),
        "tip": ("number", _MONEY),
        "total": ("number", _MONEY),
        "payment_method": ("text", {}),
        "card_last_four": ("text", {}),
        "cashier_name": ("text", {}),
        "item_count": ("number", _COUNT),
    },
    "support_email": {
        "order_number": ("text", {}),
        "customer_name": ("text", {}),
        "customer_email": ("text", {}),
        "phone_number": ("text", {}),
        "product_name": ("text", {}),
        "order_date": ("date", {}),
        "delivery_date": ("date", {}),
        "price_paid": ("number", _MONEY),
        "tax": ("number", _MONEY),
        "shipping_cost": ("number", _MONEY),
        "total_paid": ("number", _MONEY),
        "damage_types": ("list", _LIST),
        "requested_resolution": ("text", {}),
        "days_to_delivery": ("duration", _DAYS),
        "tracking_number": ("text", {}),
    },
    "shipping_notice": {
        "tracking_number": ("text", {}),
        "carrier": ("text", {}),
        "order_number": ("text", {}),
        "retailer_name": ("text", {}),
        "recipient_name": ("text", {}),
        "destination_city": ("text", {}),
        "ship_date": ("date", {}),
        "estimated_delivery": ("date", {}),
        "package_weight": ("number", {"decimals": 1}),
        "package_count": ("number", _COUNT),
        "shipping_cost": ("number", _MONEY),
        "items": ("list", _LIST),
        "transit_days": ("duration", _DAYS),
        "days_until_delivery": ("duration", _DAYS),
    },
}

INSTRUCTIONS: dict[str, list[str]] = {
    "invoice_number": ["The invoice number.", "Invoice ID or reference number.", "Identifier printed on the invoice."],
    "vendor_name": ["Name of the company issuing the invoice.", "Who sent the invoice (vendor/supplier).", "The seller's business name."],
    "vendor_email": ["Vendor's billing email address.", "Email address to contact the issuer.", "The supplier's contact email."],
    "customer_name": ["Who is being billed.", "Name of the customer on the document.", "The billed party's name."],
    "issue_date": ["Date the invoice was issued.", "Invoice date.", "When the invoice was created."],
    "due_date": ["Date payment is due.", "Payment due date.", "Deadline for paying the invoice."],
    "days_until_due": ["Number of days from the invoice date to the due date.", "Payment term length in days.", "How many days the customer has to pay."],
    "po_number": ["The purchase order (PO) number.", "Customer's PO reference.", "Purchase order number, if any."],
    "subtotal": ["Subtotal before tax and other charges.", "Pre-tax subtotal.", "Sum of the line items before adjustments."],
    "discount": ["Discount amount applied.", "Total discount, as a positive amount.", "How much was taken off the price."],
    "tax": ["Tax amount charged.", "Sales tax / VAT amount.", "Amount of tax."],
    "shipping": ["Shipping or delivery charge.", "Freight / shipping fee.", "Cost of shipping."],
    "total": ["Final total amount.", "Grand total to be paid.", "Total amount including tax and charges."],
    "line_items": ["Names of the billed line items, in order.", "List the item descriptions.", "What products or services are listed."],
    "item_count": ["Number of distinct line items.", "How many items are listed.", "Count of line items."],
    "merchant_name": ["Store or merchant name.", "Where the purchase was made.", "Name of the business on the receipt."],
    "receipt_number": ["Receipt or transaction number.", "Transaction ID.", "Receipt reference number."],
    "transaction_date": ["Date of the purchase.", "Transaction date.", "When the purchase happened (date)."],
    "transaction_time": ["Time of the purchase.", "Transaction time of day.", "When the purchase happened (time)."],
    "items": ["Names of the items, in order.", "List the purchased or shipped items.", "Item names listed on the document."],
    "tip": ["Tip / gratuity amount.", "Gratuity added.", "How much was tipped."],
    "payment_method": ["How the purchase was paid (card brand or cash).", "Payment method.", "Payment type used."],
    "card_last_four": ["Last four digits of the card used.", "Card number ending.", "Final 4 digits of the payment card."],
    "cashier_name": ["Name of the cashier.", "Who served the customer.", "Cashier or server name."],
    "order_number": ["The order number.", "Order ID or reference.", "Customer order identifier."],
    "customer_email": ["Customer's email address.", "Sender's email.", "Email of the person writing."],
    "phone_number": ["Customer's phone number.", "Contact phone number.", "Phone number given by the customer."],
    "product_name": ["Which product the message is about.", "Name of the item ordered.", "The product that was damaged."],
    "order_date": ["Date the order was placed.", "When the customer ordered.", "Order placement date."],
    "delivery_date": ["Date the order arrived.", "Delivery date.", "When the package was received."],
    "price_paid": ["Price paid for the item, excluding tax and shipping.", "Item price.", "How much the product itself cost."],
    "shipping_cost": ["Shipping cost paid.", "Delivery / postage charge.", "How much shipping cost."],
    "total_paid": ["Total amount the customer paid, including tax and shipping.", "All-in amount charged.", "Total charged to the customer."],
    "damage_types": ["Kinds of damage described, in order.", "List each problem reported with the item.", "Damage the customer mentions."],
    "requested_resolution": ["What the customer asks for (refund, replacement, ...).", "Requested remedy.", "Resolution the customer wants."],
    "days_to_delivery": ["Days between ordering and delivery.", "How many days the order took to arrive.", "Number of days from order date to delivery date."],
    "tracking_number": ["Shipment tracking number.", "Tracking ID for the package.", "Carrier tracking number."],
    "carrier": ["Shipping carrier.", "Which courier carries the package.", "Delivery company name."],
    "retailer_name": ["Store that sent the shipment.", "Retailer / sender name.", "Who the order is from."],
    "recipient_name": ["Name of the recipient.", "Who the package is addressed to.", "Ship-to name."],
    "destination_city": ["Destination city.", "City the package is going to.", "Ship-to city."],
    "ship_date": ["Date the package shipped.", "Dispatch date.", "When the shipment left."],
    "estimated_delivery": ["Estimated delivery date.", "Expected arrival date.", "When the package should arrive."],
    "package_weight": ["Total package weight.", "Shipment weight.", "How heavy the shipment is."],
    "package_count": ["Number of packages.", "How many parcels are in the shipment.", "Package/piece count."],
    "transit_days": ["Days between ship date and estimated delivery.", "Expected transit time in days.", "How many days the shipment is in transit."],
    "days_until_delivery": ["Days from the reference date until the estimated delivery.", "How many days until the package is expected (relative to today).", "Days remaining until estimated arrival from the reference date."],
}

# null field -> fields whose presence makes a convincing decoy
DECOYS: dict[str, dict[str, tuple[str, ...]]] = {
    "invoice": {
        "discount": ("tax", "shipping"), "po_number": ("invoice_number",),
        "shipping": ("tax", "discount"), "tax": ("discount", "shipping"),
        "due_date": ("issue_date",), "days_until_due": ("issue_date",),
        "vendor_email": ("vendor_name",),
    },
    "receipt": {
        "tip": ("tax",), "tax": ("tip",), "card_last_four": ("receipt_number",),
        "cashier_name": ("merchant_name",), "transaction_time": ("transaction_date",),
        "payment_method": ("total",),
    },
    "support_email": {
        "tracking_number": ("order_number",), "order_number": ("tracking_number",),
        "tax": ("price_paid", "shipping_cost"), "shipping_cost": ("tax",),
        "order_date": ("delivery_date",), "delivery_date": ("order_date",),
        "phone_number": ("order_number",), "customer_email": ("customer_name",),
        "requested_resolution": ("damage_types",),
    },
    "shipping_notice": {
        "order_number": ("tracking_number",), "tracking_number": ("order_number",),
        "estimated_delivery": ("ship_date",), "ship_date": ("estimated_delivery",),
        "package_weight": ("package_count",), "package_count": ("package_weight",),
        "shipping_cost": ("package_weight",),
    },
}

# --------------------------------------------------------------------------- formatting


def _fmt_date(d: dt.date, style: str, ref: dt.date) -> str:
    if style in ("month_noyear", "ordinal_noyear", "weekday_noyear") and d.year != ref.year:
        style = "month_full"
    if style == "mdy_slash" and d.day <= 12:
        style = "mon_abbr"
    mon, full = _MON_ABBR[d.month - 1], _MON_FULL[d.month - 1]
    if style == "iso":
        return d.isoformat()
    if style == "mon_abbr":
        return f"{mon} {d.day}, {d.year}"
    if style == "mon_abbr4":
        return f"{'Sept' if d.month == 9 else mon} {d.day}, {d.year}"
    if style == "month_full":
        return f"{full} {d.day}, {d.year}"
    if style == "day_month":
        return f"{d.day} {full} {d.year}"
    if style == "day_month_abbr":
        return f"{d.day:02d} {mon} {d.year}"
    if style == "dmy_slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    if style == "dmy_dot":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    if style == "mdy_slash":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "month_noyear":
        return f"{full} {d.day}"
    if style == "ordinal_noyear":
        return f"{full} {d.day}{_ordinal(d.day)}"
    if style == "weekday_noyear":
        return f"{_WEEKDAYS[d.weekday()][:3]}, {'Sept' if d.month == 9 else mon} {d.day}"
    if style == "weekday_full":
        return f"{_WEEKDAYS[d.weekday()]}, {full} {d.day}, {d.year}"
    raise ValueError(style)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


_CURRENCIES = (("USD", 0.6), ("EUR", 0.2), ("GBP", 0.1), ("CAD", 0.1))
_MONEY_VARIANTS = {
    "USD": ("sym_en", "sym_en", "sym_en", "code_pre_en", "code_post_en", "code_pre_plain", "sym_plain"),
    "EUR": ("sym_en", "code_post_eu", "sym_post_eu", "code_pre_eu", "code_pre_en"),
    "GBP": ("sym_en", "sym_en", "code_pre_en"),
    "CAD": ("sym_en", "code_pre_en", "code_post_en"),
}
_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$"}


def _fmt_money(cents: int, style: tuple[str, str]) -> str:
    cur, var = style
    whole, frac = divmod(cents, 100)
    en = f"{whole:,}.{frac:02d}"
    eu = f"{whole:,}".replace(",", ".") + f",{frac:02d}"
    plain = f"{whole}" if frac == 0 else f"{whole}.{frac:02d}"
    sym = _SYMBOLS[cur]
    return {
        "sym_en": f"{sym}{en}",
        "sym_plain": f"{sym}{plain}",
        "code_pre_en": f"{cur} {en}",
        "code_post_en": f"{en} {cur}",
        "code_pre_plain": f"{cur} {plain}",
        "code_post_eu": f"{eu} {cur}",
        "sym_post_eu": f"{eu} {sym}",
        "code_pre_eu": f"{cur} {eu}",
    }[var]


def _fmt_time(t: dt.time, style: str) -> str:
    if style == "24h":
        return f"{t.hour:02d}:{t.minute:02d}"
    h = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"{h}:{t.minute:02d} {suffix}" if style == "12h" else f"{h}:{t.minute:02d}{suffix.lower()}"


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _ascii(s).lower()).strip("-")


def _cents(x: float) -> float:
    return round(x / 100, 2)


# --------------------------------------------------------------------------- document builder

Part = str | tuple[str, str]


class _Doc:
    """Appends text pieces; tuple parts ``(key, surface)`` record their char span under ``key``."""

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._pos = 0
        self.spans: dict[str, tuple[int, int]] = {}

    def add(self, *parts: Part) -> None:
        for p in parts:
            if isinstance(p, tuple):
                key, p = p
                if key in self.spans:
                    raise AssertionError(f"span {key!r} rendered twice")
                self.spans[key] = (self._pos, self._pos + len(p))
            self._buf.append(p)
            self._pos += len(p)

    def line(self, *parts: Part) -> None:
        self.add(*parts, "\n")

    def text(self) -> str:
        return "".join(self._buf)


def _series(key: str, surfaces: Sequence[str], conj: str | None = "and", sep: str = ", ") -> list[Part]:
    """Spanned list "a, b and c" (``conj=None`` joins every item with ``sep``)."""
    parts: list[Part] = []
    n = len(surfaces)
    for i, s in enumerate(surfaces):
        if i:
            parts.append(f" {conj} " if conj and i == n - 1 else sep)
        parts.append((f"{key}#{i}", s))
    return parts


def _flat(chunks: Iterable[Sequence[Part]], sep: str) -> list[Part]:
    out: list[Part] = []
    for i, c in enumerate(chunks):
        if i:
            out.append(sep)
        out.extend(c)
    return out


# --------------------------------------------------------------------------- record helpers


class _Rec(dict):
    """Record: rendered values plus ``omitted`` hypothetical values of absent optional fields."""

    def __init__(self) -> None:
        super().__init__()
        self.omitted: dict[str, tuple[str, Any]] = {}

    def opt(self, key: str, present: bool, value: Any, check: tuple[str, Any]) -> None:
        self[key] = value if present else None
        if not present:
            self.omitted[key] = check


def _rand_date(rng: random.Random) -> dt.date:
    return dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(730))


def _person(rng: random.Random) -> tuple[str, str]:
    return rng.choice(FIRST), rng.choice(LAST)


def _street(rng: random.Random) -> str:
    return f"{rng.randint(1, 990)} {rng.choice(STREETS)} {rng.choice(STREET_KINDS)}"


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _invoice_number(rng: random.Random, year: int) -> str:
    return rng.choice([
        lambda: f"INV-{year}-{_digits(rng, 5)}",
        lambda: f"INV{_digits(rng, 6)}",
        lambda: f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}-{_digits(rng, 5)}",
        lambda: f"{year}/{_digits(rng, 4)}",
        lambda: f"BL{_digits(rng, 7)}",
    ])()


def _po_number(rng: random.Random) -> str:
    return rng.choice([
        lambda: f"PO-{_digits(rng, 5)}",
        lambda: f"45{_digits(rng, 8)}",
        lambda: f"PO{_digits(rng, 6)}",
        lambda: f"P{_digits(rng, 4)}-{rng.choice('ABCDEFGH')}{rng.choice('KLMNPQRS')}",
    ])()


def _order_number(rng: random.Random) -> str:
    return rng.choice([
        lambda: f"A-{_digits(rng, 5)}",
        lambda: f"ORD-{_digits(rng, 7)}",
        lambda: f"{_digits(rng, 3)}-{_digits(rng, 7)}-{_digits(rng, 7)}",
        lambda: f"W{_digits(rng, 9)}",
        lambda: f"SO{_digits(rng, 6)}",
    ])()


def _receipt_number(rng: random.Random) -> str:
    return rng.choice([
        lambda: f"{_digits(rng, 4)}-{_digits(rng, 3)}-{_digits(rng, 4)}",
        lambda: f"R{_digits(rng, 7)}",
        lambda: f"TRX-{_digits(rng, 6)}",
        lambda: _digits(rng, 6),
    ])()


def _tracking(rng: random.Random, carrier: str) -> str:
    alnum = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    if carrier == "UPS":
        return "1Z" + "".join(rng.choice(alnum) for _ in range(6)) + _digits(rng, 10)
    if carrier == "FedEx":
        return _digits(rng, 12)
    if carrier == "USPS":
        return "9400" + _digits(rng, 18)
    if carrier == "DHL":
        return _digits(rng, 10)
    if carrier == "Royal Mail":
        return f"{rng.choice('ABCDEFGHJK')}{rng.choice('ABCDEFGHJK')}{_digits(rng, 9)}GB"
    return "3S" + "".join(rng.choice(alnum) for _ in range(11))


_CARRIERS = ("UPS", "FedEx", "USPS", "DHL", "Royal Mail", "PostNL")


def _phone(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return f"({rng.randint(201, 989)}) 555-{_digits(rng, 4)}"
    return f"+44 7700 900{_digits(rng, 3)}"


def _money_style(rng: random.Random) -> tuple[str, str]:
    cur = rng.choices([c for c, _ in _CURRENCIES], [w for _, w in _CURRENCIES])[0]
    return cur, rng.choice(_MONEY_VARIANTS[cur])


def _num_check(cents: int) -> tuple[str, Any]:
    return ("number", _cents(cents))


# --------------------------------------------------------------------------- invoice


def _rec_invoice(rng: random.Random) -> _Rec:
    r = _Rec()
    issue = _rand_date(rng)
    r["issue"] = issue
    r["ref"] = issue + dt.timedelta(days=rng.randint(0, 20))
    r["vendor"] = rng.choice(VENDORS)
    r["vendor_street"], r["vendor_city"] = _street(rng), rng.choice(CITIES)
    email = f"{rng.choice(['billing', 'accounts', 'invoices', 'ar'])}@{_slug(r['vendor'])}.com"
    r.opt("vendor_email", rng.random() < 0.5, email, ("text", email))
    r["customer"] = rng.choice(CLIENTS) if rng.random() < 0.65 else " ".join(_person(rng))
    r["invoice_number"] = _invoice_number(rng, issue.year)
    po = _po_number(rng)
    r.opt("po_number", rng.random() < 0.5, po, ("text", po))
    names = rng.sample(INVOICE_ITEMS, rng.randint(1, 5))
    r["items"] = [(n, rng.randint(1, 12), rng.randint(15_00, 2400_00)) for n in names]
    sub = sum(q * u for _, q, u in r["items"])
    r["subtotal"] = sub
    if rng.random() < 0.5:
        pct = rng.choice([5, 10, 15])
        disc, disc_label = round(sub * pct / 100), f"{pct}%"
    else:
        disc, disc_label = min(rng.choice([25_00, 50_00, 100_00]), sub // 2), None
    r["disc_label"] = disc_label
    r.opt("discount", rng.random() < 0.3, disc, _num_check(disc))
    rate = rng.choice([5, 7.25, 8, 8.25, 10, 20])
    r["tax_rate"] = rate
    tax = round((sub - (r["discount"] or 0)) * rate / 100)
    r.opt("tax", rng.random() < 0.7, tax, _num_check(tax))
    ship = rng.randint(8_00, 120_00)
    r.opt("shipping", rng.random() < 0.3, ship, _num_check(ship))
    r["total"] = sub - (r["discount"] or 0) + (r["tax"] or 0) + (r["shipping"] or 0)
    adjusted = any(r[k] is not None for k in ("discount", "tax", "shipping"))
    r["total_printed"] = rng.random() < 0.7 or not adjusted
    terms = rng.choice([7, 10, 14, 15, 30, 45, 60, 90])
    due = issue + dt.timedelta(days=terms)
    mode = rng.choices(["date", "terms", "both", "none"], [0.4, 0.3, 0.15, 0.15])[0]
    r["terms"] = terms if mode in ("terms", "both") else None
    r["due"] = due if mode in ("date", "both") else None
    r["due_known"] = mode != "none"
    if mode == "none":
        r.omitted["due_date"] = ("date", due)
        r.omitted["days_until_due"] = ("phrases", [f"net {terms}", f"{terms} days"])
    r["money"] = _money_style(rng)
    r["date_style"] = rng.choice(
        ["iso", "mon_abbr", "mon_abbr4", "month_full", "day_month", "dmy_slash", "dmy_dot", "day_month_abbr"]
    )
    return r


def _inv_amount_lines(r: _Rec, rng: random.Random, labels: dict[str, list[str]]) -> list[list[Part]]:
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    mid: list[list[Part]] = []
    if r["discount"] is not None:
        lab = rng.choice(labels["discount"])
        if r["disc_label"]:
            lab = f"{lab} ({r['disc_label']})"
        mid.append([f"{lab}: -", ("discount", M(r["discount"]))])
    if r["tax"] is not None:
        lab = rng.choice(labels["tax"])
        mid.append([f"{lab} ({r['tax_rate']}%): ", ("tax", M(r["tax"]))])
    if r["shipping"] is not None:
        mid.append([f"{rng.choice(labels['shipping'])}: ", ("shipping", M(r["shipping"]))])
    rng.shuffle(mid)
    lines = [[f"{rng.choice(labels['subtotal'])}: ", ("subtotal", M(r["subtotal"]))], *mid]
    if r["total_printed"]:
        lines.append([f"{rng.choice(labels['total'])}: ", ("total", M(r["total"]))])
    return lines


_INV_LABELS = {
    "discount": ["Discount", "Less discount", "Promotional discount"],
    "tax": ["Tax", "Sales tax", "VAT"],
    "shipping": ["Shipping", "Delivery", "Freight"],
    "subtotal": ["Subtotal", "Sub-total", "Net amount"],
    "total": ["Total", "Total due", "Amount due", "Balance due", "TOTAL"],
}


def _terms_part(r: _Rec, rng: random.Random) -> list[Part]:
    n = str(r["terms"])
    return rng.choice([
        ["Terms: Net ", ("terms_days", n)],
        ["Payment terms: ", ("terms_days", f"{n} days")],
        ["Payable within ", ("terms_days", f"{n} days"), " of invoice date"],
    ])


def _inv_t0(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    d.line(("vendor", r["vendor"]))
    d.line(r["vendor_street"])
    d.line(r["vendor_city"])
    if r["vendor_email"]:
        d.line(rng.choice(["Email: ", "", "Billing contact: "]), ("vendor_email", r["vendor_email"]))
    d.line()
    d.line(rng.choice(["INVOICE", "Invoice", "TAX INVOICE"]))
    d.line()
    meta: list[list[Part]] = [
        [rng.choice(["Invoice No: ", "Invoice #: ", "Invoice Number: ", "Inv. No. "]), ("invoice_number", r["invoice_number"])],
        [rng.choice(["Date: ", "Invoice Date: ", "Issued: "]), ("issue_date", D(r["issue"]))],
    ]
    if r["due"]:
        meta.append([rng.choice(["Due Date: ", "Payment Due: ", "Due: "]), ("due_date", D(r["due"]))])
    if r["terms"]:
        meta.append(_terms_part(r, rng))
    if r["po_number"]:
        meta.append([rng.choice(["PO Number: ", "PO #: ", "Purchase Order: "]), ("po_number", r["po_number"])])
    rng.shuffle(meta)
    for m in meta:
        d.line(*m)
    d.line()
    d.line(rng.choice(["Bill To:", "Billed To:", "Customer:"]))
    d.line(("customer", r["customer"]))
    d.line()
    d.line(f"{'Description':<34}{'Qty':>5}  {'Unit Price':>14}  {'Amount':>14}")
    for i, (name, q, u) in enumerate(r["items"]):
        d.line(("line_items#%d" % i, name), " " * (34 - len(name)), f"{q:>5}  {M(u):>14}  {M(q * u):>14}")
    d.line()
    for line in _inv_amount_lines(r, rng, _INV_LABELS):
        d.line(*line)
    d.add(rng.choice(["Thank you for your business.", "Please remit payment by bank transfer.", ""]))


def _inv_t1(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    d.line(rng.choice(["Hi ", "Hello ", "Dear "]), ("customer", r["customer"]), ",")
    d.line()
    d.line(
        rng.choice(["Please find attached invoice ", "Attached is invoice ", "Here is invoice "]),
        ("invoice_number", r["invoice_number"]), " from ", ("vendor", r["vendor"]),
        rng.choice([", issued on ", ", dated "]), ("issue_date", D(r["issue"])), ".",
    )
    sentences: list[list[Part]] = []
    names = [n for n, _, _ in r["items"]]
    sentences.append([rng.choice(["This invoice covers ", "Billed items: ", "It includes "]),
                      *_series("line_items", names), "."])
    amt: list[Part] = ["The subtotal is ", ("subtotal", M(r["subtotal"]))]
    if r["discount"] is not None:
        amt += [", less a discount of ", ("discount", M(r["discount"]))]
    if r["tax"] is not None:
        amt += [", plus ", ("tax", M(r["tax"])), " in tax"]
    if r["shipping"] is not None:
        amt += [" and ", ("shipping", M(r["shipping"])), " for shipping"]
    amt.append(".")
    if r["total_printed"]:
        amt += [rng.choice([" The total amount due is ", " Total due: ", " Altogether that comes to "]),
                ("total", M(r["total"])), "."]
    sentences.append(amt)
    if r["due"] and r["terms"]:
        sentences.append(["Payment is due within ", ("terms_days", f"{r['terms']} days"), ", i.e. by ",
                          ("due_date", D(r["due"])), "."])
    elif r["due"]:
        sentences.append([rng.choice(["Payment is due by ", "Kindly pay by ", "The due date is "]),
                          ("due_date", D(r["due"])), "."])
    elif r["terms"]:
        sentences.append(["Our terms are net ", ("terms_days", str(r["terms"])), " days."])
    if r["po_number"]:
        sentences.append(["This invoice references your purchase order ", ("po_number", r["po_number"]), "."])
    rng.shuffle(sentences)
    for s in sentences:
        d.line(*s)
    if r["vendor_email"]:
        d.line("If you have any questions, reply to ", ("vendor_email", r["vendor_email"]), ".")
    d.line()
    d.add(rng.choice(["Kind regards,\nAccounts team", "Best,\nBilling department", "Thanks,\nFinance"]))


def _inv_t2(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    d.line(("vendor", r["vendor"]), " | INVOICE ", ("invoice_number", r["invoice_number"]))
    segs: list[list[Part]] = []
    if r["due"]:
        segs.append(["Due ", ("due_date", D(r["due"]))])
    if r["terms"]:
        segs.append(["Net ", ("terms_days", str(r["terms"]))])
    if r["po_number"]:
        segs.append(["PO ", ("po_number", r["po_number"])])
    rng.shuffle(segs)
    d.line(*_flat([["Date ", ("issue_date", D(r["issue"]))], *segs], " | "))
    cust: list[list[Part]] = [["To: ", ("customer", r["customer"])]]
    if r["vendor_email"]:
        cust.append(["Questions: ", ("vendor_email", r["vendor_email"])])
    d.line(*_flat(cust, " | "))
    d.line("Items: ", *_series("line_items", [n for n, _, _ in r["items"]], conj=None, sep="; "))
    d.line(*_flat(_inv_amount_lines(r, rng, _INV_LABELS), " | "))


def _inv_t3(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    d.line("BILLED TO")
    d.line(("customer", r["customer"]))
    d.line()
    d.line("FROM")
    if r["vendor_email"]:
        d.line(("vendor", r["vendor"]), " · ", ("vendor_email", r["vendor_email"]))
    else:
        d.line(("vendor", r["vendor"]))
    d.line()
    if r["total_printed"]:
        d.line(rng.choice(["AMOUNT DUE: ", "BALANCE: ", "PLEASE PAY: "]), ("total", M(r["total"])))
    if r["due"]:
        d.line("DUE DATE: ", ("due_date", D(r["due"])))
    d.line()
    refs: list[list[Part]] = [["Reference: ", ("invoice_number", r["invoice_number"])],
                              ["Issued: ", ("issue_date", D(r["issue"]))]]
    if r["po_number"]:
        refs.append(["Your PO: ", ("po_number", r["po_number"])])
    if r["terms"]:
        refs.append(["Terms: ", ("terms_days", f"{r['terms']} days")])
    rng.shuffle(refs)
    d.line(*_flat(refs, "    "))
    d.line()
    d.line("Line items:")
    for i, (name, q, u) in enumerate(r["items"]):
        d.line(f" {i + 1}. ", ("line_items#%d" % i, name), f" — {q} × {M(u)} = {M(q * u)}")
    d.line()
    labels = {**_INV_LABELS, "shipping": ["Shipping & handling", "Postage"]}
    lines = [ln for ln in _inv_amount_lines(r, rng, labels) if ln[1][0] != "total"]
    for line in lines:
        d.line(*line)


def _gold_invoice(r: _Rec, g: _Gold) -> None:
    g.ext("invoice_number", "invoice_number", r["invoice_number"])
    g.ext("vendor_name", "vendor", r["vendor"])
    g.opt_ext("vendor_email", "vendor_email", r["vendor_email"])
    g.ext("customer_name", "customer", r["customer"])
    g.ext("issue_date", "issue_date", r["issue"].isoformat())
    due = r["issue"] + dt.timedelta(days=r["terms"]) if r["terms"] else r["due"]
    if r["due"]:
        g.ext("due_date", "due_date", r["due"].isoformat())
    elif r["terms"]:
        g.comp("due_date", due.isoformat())
    else:
        g.null("due_date")
    if r["terms"]:
        g.ext("days_until_due", "terms_days", r["terms"])
    elif r["due"]:
        g.comp("days_until_due", (r["due"] - r["issue"]).days)
    else:
        g.null("days_until_due")
    g.opt_ext("po_number", "po_number", r["po_number"])
    g.ext("subtotal", "subtotal", _cents(r["subtotal"]))
    for k in ("discount", "tax", "shipping"):
        g.opt_ext(k, k, None if r[k] is None else _cents(r[k]))
    if r["total_printed"]:
        g.ext("total", "total", _cents(r["total"]))
    else:
        g.comp("total", _cents(r["total"]))
    g.lst("line_items", "line_items", [n for n, _, _ in r["items"]])
    g.comp("item_count", len(r["items"]))


# --------------------------------------------------------------------------- receipt


def _rec_receipt(rng: random.Random) -> _Rec:
    r = _Rec()
    date = _rand_date(rng)
    r["date"], r["ref"] = date, date + dt.timedelta(days=rng.randint(0, 5))
    r["merchant"] = rng.choice(MERCHANTS)
    r["street"], r["city"] = _street(rng), rng.choice(CITIES)
    r["receipt_number"] = _receipt_number(rng)
    t = dt.time(rng.randint(7, 21), rng.randint(0, 59))
    r["time_style"] = rng.choice(["24h", "12h", "12h_lower"])
    r.opt("time", rng.random() < 0.8, t, ("phrases", [_fmt_time(t, s) for s in ("24h", "12h", "12h_lower")]))
    names = rng.sample(RECEIPT_ITEMS, rng.randint(1, 6))
    r["items"] = [(n, rng.randint(99, 49_99)) for n in names]
    sub = sum(p for _, p in r["items"])
    r["subtotal"] = sub
    rate = rng.choice([5, 6.5, 8, 8.875, 9.5, 10])
    r["tax_rate"] = rate
    tax = round(sub * rate / 100)
    r.opt("tax", rng.random() < 0.75, tax, _num_check(tax))
    tip = round(sub * rng.choice([15, 18, 20]) / 100)
    r.opt("tip", rng.random() < 0.3, tip, _num_check(tip))
    r["total"] = sub + (r["tax"] or 0) + (r["tip"] or 0)
    adjusted = r["tax"] is not None or r["tip"] is not None
    r["total_printed"] = rng.random() < 0.8 or not adjusted
    method = rng.choice(CARDS + ["Cash"])
    r.opt("method", rng.random() < 0.85, method, ("phrases", [method]))
    last4 = _digits(rng, 4)
    r.opt("last4", r["method"] not in (None, "Cash") and rng.random() < 0.9, last4, ("text", last4))
    r["card_mask"] = rng.choice([" ****", " ending in ", " xxxx-", " ************"])
    cashier = rng.choice(FIRST)
    r.opt("cashier", rng.random() < 0.45, cashier, ("text", cashier))
    r["money"] = _money_style(rng)
    r["date_style"] = rng.choice(["iso", "mdy_slash", "dmy_slash", "dmy_dot", "mon_abbr", "day_month_abbr"])
    return r


def _rcp_payment(r: _Rec, upper: bool) -> list[Part]:
    method = r["method"].upper() if upper else r["method"]
    if r["last4"]:
        return [("method", method), r["card_mask"], ("last4", r["last4"])]
    return [("method", method)]


def _rcp_amounts(r: _Rec, rng: random.Random, upper: bool, width: int) -> list[list[Part]]:
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    lab = lambda s: (s.upper() if upper else s).ljust(width)  # noqa: E731
    lines: list[list[Part]] = [[lab("Subtotal"), ("subtotal", M(r["subtotal"]))]]
    if r["tax"] is not None:
        lines.append([lab(rng.choice(["Tax", f"Tax {r['tax_rate']}%", "Sales Tax"])), ("tax", M(r["tax"]))])
    if r["tip"] is not None:
        lines.append([lab(rng.choice(["Tip", "Gratuity"])), ("tip", M(r["tip"]))])
    if r["total_printed"]:
        lines.append([lab(rng.choice(["Total", "Amount Paid", "Balance"])), ("total", M(r["total"]))])
    return lines


def _rcp_t0(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    upper = rng.random() < 0.5
    d.line(("merchant", r["merchant"].upper() if upper else r["merchant"]))
    d.line(f"{r['street']}, {r['city']}")
    if r["time"]:
        d.line(("date", D(r["date"])), "   ", ("time", _fmt_time(r["time"], r["time_style"])))
    else:
        d.line(("date", D(r["date"])))
    d.line(rng.choice(["Receipt #", "Trans: ", "Receipt No. "]), ("receipt_number", r["receipt_number"]))
    if r["cashier"]:
        d.line(rng.choice(["Cashier: ", "Served by ", "Your server: "]), ("cashier", r["cashier"]))
    d.line("-" * 32)
    for i, (name, p) in enumerate(r["items"]):
        shown = name.upper() if upper else name
        d.line(("items#%d" % i, shown), " " * (22 - len(shown)), M(p))
    d.line("-" * 32)
    for ln in _rcp_amounts(r, rng, upper, 22):
        d.line(*ln)
    if r["method"]:
        d.line(*_rcp_payment(r, upper))
    d.add(rng.choice(["THANK YOU!", "Thanks for shopping with us", "Please come again"]))


def _rcp_t1(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    head: list[Part] = [rng.choice(["Thanks for your purchase at ", "Your receipt from "]), ("merchant", r["merchant"]),
                        "! Receipt ", ("receipt_number", r["receipt_number"]), " was issued on ", ("date", D(r["date"]))]
    if r["time"]:
        head += [" at ", ("time", _fmt_time(r["time"], r["time_style"]))]
    d.line(*head, ".")
    sents: list[list[Part]] = [["You bought ", *_series("items", [n for n, _ in r["items"]]), "."]]
    amt: list[Part] = ["Subtotal ", ("subtotal", M(r["subtotal"]))]
    if r["tax"] is not None:
        amt += [", tax ", ("tax", M(r["tax"]))]
    if r["tip"] is not None:
        amt += [", tip ", ("tip", M(r["tip"]))]
    amt.append(".")
    sents.append(amt)
    if r["total_printed"] or r["method"]:
        pay: list[Part] = []
        if r["total_printed"]:
            pay += ["Total charged: ", ("total", M(r["total"]))]
        if r["method"]:
            pay += [" — paid with " if pay else "Paid with ", *_rcp_payment(r, False)]
        sents.append([*pay, "."])
    if r["cashier"]:
        sents.append(["You were served by ", ("cashier", r["cashier"]), "."])
    rng.shuffle(sents)
    for s in sents:
        d.line(*s)


def _rcp_t2(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    kv: list[list[Part]] = [
        [rng.choice(["Merchant: ", "Store: "]), ("merchant", r["merchant"])],
        ["Date: ", ("date", D(r["date"]))],
        [rng.choice(["Transaction ID: ", "Receipt: "]), ("receipt_number", r["receipt_number"])],
    ]
    if r["time"]:
        kv.append(["Time: ", ("time", _fmt_time(r["time"], r["time_style"]))])
    if r["cashier"]:
        kv.append(["Cashier: ", ("cashier", r["cashier"])])
    if r["method"]:
        kv.append(["Payment: ", *_rcp_payment(r, False)])
    rng.shuffle(kv)
    for ln in kv:
        d.line(*ln)
    d.line("Items:")
    for i, (name, p) in enumerate(r["items"]):
        d.line("- ", ("items#%d" % i, name), "  ", M(p))
    for ln in _rcp_amounts(r, rng, False, 0):
        ln[0] = ln[0].rstrip() + ": "
        d.line(*ln)


def _rcp_t3(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    head: list[Part] = [("merchant", r["merchant"]), " #", ("receipt_number", r["receipt_number"]), " ",
                        ("date", D(r["date"]))]
    if r["time"]:
        head += [" ", ("time", _fmt_time(r["time"], r["time_style"]))]
    d.line(*head)
    for i, (name, p) in enumerate(r["items"]):
        d.line(("items#%d" % i, name), " @ ", M(p))
    segs: list[list[Part]] = [["Sub ", ("subtotal", M(r["subtotal"]))]]
    if r["tax"] is not None:
        segs.append(["Tax ", ("tax", M(r["tax"]))])
    if r["tip"] is not None:
        segs.append(["Tip ", ("tip", M(r["tip"]))])
    if r["total_printed"]:
        segs.append(["Total ", ("total", M(r["total"]))])
    d.line(*_flat(segs, " / "))
    tail: list[list[Part]] = []
    if r["method"]:
        tail.append(["Paid: ", *_rcp_payment(r, False)])
    if r["cashier"]:
        tail.append(["Clerk ", ("cashier", r["cashier"])])
    if tail:
        d.line(*_flat(tail, " · "))


def _gold_receipt(r: _Rec, g: _Gold) -> None:
    g.ext("merchant_name", "merchant", g.surface("merchant"))
    g.ext("receipt_number", "receipt_number", r["receipt_number"])
    g.ext("transaction_date", "date", r["date"].isoformat())
    g.opt_ext("transaction_time", "time", None if r["time"] is None else r["time"].strftime("%H:%M"))
    g.lst("items", "items", [g.surface(f"items#{i}") for i in range(len(r["items"]))])
    g.ext("subtotal", "subtotal", _cents(r["subtotal"]))
    g.opt_ext("tax", "tax", None if r["tax"] is None else _cents(r["tax"]))
    g.opt_ext("tip", "tip", None if r["tip"] is None else _cents(r["tip"]))
    if r["total_printed"]:
        g.ext("total", "total", _cents(r["total"]))
    else:
        g.comp("total", _cents(r["total"]))
    g.opt_ext("payment_method", "method", None if r["method"] is None else g.surface("method"))
    g.opt_ext("card_last_four", "last4", r["last4"])
    g.opt_ext("cashier_name", "cashier", r["cashier"])
    g.comp("item_count", len(r["items"]))


# --------------------------------------------------------------------------- support email


def _rec_email(rng: random.Random) -> _Rec:
    r = _Rec()
    first, last = _person(rng)
    r["first"], r["last"] = first, last
    r["signature"] = first if rng.random() < 0.5 else f"{first} {last}"
    email = f"{_ascii(first).lower()}.{_ascii(last).lower()}@{rng.choice(['gmail.com', 'outlook.com', 'fastmail.com', 'proton.me'])}"
    r.opt("email", rng.random() < 0.4, email, ("text", email))
    phone = _phone(rng)
    r.opt("phone", rng.random() < 0.3, phone, ("text", phone))
    r["order_number"] = _order_number(rng)
    r["product"] = rng.choice(PRODUCTS)
    order = _rand_date(rng)
    delivered = order + dt.timedelta(days=rng.randint(2, 14))
    r["ref"] = delivered + dt.timedelta(days=rng.randint(0, 10))
    r.opt("order_date", rng.random() < 0.5, order, ("date", order))
    r.opt("delivery_date", rng.random() < 0.85, delivered, ("date", delivered))
    price = rng.randint(19_00, 1499_00)
    if rng.random() < 0.3:
        price = price // 100 * 100 + 99 if rng.random() < 0.5 else price // 100 * 100
    r["price"] = price
    tax = round(price * rng.choice([6, 7.5, 8.25, 9, 10]) / 100)
    r.opt("tax", rng.random() < 0.55, tax, _num_check(tax))
    ship = rng.choice([4_99, 7_95, 9_99, 12_50, 15_00, 24_99])
    r.opt("shipping", rng.random() < 0.35, ship, _num_check(ship))
    r["total"] = price + (r["tax"] or 0) + (r["shipping"] or 0)
    adjusted = r["tax"] is not None or r["shipping"] is not None
    r["total_stated"] = adjusted and rng.random() < 0.35
    r["damages"] = rng.sample(DAMAGES, rng.choices([1, 2, 3], [0.45, 0.4, 0.15])[0])
    res = rng.choice(RESOLUTIONS)
    r.opt("resolution", rng.random() < 0.8, res, ("phrases", [res]))
    carrier = rng.choice(_CARRIERS)
    trk = _tracking(rng, carrier)
    r.opt("tracking", rng.random() < 0.25, trk, ("text", trk))
    r["money"] = _money_style(rng)
    r["date_style"] = rng.choice(
        ["month_noyear", "ordinal_noyear", "mon_abbr4", "month_full", "dmy_slash", "iso", "weekday_noyear"]
    )
    return r


def _eml_money_sentence(r: _Rec, rng: random.Random) -> list[Part]:
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    s: list[Part] = [rng.choice(["I paid ", "It cost me ", "The price was "]), ("price", M(r["price"]))]
    extras: list[list[Part]] = []
    if r["tax"] is not None:
        extras.append([("tax", M(r["tax"])), " tax"])
    if r["shipping"] is not None:
        extras.append([("shipping", M(r["shipping"])), " for shipping"])
    if extras:
        s += [" plus ", *_flat(extras, " and ")]
    if r["total_stated"]:
        s += [rng.choice([", so ", " — "]), ("total", M(r["total"])), " in total"]
    s.append(".")
    return s


_MASS_NOUNS = frozenset({"water damage", "dead pixels", "loose buttons", "torn packaging", "stained fabric", "store credit"})


def _art(phrase: str) -> str:
    if phrase in _MASS_NOUNS:
        return ""
    return "an " if phrase[0] in "aeiou" else "a "


def _eml_damage(r: _Rec) -> list[Part]:
    parts: list[Part] = []
    n = len(r["damages"])
    for i, dmg in enumerate(r["damages"]):
        if i:
            parts.append(" and " if i == n - 1 else ", ")
        parts += [_art(dmg), (f"damages#{i}", dmg)]
    return parts


def _eml_t0(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    d.line(rng.choice(["Subject: Damaged item", "Subject: Problem with my order", "Subject: Broken on arrival"]))
    d.line()
    d.line(rng.choice(["Hi,", "Hello,", "Hi there,"]))
    d.line()
    first: list[Part] = ["My order ", ("order_number", r["order_number"])]
    if r["delivery_date"]:
        first += [" arrived on ", ("delivery_date", D(r["delivery_date"]))]
    else:
        first += [" finally arrived"]
    first += [" with ", *_eml_damage(r), "."]
    sents: list[list[Part]] = [first]
    prod: list[Part] = ["It's the ", ("product", r["product"])]
    if r["order_date"]:
        prod += [", which I ordered on ", ("order_date", D(r["order_date"]))]
    sents.append([*prod, "."])
    sents.append(_eml_money_sentence(r, rng))
    if r["tracking"]:
        sents.append(["The tracking number was ", ("tracking", r["tracking"]), "."])
    tail = sents[1:]
    rng.shuffle(tail)
    for s in [sents[0], *tail]:
        d.line(*s)
    if r["resolution"]:
        d.line(rng.choice(["I'd like ", "Can I please get ", "I would like "]), _art(r["resolution"]),
               ("resolution", r["resolution"]), ".")
    if r["phone"]:
        d.line("You can reach me at ", ("phone", r["phone"]), ".")
    if r["email"]:
        d.line("Please reply to ", ("email", r["email"]), ".")
    d.line()
    d.add(rng.choice(["Thanks,\n", "Thank you,\n", "Best,\n"]), ("customer", r["signature"]))


def _eml_t1(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    d.line("Dear Customer Service,")
    d.line()
    open_: list[Part] = ["I am writing regarding order #", ("order_number", r["order_number"])]
    if r["order_date"]:
        open_ += [", which I placed on ", ("order_date", D(r["order_date"]))]
    if r["delivery_date"]:
        open_ += [" and received on " if r["order_date"] else ", which I received on ",
                  ("delivery_date", D(r["delivery_date"]))]
    d.line(*open_, ".")
    d.line("Unfortunately the ", ("product", r["product"]), " was damaged:")
    for i, dmg in enumerate(r["damages"]):
        d.line("  - ", (f"damages#{i}", dmg.capitalize() if i % 2 == 0 else dmg))
    money: list[Part] = []
    if r["total_stated"]:
        money += ["The charge on my card was ", ("total", M(r["total"])), ", which included ", ("price", M(r["price"])),
                  " for the item"]
    else:
        money += ["The item itself cost ", ("price", M(r["price"]))]
    if r["tax"] is not None:
        money += [", " if r["total_stated"] else ", plus ", ("tax", M(r["tax"])), " in tax"]
    if r["shipping"] is not None:
        money += [" and ", ("shipping", M(r["shipping"])), " in shipping"]
    d.line(*money, ".")
    if r["tracking"]:
        d.line("For reference, the shipment tracking ID is ", ("tracking", r["tracking"]), ".")
    if r["resolution"]:
        d.line("I would appreciate ", _art(r["resolution"]), ("resolution", r["resolution"]),
               " at your earliest convenience.")
    else:
        d.line("Please let me know how you intend to resolve this.")
    d.line()
    d.line("Sincerely,")
    d.line(("customer", r["signature"]))
    contact: list[Part] = [("phone", r["phone"])] if r["phone"] else []
    if r["email"]:
        contact.append(("email", r["email"]))
    rng.shuffle(contact)
    for c in contact:
        d.line(c)


def _eml_t2(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    s: list[Part] = [rng.choice(["hey - got my ", "hi, just got my ", "so my "]), ("product", r["product"])]
    if r["delivery_date"]:
        s += [" (", ("delivery_date", D(r["delivery_date"])), ")"]
    s += [" and it has ", *_eml_damage(r), ". order ", ("order_number", r["order_number"])]
    if r["order_date"]:
        s += [", ordered ", ("order_date", D(r["order_date"]))]
    d.line(*s, ".")
    pay: list[Part] = ["paid ", ("price", M(r["price"]))]
    if r["tax"] is not None:
        pay += [" + ", ("tax", M(r["tax"])), " tax"]
    if r["shipping"] is not None:
        pay += [" + ", ("shipping", M(r["shipping"])), " shipping"]
    if r["total_stated"]:
        pay += [" = ", ("total", M(r["total"]))]
    d.line(*pay)
    if r["tracking"]:
        d.line("tracking # ", ("tracking", r["tracking"]))
    if r["resolution"]:
        d.line("want ", _art(r["resolution"]), ("resolution", r["resolution"]), " pls")
    contact: list[list[Part]] = []
    if r["phone"]:
        contact.append(["call ", ("phone", r["phone"])])
    if r["email"]:
        contact.append(["or email ", ("email", r["email"])] if contact else ["email ", ("email", r["email"])])
    if contact:
        d.line(*_flat(contact, " "))
    d.add("- ", ("customer", r["signature"]))


def _eml_t3(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    if r["email"]:
        d.line("From: ", ("email", r["email"]))
    d.line("To: support@", _slug(rng.choice(RETAILERS)), ".com")
    d.line("Subject: Order ", ("order_number", r["order_number"]), " — ", rng.choice(["arrived damaged", "need help", "damaged delivery"]))
    d.line()
    d.line("Hello,")
    d.line()
    s: list[Part] = ["The package"]
    if r["tracking"]:
        s += [" (tracking ", ("tracking", r["tracking"]), ")"]
    s += [" was delivered"]
    if r["delivery_date"]:
        s += [" on ", ("delivery_date", D(r["delivery_date"]))]
    s += [" and the ", ("product", r["product"]), " inside has ", *_eml_damage(r), "."]
    sents: list[list[Part]] = [s]
    if r["order_date"]:
        sents.append(["I placed the order on ", ("order_date", D(r["order_date"])), "."])
    sents.append(_eml_money_sentence(r, rng))
    if r["resolution"]:
        sents.append(["Please arrange ", _art(r["resolution"]), ("resolution", r["resolution"]), "."])
    if r["phone"]:
        sents.append(["My number is ", ("phone", r["phone"]), "."])
    tail = sents[1:]
    rng.shuffle(tail)
    d.line(*_flat([sents[0], *tail], " "))
    d.line()
    d.add(rng.choice(["Regards,\n", "Cheers,\n", "Thanks in advance,\n"]), ("customer", r["signature"]))


def _gold_email(r: _Rec, g: _Gold) -> None:
    g.ext("order_number", "order_number", r["order_number"])
    g.ext("customer_name", "customer", r["signature"])
    g.opt_ext("customer_email", "email", r["email"])
    g.opt_ext("phone_number", "phone", r["phone"])
    g.ext("product_name", "product", r["product"])
    g.opt_ext("order_date", "order_date", r["order_date"] and r["order_date"].isoformat())
    g.opt_ext("delivery_date", "delivery_date", r["delivery_date"] and r["delivery_date"].isoformat())
    g.ext("price_paid", "price", _cents(r["price"]))
    g.opt_ext("tax", "tax", None if r["tax"] is None else _cents(r["tax"]))
    g.opt_ext("shipping_cost", "shipping", None if r["shipping"] is None else _cents(r["shipping"]))
    if r["total_stated"]:
        g.ext("total_paid", "total", _cents(r["total"]))
    elif r["tax"] is not None or r["shipping"] is not None:
        g.comp("total_paid", _cents(r["total"]))
    else:
        g.ext("total_paid", "price", _cents(r["price"]))
    g.lst("damage_types", "damages", [g.surface(f"damages#{i}") for i in range(len(r["damages"]))])
    g.opt_ext("requested_resolution", "resolution", r["resolution"])
    if r["order_date"] and r["delivery_date"]:
        g.comp("days_to_delivery", (r["delivery_date"] - r["order_date"]).days)
    g.opt_ext("tracking_number", "tracking", r["tracking"])


# --------------------------------------------------------------------------- shipping notice


def _rec_shipping(rng: random.Random) -> _Rec:
    r = _Rec()
    r["carrier"] = rng.choice(_CARRIERS)
    trk = _tracking(rng, r["carrier"])
    r.opt("tracking", rng.random() < 0.9, trk, ("text", trk))
    order = _order_number(rng)
    r.opt("order_number", rng.random() < 0.7, order, ("text", order))
    r["retailer"] = rng.choice(RETAILERS)
    r["first"], r["last"] = _person(rng)
    r["recipient"] = f"{r['first']} {r['last']}"
    r["city"] = rng.choice(CITIES)
    ship = _rand_date(rng)
    eta = ship + dt.timedelta(days=rng.randint(1, 9))
    r["ref"] = ship + dt.timedelta(days=rng.randint(0, (eta - ship).days))
    r.opt("ship_date", rng.random() < 0.85, ship, ("date", ship))
    r.opt("eta", rng.random() < 0.8, eta, ("date", eta))
    r["weight_unit"] = rng.choice(["kg", "lb", "lbs"])
    w = round(rng.uniform(0.3, 24.0), 1)
    r.opt("weight", rng.random() < 0.5, w, ("phrases", [f"{w} {u}" for u in ("kg", "lb", "lbs")]))
    n = rng.choices([1, 2, 3, 4], [0.1, 0.5, 0.25, 0.15])[0]
    r.opt("packages", rng.random() < 0.4, n, ("phrases", [f"{n} package", f"{n} parcel", f"pieces: {n}"]))
    cost = rng.randint(3_99, 49_99)
    r.opt("cost", rng.random() < 0.3, cost, _num_check(cost))
    items = rng.sample(PRODUCTS, rng.randint(1, 3))
    r.opt("items", rng.random() < 0.6, items, ("phrases", items))
    r["money"] = _money_style(rng)
    r["date_style"] = rng.choice(
        ["mon_abbr", "month_full", "iso", "weekday_full", "day_month_abbr", "dmy_slash", "month_noyear"]
    )
    return r


def _shp_weight(r: _Rec) -> str:
    return f"{r['weight']} {r['weight_unit']}"


def _shp_t0(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    d.line(rng.choice(["Your order is on its way!", "Good news: your package has shipped", "Shipping update"]))
    d.line()
    s: list[Part] = [f"Hi {r['first']}, your ", ("retailer", r["retailer"]), " order"]
    if r["order_number"]:
        s += [" ", ("order_number", r["order_number"])]
    s += [" has shipped"]
    if r["ship_date"]:
        s += [" on ", ("ship_date", D(r["ship_date"]))]
    s += [" via ", ("carrier", r["carrier"]), "."]
    d.line(*s)
    kv: list[list[Part]] = []
    if r["tracking"]:
        kv.append(["Tracking number: ", ("tracking", r["tracking"])])
    if r["eta"]:
        kv.append([rng.choice(["Estimated delivery: ", "Arriving by: ", "Expected: "]), ("eta", D(r["eta"]))])
    kv.append(["Shipping to: ", ("recipient", r["recipient"]), ", ", ("city", r["city"])])
    if r["cost"] is not None:
        kv.append(["Shipping paid: ", ("cost", M(r["cost"]))])
    rng.shuffle(kv)
    for ln in kv:
        d.line(*ln)
    if r["items"]:
        d.line("Items in this shipment: ", *_series("items", r["items"]), ".")
    pk: list[list[Part]] = []
    if r["packages"]:
        pk.append([("packages", str(r["packages"])), " package" + ("s" if r["packages"] > 1 else "")])
    if r["weight"] is not None:
        pk.append(["total weight ", ("weight", _shp_weight(r))])
    if pk:
        d.line(*_flat(pk, ", "), ".")


def _shp_t1(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    d.line(rng.choice(["SHIPMENT NOTIFICATION", "DISPATCH NOTICE", "Shipment Details"]))
    d.line("=" * 24)
    kv: list[list[Part]] = [["Carrier: ", ("carrier", r["carrier"])], ["Shipper: ", ("retailer", r["retailer"])],
                            ["Ship To: ", ("recipient", r["recipient"]), " / ", ("city", r["city"])]]
    if r["tracking"]:
        kv.append([rng.choice(["Tracking #: ", "Tracking No.: ", "AWB: "]), ("tracking", r["tracking"])])
    if r["ship_date"]:
        kv.append([rng.choice(["Ship Date: ", "Shipped: "]), ("ship_date", D(r["ship_date"]))])
    if r["eta"]:
        kv.append([rng.choice(["Est. Delivery: ", "ETA: "]), ("eta", D(r["eta"]))])
    if r["order_number"]:
        kv.append([rng.choice(["Order Ref: ", "Order: "]), ("order_number", r["order_number"])])
    if r["weight"] is not None:
        kv.append(["Weight: ", ("weight", _shp_weight(r))])
    if r["packages"]:
        kv.append(["Pieces: ", ("packages", str(r["packages"]))])
    if r["cost"] is not None:
        kv.append(["Shipping charges: ", ("cost", M(r["cost"]))])
    if r["items"]:
        kv.append(["Contents: ", *_series("items", r["items"], conj=None, sep="; ")])
    rng.shuffle(kv)
    for ln in kv:
        d.line(*ln)


def _shp_t2(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    s: list[Part] = [("carrier", r["carrier"]), ": Package"]
    if r["tracking"]:
        s += [" ", ("tracking", r["tracking"])]
    s += [" from ", ("retailer", r["retailer"]), " for ", ("recipient", r["recipient"]), " (", ("city", r["city"]), ")"]
    if r["eta"]:
        s += [" is scheduled for delivery ", ("eta", D(r["eta"]))]
    else:
        s += [" is in transit"]
    d.line(*s, ".")
    extra: list[list[Part]] = []
    if r["ship_date"]:
        extra.append(["Shipped ", ("ship_date", D(r["ship_date"]))])
    if r["order_number"]:
        extra.append(["Order ", ("order_number", r["order_number"])])
    if r["packages"]:
        extra.append([("packages", str(r["packages"])), " pc"])
    if r["weight"] is not None:
        extra.append([("weight", _shp_weight(r))])
    if r["cost"] is not None:
        extra.append(["Postage ", ("cost", M(r["cost"]))])
    if r["items"]:
        extra.append(["Contains ", *_series("items", r["items"])])
    rng.shuffle(extra)
    if extra:
        d.line(*_flat(extra, ". "), ".")
    d.add(rng.choice(["Reply STOP to opt out.", "Track online anytime.", ""]))


def _shp_t3(r: _Rec, rng: random.Random, d: _Doc) -> None:
    D = lambda x: _fmt_date(x, r["date_style"], r["ref"])  # noqa: E731
    M = lambda c: _fmt_money(c, r["money"])  # noqa: E731
    if r["order_number"]:
        d.line("Dispatch confirmation — Order ", ("order_number", r["order_number"]))
    else:
        d.line("Dispatch confirmation")
    d.line()
    d.line("Dear ", ("recipient", r["recipient"]), ",")
    d.line()
    s: list[Part] = ["We have dispatched your parcel from ", ("retailer", r["retailer"])]
    if r["weight"] is not None:
        s += [" (", ("weight", _shp_weight(r)), ")"]
    if r["ship_date"]:
        s += [" on ", ("ship_date", D(r["ship_date"]))]
    if r["eta"]:
        s += [" and it should arrive by ", ("eta", D(r["eta"]))]
    s += [" to your address in ", ("city", r["city"]), "."]
    sents: list[list[Part]] = [s]
    c: list[Part] = ["Courier: ", ("carrier", r["carrier"])]
    if r["tracking"]:
        c += ["; reference ", ("tracking", r["tracking"])]
    sents.append([*c, "."])
    if r["cost"] is not None:
        sents.append(["Postage paid: ", ("cost", M(r["cost"])), "."])
    if r["packages"]:
        sents.append(["Number of parcels: ", ("packages", str(r["packages"])), "."])
    if r["items"]:
        sents.append(["Your parcel contains ", *_series("items", r["items"]), "."])
    tail = sents[1:]
    rng.shuffle(tail)
    for x in [sents[0], *tail]:
        d.line(*x)
    d.line()
    d.add(rng.choice(["Kind regards,\nThe Dispatch Team", "Thank you for shopping with us."]))


def _gold_shipping(r: _Rec, g: _Gold) -> None:
    g.opt_ext("tracking_number", "tracking", r["tracking"])
    g.ext("carrier", "carrier", r["carrier"])
    g.opt_ext("order_number", "order_number", r["order_number"])
    g.ext("retailer_name", "retailer", r["retailer"])
    g.ext("recipient_name", "recipient", r["recipient"])
    g.ext("destination_city", "city", r["city"])
    g.opt_ext("ship_date", "ship_date", r["ship_date"] and r["ship_date"].isoformat())
    g.opt_ext("estimated_delivery", "eta", r["eta"] and r["eta"].isoformat())
    g.opt_ext("package_weight", "weight", r["weight"], extras={"unit": r["weight_unit"].rstrip("s")})
    g.opt_ext("package_count", "packages", r["packages"])
    g.opt_ext("shipping_cost", "cost", None if r["cost"] is None else _cents(r["cost"]))
    if r["items"]:
        g.lst("items", "items", r["items"])
    else:
        g.null("items")
    if r["ship_date"] and r["eta"]:
        g.comp("transit_days", (r["eta"] - r["ship_date"]).days)
    if r["eta"]:
        g.comp("days_until_delivery", (r["eta"] - r["ref"]).days)


# --------------------------------------------------------------------------- gold assembly


class _Gold:
    """Collects candidate gold entries (one per catalog field) for a rendered document."""

    def __init__(self, doc: _Doc, state: str) -> None:
        self.doc, self.state = doc, state
        self.entries: dict[str, dict[str, Any]] = {}
        self.extras: dict[str, dict[str, Any]] = {}

    def surface(self, key: str) -> str:
        s, e = self.doc.spans[key]
        return self.state[s:e]

    def ext(self, name: str, key: str, value: Any) -> None:
        s, e = self.doc.spans[key]
        self.entries[name] = {"value": value, "kind": "extract", "span": [s, e], "spans": None}

    def opt_ext(self, name: str, key: str, value: Any, extras: dict[str, Any] | None = None) -> None:
        if value is None:
            self.null(name)
        else:
            self.ext(name, key, value)
        if extras:
            self.extras[name] = extras

    def lst(self, name: str, key: str, values: list[str]) -> None:
        spans = [list(self.doc.spans[f"{key}#{i}"]) for i in range(len(values))]
        self.entries[name] = {"value": list(values), "kind": "extract", "span": None, "spans": spans}

    def comp(self, name: str, value: Any) -> None:
        self.entries[name] = {"value": value, "kind": "compute", "span": None, "spans": None}

    def null(self, name: str) -> None:
        self.entries[name] = {"value": None, "kind": "null", "span": None, "spans": None}


_RECORDS: dict[str, Callable[[random.Random], _Rec]] = {
    "invoice": _rec_invoice, "receipt": _rec_receipt, "support_email": _rec_email, "shipping_notice": _rec_shipping,
}
_TEMPLATES: dict[str, list[Callable[[_Rec, random.Random, _Doc], None]]] = {
    "invoice": [_inv_t0, _inv_t1, _inv_t2, _inv_t3],
    "receipt": [_rcp_t0, _rcp_t1, _rcp_t2, _rcp_t3],
    "support_email": [_eml_t0, _eml_t1, _eml_t2, _eml_t3],
    "shipping_notice": [_shp_t0, _shp_t1, _shp_t2, _shp_t3],
}
_GOLD: dict[str, Callable[[_Rec, _Gold], None]] = {
    "invoice": _gold_invoice, "receipt": _gold_receipt, "support_email": _gold_email, "shipping_notice": _gold_shipping,
}
# record key (omitted) -> catalog field name, where they differ
_OMIT_TO_FIELD = {
    "receipt": {"time": "transaction_time", "method": "payment_method", "last4": "card_last_four", "cashier": "cashier_name"},
    "support_email": {"email": "customer_email", "phone": "phone_number", "shipping": "shipping_cost",
                      "resolution": "requested_resolution", "tracking": "tracking_number"},
    "shipping_notice": {"tracking": "tracking_number", "eta": "estimated_delivery", "weight": "package_weight",
                        "packages": "package_count", "cost": "shipping_cost"},
}

TEMPLATE_IDS: tuple[str, ...] = tuple(f"{t}/t{i}" for t in DOC_TYPES for i in range(len(_TEMPLATES[t])))


def field_spec(doc_type: str, name: str, rng: random.Random, currency: str | None = None,
               extras: dict[str, Any] | None = None) -> dict[str, Any]:
    type_, static = CATALOG[doc_type][name]
    spec: dict[str, Any] = {"type": type_, "instructions": rng.choice(INSTRUCTIONS[name])}
    if static is _MONEY and currency:
        spec["unit"] = currency
    spec.update(static)
    if extras:
        spec.update(extras)
    return spec


def _mentions(state: str, check: tuple[str, Any], ref: str) -> bool:
    kind, val = check
    low = state.casefold()
    if kind == "text":
        return str(val).casefold() in low
    if kind == "phrases":
        return any(str(p).casefold() in low for p in val)
    if kind == "number":
        for m in re.finditer(r"\d+(?:[.,]\d+)*", state):
            try:
                if abs(parse_number(m.group(0)) - float(val)) < 1e-9:
                    return True
            except ValueError:
                continue
        return False
    if kind == "date":
        return any(iso == val.isoformat() for _, _, iso in find_dates(state, ref))
    raise ValueError(kind)


def validate_example(ex: dict[str, Any]) -> None:
    """Raise AssertionError unless every extract span normalizes to its gold value."""
    req = ex["request"]
    state, ref = req["state"], req["reference_date"]
    for name, g in ex["gold"].items():
        spec = req["fields"][name]
        if g["kind"] != "extract":
            assert g["span"] is None and g["spans"] is None, name
            if g["kind"] == "null":
                assert g["value"] is None, name
            continue
        if spec["type"] == "list":
            assert len(g["spans"]) == len(g["value"]), name
            pairs = [(spec["items"], (s, e), v) for (s, e), v in zip(g["spans"], g["value"])]
        else:
            pairs = [(spec["type"], tuple(g["span"]), g["value"])]
        for type_, (s, e), v in pairs:
            raw = state[s:e]
            assert raw and raw == raw.strip(), (name, raw)
            got = normalize_value(type_, raw, spec.get("decimals"), ref)
            want = normalize_value(type_, v, spec.get("decimals"), ref)
            ok = text_equal(got, want) if type_ == "text" else got == want
            assert ok, f"{ex['id']} {name}: span {raw!r} -> {got!r} != gold {want!r}"


def _select(rng: random.Random, entries: dict[str, dict[str, Any]], decoys: dict[str, tuple[str, ...]]) -> list[str]:
    by_kind: dict[str, list[str]] = {"extract": [], "compute": [], "null": []}
    for name, g in entries.items():
        by_kind[g["kind"]].append(name)
    n_total = min(rng.randint(MIN_FIELDS, MAX_FIELDS), len(entries))
    n_null = min(sum(rng.random() < P_NULL for _ in range(n_total)), len(by_kind["null"]))
    n_comp = min(sum(rng.random() < P_COMPUTE for _ in range(n_total)), len(by_kind["compute"]))
    n_ext = max(1, min(n_total - n_null - n_comp, len(by_kind["extract"])))
    chosen = rng.sample(by_kind["extract"], n_ext) + rng.sample(by_kind["compute"], n_comp)
    # Null fields: prefer ones whose decoy (similar-looking value) is printed in the document.
    pool = list(by_kind["null"])
    for _ in range(n_null):
        weights = [3.0 if any(entries.get(d, {}).get("kind") == "extract" for d in decoys.get(n, ())) else 1.0
                   for n in pool]
        pick = rng.choices(pool, weights)[0]
        pool.remove(pick)
        chosen.append(pick)
    rng.shuffle(chosen)
    return chosen


def _render(rng: random.Random, doc_type: str) -> tuple[_Rec, int, str, _Gold, dict[str, tuple[str, Any]]]:
    """Draw a record, render it through a random template, and build gold for every catalog field.

    Returns ``(record, template index, state, gold, omitted)`` where ``omitted`` maps each null
    field to its hypothetical value check. Redraws when an absent value happens to appear in the text.
    """
    alias = _OMIT_TO_FIELD.get(doc_type, {})
    for _ in range(100):
        rec = _RECORDS[doc_type](rng)
        t_idx = rng.randrange(len(_TEMPLATES[doc_type]))
        doc = _Doc()
        _TEMPLATES[doc_type][t_idx](rec, rng, doc)
        state = doc.text()
        g = _Gold(doc, state)
        _GOLD[doc_type](rec, g)
        omitted = {alias.get(k, k): v for k, v in rec.omitted.items()}
        null_names = [n for n, e in g.entries.items() if e["kind"] == "null"]
        assert set(null_names) <= set(omitted), (doc_type, set(null_names) - set(omitted))
        if not any(_mentions(state, omitted[n], rec["ref"].isoformat()) for n in null_names):
            return rec, t_idx, state, g, {n: omitted[n] for n in null_names}
    raise RuntimeError(f"could not generate a valid {doc_type} example")


def _build(rng: random.Random, doc_type: str, idx: int) -> tuple[dict[str, Any], dict[str, tuple[str, Any]]]:
    """One validated example plus the hypothetical values of its null fields."""
    rec, t_idx, state, g, omitted = _render(rng, doc_type)
    ref = rec["ref"].isoformat()
    names = _select(rng, g.entries, DECOYS[doc_type])
    fields = {n: field_spec(doc_type, n, rng, rec["money"][0], g.extras.get(n)) for n in names}
    ex = {
        "id": f"{_ID_PREFIX[doc_type]}-{idx:06d}",
        "doc_type": doc_type,
        "template_id": f"{doc_type}/t{t_idx}",
        "request": {"state": state, "fields": fields, "reference_date": ref},
        "gold": {n: g.entries[n] for n in names},
    }
    validate_example(ex)
    return ex, {n: omitted[n] for n in names if n in omitted}



def generate_with_nulls(
    n: int, seed: int, doc_types: Sequence[str] | None = None
) -> list[tuple[dict[str, Any], dict[str, tuple[str, Any]]]]:
    """Like `generate`, also returning each example's null fields' hypothetical (absent) values."""
    types = tuple(doc_types) if doc_types else DOC_TYPES
    unknown = set(types) - set(DOC_TYPES)
    if unknown:
        raise ValueError(f"unknown doc types {sorted(unknown)}")
    out = []
    for i in range(n):
        rng = random.Random(f"synth:{seed}:{i}")
        out.append(_build(rng, types[rng.randrange(len(types))], i))
    return out


def generate(n: int, seed: int, doc_types: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """``n`` validated contract-format examples; deterministic in ``(n, seed, doc_types)``.

    Example ``i`` depends only on ``(seed, i, doc_types)``, so prefixes are stable across ``n``.
    """
    return [ex for ex, _ in generate_with_nulls(n, seed, doc_types)]


def null_is_absent(state: str, check: tuple[str, Any], reference_date: str) -> bool:
    return not _mentions(state, check, reference_date)


def split_by_template(
    examples: Iterable[dict[str, Any]], holdout_templates: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Template-disjoint split: ``(examples not in holdout_templates, examples in holdout_templates)``."""
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for ex in examples:
        (held if ex["template_id"] in holdout_templates else train).append(ex)
    return train, held


def write_jsonl(examples: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
