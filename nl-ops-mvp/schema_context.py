"""Business metadata and deterministic, token-conscious schema retrieval."""
import json
import re

SCHEMA_CATALOG = {
    "employees": {
        "description": "Internal staff employed by the synthetic enterprise; not customer accounts.",
        "columns": {
            "id": {"type": "INTEGER", "description": "Primary key identifying an employee."},
            "name": {"type": "TEXT", "description": "Employee's full display name."},
            "department": {"type": "TEXT", "description": "Business department, e.g. Engineering, Support, or Finance."},
            "salary": {"type": "REAL", "description": "Annual compensation in USD in this synthetic dataset."},
        },
        "relationships": [],
    },
    "customers": {
        "description": "External customer accounts, also called users; distinct from internal employees.",
        "columns": {
            "id": {"type": "INTEGER", "description": "Primary key identifying a customer."},
            "email": {"type": "TEXT", "description": "Customer's synthetic email address."},
            "country": {"type": "TEXT", "description": "Country name. Represented European countries: United Kingdom, Germany, France."},
            "signup_date": {"type": "TEXT", "description": "Account signup date as an ISO YYYY-MM-DD string."},
        },
        "relationships": [],
    },
    "tickets": {
        "description": "IT helpdesk requests submitted by employees. All tickets are IT tickets, regardless of employee department.",
        "columns": {
            "id": {"type": "INTEGER", "description": "Primary key identifying a helpdesk ticket."},
            "employee_id": {"type": "INTEGER", "description": "Employee who submitted the ticket; foreign key to employees.id."},
            "subject": {"type": "TEXT", "description": "Brief description of the IT issue, e.g. VPN access."},
            "status": {"type": "TEXT", "description": "Ticket status; seeded values are open and closed."},
        },
        "relationships": [{"column": "employee_id", "references_table": "employees", "references_column": "id",
                           "mapping": "tickets.employee_id references employees.id"}],
    },
}

ENTITY_KEYWORDS = {
    "employees": {"employee", "employees", "staff", "personnel", "salary", "salaries", "compensation", "pay", "department", "departments"},
    "customers": {"customer", "customers", "user", "users", "account", "accounts", "email", "emails", "country", "countries", "signup", "signup_date", "signed", "signups"},
    "tickets": {"ticket", "tickets", "helpdesk", "issue", "issues", "vpn", "subject", "status", "open", "closed"},
}


def get_relevant_schema(user_query: str) -> str:
    """Return JSON for matched entities plus outgoing foreign-key targets.

    Unknown/ambiguous vocabulary falls back to the small complete catalog.
    This is prompt scoping, not an authorization mechanism.
    """
    if not isinstance(user_query, str):
        raise TypeError("user_query must be a string")
    words = set(re.findall(r"\b\w+\b", user_query.casefold()))
    selected = {table for table, keywords in ENTITY_KEYWORDS.items() if words & keywords}
    if not selected:
        selected = set(SCHEMA_CATALOG)
    # Include dependencies for valid joins without exposing unrelated tables.
    pending = list(selected)
    while pending:
        for relation in SCHEMA_CATALOG[pending.pop()]["relationships"]:
            target = relation["references_table"]
            if target not in selected:
                selected.add(target)
                pending.append(target)
    return json.dumps({"tables": {table: metadata for table, metadata in SCHEMA_CATALOG.items()
                                 if table in selected}}, ensure_ascii=False, indent=2)
