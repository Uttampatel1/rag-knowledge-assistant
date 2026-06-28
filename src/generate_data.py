"""Generate a small synthetic knowledge base so the demo runs out of the box.

Creates a few markdown documents about a fictional SaaS product ("Northwind
Analytics") under ``data/``. No external downloads, no proprietary content.

Run:  python -m src.generate_data
"""
from __future__ import annotations

import os

from .config import get_settings

DOCUMENTS: dict[str, str] = {
    "product_overview.md": """# Northwind Analytics — Product Overview

Northwind Analytics is a cloud platform that turns raw event data into
dashboards and alerts. Teams connect a data source, define metrics, and share
live dashboards with stakeholders.

The platform has three tiers: Starter, Growth, and Enterprise. Starter is free
for up to 3 users and 100,000 events per month. Growth costs $49 per user per
month and includes unlimited dashboards and email alerts. Enterprise adds SSO,
audit logs, and a dedicated success manager.

Data is encrypted in transit using TLS 1.2+ and at rest using AES-256.
""",
    "getting_started.md": """# Getting Started

To create your first dashboard, connect a data source from Settings > Sources.
Northwind supports PostgreSQL, BigQuery, Snowflake, and CSV upload.

After connecting a source, define a metric by choosing an aggregation (count,
sum, average) and an optional filter. Metrics refresh every 15 minutes on the
Growth tier and every 5 minutes on Enterprise.

You can invite teammates from Settings > Members. Roles are Viewer, Editor, and
Admin. Only Admins can manage billing and connected sources.
""",
    "billing_faq.md": """# Billing FAQ

Billing is monthly by default. Annual billing is available and gives a 20%
discount compared to monthly pricing.

You can upgrade or downgrade at any time. Upgrades take effect immediately and
are prorated. Downgrades take effect at the start of the next billing cycle.

We accept all major credit cards. Enterprise customers can pay by invoice with
net-30 terms. Refunds are available within 14 days of a charge for annual plans.
""",
    "security_and_data.md": """# Security and Data Handling

Northwind Analytics is SOC 2 Type II compliant. Access to production systems is
restricted and logged. Customer data is logically isolated per workspace.

Data retention is configurable. By default, raw events are retained for 13
months. Aggregated metrics are retained indefinitely unless deleted. Customers
can request full data deletion, which completes within 30 days.

Single sign-on (SSO) via SAML and SCIM provisioning are available on the
Enterprise tier. Two-factor authentication is available on all tiers.
""",
}


def write_sample_docs(data_dir: str | None = None) -> list[str]:
    data_dir = data_dir or get_settings().data_dir
    os.makedirs(data_dir, exist_ok=True)
    written: list[str] = []
    for name, content in DOCUMENTS.items():
        path = os.path.join(data_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(path)
    return written


if __name__ == "__main__":
    paths = write_sample_docs()
    print(f"Wrote {len(paths)} sample documents:")
    for p in paths:
        print(f"  - {p}")
