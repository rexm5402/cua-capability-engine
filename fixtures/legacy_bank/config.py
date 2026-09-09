"""Per-tenant configuration for the legacy bank fixture.

The whole point of this file: the two "tenants" are the SAME vendor product
deployed at two different institutions. Everything that differs between them
-- branding, field order, labels, and the exact wording of every business
message -- lives here in one dict, so a reviewer can see the delta at a glance
and no message string is hardcoded anywhere in app.py or the templates.
"""

# Field NAMES are deliberately meaningless in both tenants (f_7, ctl00_x3 ...).
# They are shared across tenants on purpose: an automation that keys off the
# name attribute would pass tenant A and tenant B alike and would still be
# wrong -- the real, stable handles are the labels and the label-adjacent
# table cells, which DO differ.
SEARCH_FIELDS = {
    "member_id": "f_7",
    "branch": "f_3",
    "include_closed": "f_9",
}

SUBACCOUNT_FIELDS = {
    "product": "ctl00_x3",
    "nickname": "ctl00_x4",
    "initial_deposit": "f_21",
    "funding_source": "f_22",
    "statement_pref": "ctl00_x9",
    "effective_date": "f_24",
}

TENANTS = {
    "a": {
        "code": "a",
        "institution": "First Meridian Credit Union",
        "product": "CoreTeller 4.2",
        "colors": {
            "chrome": "#000080",   # navy title bars, Windows 95 energy
            "chrome_text": "#ffffff",
            "page": "#c0c0c0",
            "panel": "#e8e8e8",
            "accent": "#800000",
        },
        # Labels: adjacent-cell text is the ONLY reliable handle, so it differs.
        "labels": {
            "member_id": "Member ID",
            "branch": "Branch Code",
            "include_closed": "Include Closed Accounts",
            "search_button": "Retrieve",
            "product": "Product Code",
            "nickname": "Account Nickname",
            "initial_deposit": "Initial Deposit",
            "funding_source": "Funding Source",
            "statement_pref": "Statement Delivery",
            "effective_date": "Effective Date",
            "submit_button": "Submit Request",
            "savings": "Savings Balance",
            "checking": "Checking Balance",
            "reference": "Reference Number",
        },
        # Search form field ORDER. Tenant B reverses the first two.
        "search_order": ["member_id", "branch", "include_closed"],
        # Every user-visible business message, including fault-injected ones.
        "messages": {
            "not_found": "No records located.",
            "perm_denied": "You are not authorized to view this record.",
            "validation": "Field in error: Initial Deposit must be a numeric amount of 25.00 or greater.",
            "dialog_title": "System Notice",
            "dialog_body": "A scheduled maintenance window is pending. Click Continue to proceed.",
            "dialog_button": "Continue",
            "timeout": "Your session has expired.",
            "error": "Unhandled exception in module TLR_CORE.SUBACCT (code 0x8007000E).",
            "login_prompt": "Sign on to CoreTeller",
            "confirm_headline": "Sub-account request accepted.",
        },
        "ref_prefix": "FM",
    },
    "b": {
        "code": "b",
        "institution": "Harborline Savings Bank",
        "product": "CoreTeller 4.2",
        "colors": {
            "chrome": "#004000",   # green chrome instead of navy
            "chrome_text": "#ffff00",
            "page": "#d4d0c8",
            "panel": "#f0efe4",
            "accent": "#003366",
        },
        "labels": {
            "member_id": "Member Number",
            "branch": "Branch",
            "include_closed": "Show Closed Accounts",
            "search_button": "Find",
            "product": "Account Product",
            "nickname": "Nickname",
            "initial_deposit": "Opening Deposit",
            "funding_source": "Source of Funds",
            "statement_pref": "Statements",
            "effective_date": "Open Date",
            "submit_button": "Save",
            "savings": "Savings",
            "checking": "Checking",
            "reference": "Confirmation No.",
        },
        # DIFFERENT ORDER: branch first. Positional automation breaks here.
        "search_order": ["branch", "member_id", "include_closed"],
        "messages": {
            "not_found": "0 records found.",
            "perm_denied": "Access to this account is restricted.",
            "validation": "Invalid entry: Opening Deposit requires a numeric value of at least 25.00.",
            "dialog_title": "System Notice",
            "dialog_body": "An operator advisory is queued for this workstation. Click Continue to proceed.",
            "dialog_button": "Continue",
            "timeout": "Your session has expired.",
            "error": "Unhandled exception in module TLR_CORE.SUBACCT (code 0x8007000E).",
            "login_prompt": "Harborline Teller Sign-On",
            "confirm_headline": "Your request has been recorded.",
        },
        "ref_prefix": "HL",
    },
}

# Fully synthetic members. Names, IDs and balances are invented; nothing here
# resembles a real person or a real account.
MEMBERS = {
    "10001": {
        "id": "10001", "name": "Dana Whitfield", "branch": "004",
        "savings": "4,182.55", "checking": "912.40", "status": "ACTIVE",
        "opened": "1998-03-14",
    },
    "10002": {
        "id": "10002", "name": "Marcus Enfield", "branch": "011",
        "savings": "217.03", "checking": "3,540.18", "status": "ACTIVE",
        "opened": "2003-11-02",
    },
    "10003": {
        "id": "10003", "name": "Priya Raghunath", "branch": "004",
        "savings": "26,904.12", "checking": "1,180.77", "status": "ACTIVE",
        "opened": "2011-06-21",
    },
    "10004": {
        "id": "10004", "name": "Oscar Delacroix", "branch": "027",
        "savings": "0.00", "checking": "58.19", "status": "DORMANT",
        "opened": "1994-09-30",
    },
    "10005": {
        "id": "10005", "name": "Ingrid Solheim", "branch": "011",
        "savings": "8,775.00", "checking": "0.00", "status": "ACTIVE",
        "opened": "2019-01-08",
    },
}

PRODUCT_CODES = [
    ("SAV-02", "Regular Savings"),
    ("SAV-07", "High Yield Savings"),
    ("CHK-01", "Basic Checking"),
    ("CD-12", "12 Month Certificate"),
]

FUNDING_SOURCES = [
    ("XFER", "Transfer from existing account"),
    ("CASH", "Cash deposit at branch"),
    ("CHK", "Check deposit"),
]

INJECT_MODES = (
    "not_found", "perm_denied", "validation",
    "dialog", "timeout", "slow", "error",
)

# Injected conditions that can fire on ANY page vs. ones tied to a specific step.
UNIVERSAL_MODES = {"dialog", "timeout", "slow", "error"}
