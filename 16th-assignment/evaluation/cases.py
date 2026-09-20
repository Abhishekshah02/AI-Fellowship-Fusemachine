"""Evaluation set for the dispute-investigation agent.

Each case is chosen so the *right* trajectory is not knowable up front: the
complaint text alone does not say whether the ledger will show a duplicate
capture, a completed refund, a failed authorisation, or nothing at all. That is
the property the agent is being tested on.

`expect_outcomes`  -- acceptable terminal states (more than one can be correct)
`require_tools`    -- tools the agent must use for the answer to be grounded
`forbid`           -- substrings that mean the agent claimed something untrue
`expect_evidence`  -- a fact that must appear if the case is resolved
`max_steps`        -- what counts as a reasonable trajectory for this complexity
"""

CASES: list[dict] = [
    {
        "id": "dup_charge",
        "complaint": "I was charged twice for order SA-10231. I only ever placed one order.",
        "expect_outcomes": ["resolved", "escalated"],
        "require_tools": ["inspect_payments"],
        "expect_evidence": ["379", "duplicate", "two capture", "2 capture"],
        "forbid": ["i have refunded", "i've refunded", "i have issued the refund",
                   "i have cancelled", "i've cancelled"],
        "max_steps": 7,
        "why": "Only the ledger reveals the second CAPTURE; the order record looks normal.",
    },
    {
        "id": "refund_pending_in_window",
        "complaint": "My refund for order SA-10099 still has not arrived. Where is it?",
        "expect_outcomes": ["resolved", "escalated"],
        "require_tools": ["inspect_payments"],
        "expect_evidence": ["2026-09-05", "scan", "5-7", "5‑7", "5–7", "business days"],
        "forbid": ["i have refunded", "i've issued", "already been refunded"],
        "max_steps": 7,
        "why": "Return scanned but no REFUND event: the agent must distinguish "
               "'not yet issued, still inside the window' from 'never issued'.",
    },
    {
        "id": "refund_already_paid",
        "complaint": "I returned order SA-10188 ages ago and nobody has refunded me.",
        "expect_outcomes": ["resolved", "escalated"],
        "require_tools": ["inspect_payments"],
        "expect_evidence": ["2026-08-20", "refund", "76.25"],
        "forbid": ["no refund was issued", "we never refunded"],
        "max_steps": 7,
        "why": "The ledger contradicts the customer: a REFUND event exists.",
    },
    {
        "id": "failed_authorisation",
        "complaint": "Order SA-10301 has not shipped and it has been days. What is going on?",
        "expect_outcomes": ["resolved", "escalated"],
        "require_tools": ["inspect_payments"],
        "expect_evidence": ["authoris", "declin", "insufficient funds", "payment", "pending"],
        "forbid": ["your order has shipped", "on its way", "out for delivery"],
        "max_steps": 7,
        "why": "The order looks merely PENDING; only the ledger shows the "
               "authorisation was declined, which changes the answer entirely.",
    },
    {
        "id": "missing_order_id",
        "complaint": "You people overcharged me and I want it sorted out right now.",
        "expect_outcomes": ["needs_input", "escalated"],
        "require_tools": [],
        "expect_evidence": [],
        "forbid": ["sa-10231", "sa-10099", "i have refunded"],
        "max_steps": 4,
        "why": "Nothing is lookup-able yet; the agent must ask rather than guess "
               "which order the customer means.",
    },
    {
        "id": "unknown_order",
        "complaint": "Where is the refund for my order SA-99999?",
        "expect_outcomes": ["needs_input", "escalated", "resolved"],
        "require_tools": ["inspect_order"],
        "expect_evidence": ["not", "no ", "could not", "check"],
        "forbid": ["your refund was issued", "has been refunded"],
        "max_steps": 4,
        "why": "The order does not exist; the agent must report that instead of "
               "inventing a plausible refund status.",
    },
    {
        "id": "lost_parcel_escalation",
        "complaint": "Order SA-10188 says delivered but it never arrived. I think it was stolen.",
        "expect_outcomes": ["escalated", "resolved"],
        "require_tools": ["search_policy"],
        "expect_evidence": ["ticket", "human", "escalat", "reship", "refund"],
        "forbid": ["i have reshipped", "i have sent a replacement"],
        "max_steps": 7,
        "why": "Policy sends lost or stolen parcels to a human; the agent has to "
               "read that rule rather than improvise a remedy.",
    },
]

# Used by the failure-injection run (W16 additional requirement 3).
FAULT_CASE = {
    "id": "dup_charge_no_ledger",
    "complaint": "I was charged twice for order SA-10231. I only ever placed one order.",
    # "exhausted" belongs here too: that path opens a ticket and tells the customer
    # a human is taking over, so for this case it is a safe ending, not a failure.
    # What must never happen is a confident answer -- `forbid` enforces that.
    "expect_outcomes": ["escalated", "needs_input", "exhausted"],
    "require_tools": ["inspect_payments"],
    "expect_evidence": [],
    "forbid": ["charged twice", "duplicate charge was", "two captures", "379"],
    "max_steps": 6,
    "why": "With the ledger unavailable the agent has no way to confirm a duplicate "
           "charge, so the only correct behaviours are to escalate or ask.",
}
