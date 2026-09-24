#!/usr/bin/env python3
"""The frozen E023/E028 claim-evidence formatter."""
from __future__ import annotations


def format_claim_evidence(claim: str, evidence: str) -> str:
    return ("Determine the relationship between the claim and the evidence.\n"
            "The evidence may support the claim, contradict the claim, or be insufficient to decide.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Choose exactly one class: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    print(format_claim_evidence(args.claim, args.evidence))
