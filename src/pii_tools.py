"""
Msty Admin MCP - PII Scrubbing Tools

Tools for detecting and removing Personally Identifiable Information (PII)
from text and files, supporting Msty's privacy-first approach.
"""

import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# PII PATTERNS
# ============================================================================

PII_PATTERNS = {
    "email": {
        "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "description": "Email addresses",
        "severity": "high",
        "replacement": "[EMAIL_REDACTED]"
    },
    "phone_us": {
        "pattern": r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "description": "US Phone numbers",
        "severity": "high",
        "replacement": "[PHONE_REDACTED]"
    },
    "phone_intl": {
        "pattern": r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',
        "description": "International phone numbers",
        "severity": "high",
        "replacement": "[PHONE_REDACTED]"
    },
    "ssn": {
        "pattern": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        "description": "Social Security Numbers",
        "severity": "critical",
        "replacement": "[SSN_REDACTED]"
    },
    "credit_card": {
        "pattern": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        "description": "Credit card numbers",
        "severity": "critical",
        "replacement": "[CC_REDACTED]"
    },
    "ip_address": {
        "pattern": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "description": "IP addresses",
        "severity": "medium",
        "replacement": "[IP_REDACTED]"
    },
    "date_of_birth": {
        "pattern": r'\b(?:DOB|Date of Birth|Born)[:\s]*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
        "description": "Dates of birth",
        "severity": "high",
        "replacement": "[DOB_REDACTED]"
    },
    "address": {
        "pattern": r'\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)\b',
        "description": "Street addresses",
        "severity": "high",
        "replacement": "[ADDRESS_REDACTED]"
    },
    "zip_code": {
        "pattern": r'\b\d{5}(?:-\d{4})?\b',
        "description": "US ZIP codes",
        "severity": "low",
        "replacement": "[ZIP_REDACTED]"
    },
    "drivers_license": {
        "pattern": r'\b(?:DL|Driver\'?s?\s*License)[:\s#]*[A-Z0-9]{5,15}\b',
        "description": "Driver's license numbers",
        "severity": "critical",
        "replacement": "[DL_REDACTED]"
    },
    "passport": {
        "pattern": r'\b(?:Passport)[:\s#]*[A-Z0-9]{6,12}\b',
        "description": "Passport numbers",
        "severity": "critical",
        "replacement": "[PASSPORT_REDACTED]"
    },
    "bank_account": {
        "pattern": r'\b(?:Account|Acct)[:\s#]*\d{8,17}\b',
        "description": "Bank account numbers",
        "severity": "critical",
        "replacement": "[ACCOUNT_REDACTED]"
    },
    "medical_record": {
        "pattern": r'\b(?:MRN|Medical Record|Patient ID)[:\s#]*[A-Z0-9]{6,15}\b',
        "description": "Medical record numbers",
        "severity": "critical",
        "replacement": "[MRN_REDACTED]"
    },
}


# ============================================================================
# PII SCANNING
# ============================================================================

def scan_for_pii(
    text: str,
    categories: Optional[List[str]] = None,
    min_severity: str = "low"
) -> Dict[str, Any]:
    """
    Scan text for personally identifiable information.

    Args:
        text: Text to scan
        categories: Specific PII categories to check (None = all)
        min_severity: Minimum severity to report (low, medium, high, critical)

    Returns:
        Dict with PII findings
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "text_length": len(text),
        "findings": [],
        "summary": {
            "total_pii_found": 0,
            "by_category": {},
            "by_severity": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
    }

    severity_levels = ["low", "medium", "high", "critical"]
    min_severity_idx = severity_levels.index(min_severity)

    patterns_to_check = PII_PATTERNS
    if categories:
        patterns_to_check = {
            k: v for k, v in PII_PATTERNS.items()
            if k in categories
        }

    for category, config in patterns_to_check.items():
        severity = config["severity"]
        severity_idx = severity_levels.index(severity)

        if severity_idx < min_severity_idx:
            continue

        try:
            pattern = re.compile(config["pattern"], re.IGNORECASE)
            matches = pattern.findall(text)

            if matches:
                # Deduplicate
                unique_matches = list(set(matches))

                for match in unique_matches:
                    # Mask the actual value for security
                    masked = mask_pii_value(match, category)

                    result["findings"].append({
                        "category": category,
                        "description": config["description"],
                        "severity": severity,
                        "masked_value": masked,
                        "count": matches.count(match)
                    })

                    result["summary"]["total_pii_found"] += matches.count(match)
                    result["summary"]["by_severity"][severity] += matches.count(match)

                    if category not in result["summary"]["by_category"]:
                        result["summary"]["by_category"][category] = 0
                    result["summary"]["by_category"][category] += matches.count(match)

        except re.error as e:
            logger.warning(f"Regex error for {category}: {e}")

    # Risk assessment
    if result["summary"]["by_severity"]["critical"] > 0:
        result["risk_level"] = "critical"
        result["recommendation"] = "Immediate scrubbing required - critical PII detected"
    elif result["summary"]["by_severity"]["high"] > 0:
        result["risk_level"] = "high"
        result["recommendation"] = "Scrubbing strongly recommended - sensitive PII detected"
    elif result["summary"]["total_pii_found"] > 0:
        result["risk_level"] = "medium"
        result["recommendation"] = "Consider scrubbing before sharing"
    else:
        result["risk_level"] = "low"
        result["recommendation"] = "No significant PII detected"

    return result


def mask_pii_value(value: str, category: str) -> str:
    """Mask a PII value for safe display."""
    if len(value) <= 4:
        return "*" * len(value)

    if category in ["email"]:
        parts = value.split("@")
        if len(parts) == 2:
            return f"{parts[0][:2]}***@{parts[1]}"

    if category in ["phone_us", "phone_intl"]:
        return f"***-***-{value[-4:]}"

    if category in ["ssn"]:
        return f"***-**-{value[-4:]}"

    if category in ["credit_card"]:
        return f"****-****-****-{value[-4:]}"

    # Default masking
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


# ============================================================================
# PII SCRUBBING
# ============================================================================

def scrub_pii(
    text: str,
    categories: Optional[List[str]] = None,
    replacement_style: str = "category"
) -> Dict[str, Any]:
    """
    Remove PII from text.

    Args:
        text: Text to scrub
        categories: Specific PII categories to scrub (None = all)
        replacement_style: How to replace PII (category, generic, hash)

    Returns:
        Dict with scrubbed text and statistics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "original_length": len(text),
        "scrubbed_text": text,
        "replacements_made": 0,
        "by_category": {}
    }

    patterns_to_scrub = PII_PATTERNS
    if categories:
        patterns_to_scrub = {
            k: v for k, v in PII_PATTERNS.items()
            if k in categories
        }

    scrubbed = text

    for category, config in patterns_to_scrub.items():
        try:
            pattern = re.compile(config["pattern"], re.IGNORECASE)

            # Determine replacement
            if replacement_style == "category":
                replacement = config["replacement"]
            elif replacement_style == "generic":
                replacement = "[REDACTED]"
            elif replacement_style == "hash":
                def hash_replace(match):
                    import hashlib
                    h = hashlib.md5(match.group(0).encode()).hexdigest()[:8]
                    return f"[{category.upper()}_{h}]"
                replacement = hash_replace
            else:
                replacement = config["replacement"]

            # Count matches before replacement
            matches = pattern.findall(scrubbed)
            if matches:
                result["by_category"][category] = len(matches)
                result["replacements_made"] += len(matches)

            # Apply replacement
            if callable(replacement):
                scrubbed = pattern.sub(replacement, scrubbed)
            else:
                scrubbed = pattern.sub(replacement, scrubbed)

        except re.error as e:
            logger.warning(f"Regex error for {category}: {e}")

    result["scrubbed_text"] = scrubbed
    result["scrubbed_length"] = len(scrubbed)
    result["length_change"] = len(scrubbed) - len(text)

    return result


# ============================================================================
# PII REPORTING
# ============================================================================

def generate_pii_report(
    texts: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive PII exposure report for multiple texts.

    Args:
        texts: List of texts to analyze
        labels: Optional labels for each text (e.g., filenames)

    Returns:
        Dict with comprehensive PII report
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_texts_analyzed": len(texts),
        "total_characters_analyzed": sum(len(t) for t in texts),
        "individual_reports": [],
        "aggregate_summary": {
            "total_pii_instances": 0,
            "texts_with_pii": 0,
            "texts_clean": 0,
            "by_category": {},
            "by_severity": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            }
        },
        "risk_assessment": {}
    }

    for i, text in enumerate(texts):
        label = labels[i] if labels and i < len(labels) else f"Text {i + 1}"

        scan_result = scan_for_pii(text)

        individual = {
            "label": label,
            "length": len(text),
            "pii_found": scan_result["summary"]["total_pii_found"],
            "risk_level": scan_result["risk_level"],
            "findings_count": len(scan_result["findings"])
        }

        result["individual_reports"].append(individual)

        # Aggregate
        if scan_result["summary"]["total_pii_found"] > 0:
            result["aggregate_summary"]["texts_with_pii"] += 1
        else:
            result["aggregate_summary"]["texts_clean"] += 1

        result["aggregate_summary"]["total_pii_instances"] += scan_result["summary"]["total_pii_found"]

        for category, count in scan_result["summary"]["by_category"].items():
            if category not in result["aggregate_summary"]["by_category"]:
                result["aggregate_summary"]["by_category"][category] = 0
            result["aggregate_summary"]["by_category"][category] += count

        for severity, count in scan_result["summary"]["by_severity"].items():
            result["aggregate_summary"]["by_severity"][severity] += count

    # Overall risk assessment
    agg = result["aggregate_summary"]
    if agg["by_severity"]["critical"] > 0:
        result["risk_assessment"] = {
            "overall_risk": "critical",
            "recommendation": "Immediate action required - critical PII exposure detected",
            "priority_actions": [
                "Scrub all critical PII immediately",
                "Review access controls for affected content",
                "Consider incident response procedures"
            ]
        }
    elif agg["by_severity"]["high"] > 0:
        result["risk_assessment"] = {
            "overall_risk": "high",
            "recommendation": "Prompt action recommended - sensitive PII exposure",
            "priority_actions": [
                "Scrub high-severity PII",
                "Review content before sharing",
                "Implement PII scanning in workflows"
            ]
        }
    elif agg["total_pii_instances"] > 0:
        result["risk_assessment"] = {
            "overall_risk": "moderate",
            "recommendation": "Review and scrub as appropriate",
            "priority_actions": [
                "Review flagged content",
                "Scrub before external sharing"
            ]
        }
    else:
        result["risk_assessment"] = {
            "overall_risk": "low",
            "recommendation": "No significant PII detected",
            "priority_actions": [
                "Continue regular privacy practices"
            ]
        }

    # Generate compliance notes
    result["compliance_notes"] = {
        "gdpr_relevant": agg["by_severity"]["high"] + agg["by_severity"]["critical"] > 0,
        "hipaa_relevant": agg["by_category"].get("medical_record", 0) > 0,
        "pci_relevant": agg["by_category"].get("credit_card", 0) > 0,
    }

    return result


__all__ = [
    "PII_PATTERNS",
    "scan_for_pii",
    "scrub_pii",
    "generate_pii_report",
    "mask_pii_value",
]
