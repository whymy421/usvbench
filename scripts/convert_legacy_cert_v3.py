"""Read-only legacy certificate conversion into a missing-fields report.

This command only prints a diagnostic view to stdout. It never edits the
input, invents missing evidence, or emits a schema-v3 certificate.
"""
import sys

from validate_cert_v3 import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--legacy-report"]))
