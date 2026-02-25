#!/usr/bin/env python3
"""Quick standalone test: verify DB connection works while Msty is running."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

print("=== Test 1: get_msty_paths() ===")
from src.paths import get_msty_paths
paths = get_msty_paths()
print(json.dumps(paths, indent=2))
assert paths.get("database"), f"FAIL: database not found, got: {paths}"
print(f"OK: database = {paths['database']}")

print("\n=== Test 2: read_msty_database(query_type='stats') ===")
from src.server import read_msty_database
result = read_msty_database(query_type="stats")
data = json.loads(result)
print(json.dumps(data, indent=2)[:500])
assert "query_type" in data, f"FAIL: 'query_type' not in result: {list(data.keys())}"
print("OK: query_type present")

print("\n=== Test 3: list_configured_tools() ===")
from src.server import list_configured_tools
result2 = list_configured_tools()
data2 = json.loads(result2)
print(json.dumps(data2, indent=2)[:500])
assert "tools" in data2, f"FAIL: 'tools' not in result: {list(data2.keys())}"
assert "tool_count" in data2, f"FAIL: 'tool_count' not in result: {list(data2.keys())}"
print(f"OK: tools={data2['tool_count']}")

print("\n=== ALL TESTS PASSED ===")
