from src.scorers import _extract_category, exact_category_match


def test_extract_category_exact_match():
    assert _extract_category("Incident") == "incident"
    assert _extract_category("ask for a Request") == "request"
    assert _extract_category("PROBLEM happened") == "problem"
    assert _extract_category("please do a Change") == "change"


def test_extract_category_unknown():
    assert _extract_category("no match here") == "unknown"


def test_exact_category_match_with_string_output():
    expectations = {"type": "Incident"}
    assert exact_category_match("Incident", expectations)
    assert exact_category_match("incident (outage)", expectations)


def test_exact_category_match_with_dict_output():
    expectations = {"type": "Request"}
    assert exact_category_match({"type": "Request"}, expectations)


def test_extract_category_leading_trailing_spaces():
    """Test that leading/trailing spaces are handled."""
    assert _extract_category("  Incident  ") == "incident"
    assert _extract_category("\tRequest\n") == "request"


def test_extract_category_mixed_case():
    """Test various case combinations."""
    assert _extract_category("INCIDENT") == "incident"
    assert _extract_category("InCiDeNt") == "incident"
    assert _extract_category("rEqUeSt") == "request"


def test_extract_category_partial_match():
    """Test that partial matches work correctly."""
    assert _extract_category("problem-solving approach") == "problem"
    assert _extract_category("change request submitted") == "change"


def test_extract_category_empty_string():
    """Test behavior with empty string."""
    assert _extract_category("") == "unknown"
    assert _extract_category("   ") == "unknown"


def test_extract_category_none_input():
    """Test that None is handled gracefully."""
    assert _extract_category(None) == "unknown"


def test_extract_category_first_match_wins():
    """Test that first matched category is returned."""
    # If multiple categories appear, first one wins
    assert _extract_category("incident request") == "incident"
    assert _extract_category("problem change") == "problem"


def test_exact_category_match_case_insensitive():
    """Test case-insensitive matching."""
    expectations = {"type": "incident"}
    assert exact_category_match("INCIDENT", expectations)
    assert exact_category_match("Incident", expectations)
    assert exact_category_match("iNcIdEnT", expectations)


def test_exact_category_match_extra_text():
    """Test matching with extra surrounding text."""
    expectations = {"type": "Request"}
    assert exact_category_match("This is a Request for help", expectations)
    assert exact_category_match("REQUEST (urgent)", expectations)


def test_exact_category_match_false_cases():
    """Test cases that should NOT match."""
    expectations = {"type": "Incident"}
    assert not exact_category_match("Request", expectations)
    assert not exact_category_match("Problem", expectations)
    assert not exact_category_match("unknown", expectations)


def test_exact_category_match_missing_type_in_expectations():
    """Test behavior when type is missing from expectations."""
    expectations = {}
    # Should default to empty string when type is missing
    assert not exact_category_match("Incident", expectations)


def test_exact_category_match_dict_missing_type():
    """Test when dict output is missing 'type' key."""
    expectations = {"type": "Incident"}
    output = {"status": "resolved"}  # No 'type' key
    # Should fallback to string extraction on the dict
    assert not exact_category_match(output, expectations)
