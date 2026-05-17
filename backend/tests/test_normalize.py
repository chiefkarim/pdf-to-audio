from app.services.normalize import normalize_for_tts


def test_currency_dollars_and_cents():
    assert "50 dollars and 30 cents" in normalize_for_tts("$50.30")


def test_currency_whole_dollars():
    assert "100 dollars" in normalize_for_tts("$100")


def test_currency_with_commas():
    assert "1234" in normalize_for_tts("$1,234")


def test_percentage():
    assert "10 percent" in normalize_for_tts("10%")


def test_date_slash():
    result = normalize_for_tts("05/17/2026")
    assert "/" not in result


def test_invoice_code():
    result = normalize_for_tts("INV-001")
    assert "-" not in result
    assert "INV" in result


def test_bare_integer():
    result = normalize_for_tts("42 items")
    assert "forty" in result.lower()


def test_removes_special_chars():
    result = normalize_for_tts("total: $100 | ref: #ABC")
    assert "|" not in result
    assert "#" not in result


def test_collapses_whitespace():
    assert normalize_for_tts("  hello   world  ") == "hello world"
