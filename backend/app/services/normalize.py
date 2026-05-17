import re

from num2words import num2words


def _replace_currency(m: re.Match) -> str:
    integer_part = m.group(1).replace(",", "")
    cents_part = m.group(2)
    if cents_part:
        return f"{integer_part} dollars and {cents_part} cents"
    return f"{integer_part} dollars"


def _replace_bare_number(m: re.Match) -> str:
    token = m.group(0)
    if "." in token:
        try:
            return num2words(float(token))
        except Exception:
            return token
    else:
        try:
            return num2words(int(token))
        except Exception:
            return token


def normalize_for_tts(text: str) -> str:
    # 1. Currency
    text = re.sub(r"\$\s*([\d,]+)(?:\.(\d{2}))?", _replace_currency, text)

    # 2. Numbers with commas (bare, after currency already handled)
    text = re.sub(r"\b(\d{1,3}(?:,\d{3})+)\b", lambda m: m.group(0).replace(",", ""), text)

    # 3. Percentages
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)

    # 4. Dates — MM/DD/YYYY or DD/MM/YYYY and YYYY-MM-DD
    text = re.sub(r"\b(\d{2})/(\d{2})/(\d{4})\b", r"\1 \2 \3", text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", r"\1 \2 \3", text)

    # 5. Invoice/reference codes — all-caps+digits with hyphens
    text = re.sub(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+)\b", lambda m: m.group(0).replace("-", " "), text)

    # 6. Bare integers and decimals
    text = re.sub(r"\b\d+(?:\.\d+)?\b", _replace_bare_number, text)

    # 7. Normalise Unicode punctuation to ASCII equivalents
    text = text.replace("“", '"').replace("”", '"')  # curly double quotes
    text = text.replace("‘", "'").replace("’", "'")  # curly single quotes / apostrophe
    text = text.replace("—", " ").replace("–", " ")  # em-dash, en-dash

    # 8. Remove non-speech characters
    text = re.sub(r"[|\\^~{}\[\]<>#@*]", " ", text)

    # 9. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
