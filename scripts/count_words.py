import pandas as pd

# --- 2. Calculate Word Count ---
def count_words(text):
    # Handle NaN or non-string inputs by converting to string and stripping whitespace
    # Then split by whitespace and count the resulting parts.
    # An empty string after stripping will result in a list [''] so we check if it's truly empty.
    if pd.isna(text) or not str(text).strip():
        return 0

    return len(str(text).split())
    