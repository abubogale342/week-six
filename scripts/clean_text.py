from unstructured.partition.text import partition_text

def clean_text(text):
    elements = partition_text(text=text)
    cleaned_text = " ".join([
        el.text for el in elements
        if el.category in ["NarrativeText", "Title", "ListItem"]
    ])
    return cleaned_text

