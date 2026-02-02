import re

def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'https?://\S+|www.\.\s+'," ",text)
    text=re.sub(r'<.*?>'," ",text)
    text=re.sub(r'[^a-xA-Z\s]'," ",text)

    text=re.sub(r'\s+'," ",text).strip()

    return text