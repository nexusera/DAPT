import re
import json
import logging

logger = logging.getLogger(__name__)

def clean_json_content(content):
    """
    Cleans common JSON syntax errors found in the schema file:
    - Removes comments (// and #)
    - Fixes trailing commas
    - Replaces Chinese commas
    - Fixes malformed lists
    - Fixes missing commas between objects
    """
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'\s*#.*', '', content)
    content = re.sub(r'\[\s*,', '[', content)
    content = re.sub(r'}\s*"', '}, "', content)
    content = content.replace('，', ',')
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return content

def load_schema(filepath):
    """Safe loading of schema/keys definition file"""
    try:
        from pathlib import Path
        raw_content = Path(filepath).read_text(encoding='utf-8')
        cleaned_content = clean_json_content(raw_content)
        return json.loads(cleaned_content)
    except Exception as e:
        logger.error(f"Failed to load schema from {filepath}: {e}")
        return {}
