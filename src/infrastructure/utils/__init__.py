"""Utility functions for the Friday AI Trading System.

This module provides common utility functions used throughout the application.
"""

import datetime
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Union


def generate_uuid() -> str:
    """Generate a UUID.

    Returns:
        str: A UUID string.
    """
    return str(uuid.uuid4())


def timestamp_now() -> int:
    """Get the current timestamp in seconds.

    Returns:
        int: The current timestamp in seconds.
    """
    return int(datetime.datetime.now().timestamp())


def format_datetime(
    dt: Optional[datetime.datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format a datetime object as a string.

    Args:
        dt: The datetime object to format. If None, the current datetime is used.
        fmt: The format string to use.

    Returns:
        str: The formatted datetime string.
    """
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime(fmt)


def parse_datetime(dt_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime.datetime:
    """Parse a datetime string into a datetime object.

    Args:
        dt_str: The datetime string to parse.
        fmt: The format string to use.

    Returns:
        datetime.datetime: The parsed datetime object.

    Raises:
        ValueError: If the datetime string cannot be parsed.
    """
    return datetime.datetime.strptime(dt_str, fmt)


def ensure_dir(directory: str) -> None:
    """Ensure that a directory exists.

    Args:
        directory: The directory path.

    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """Save data to a JSON file.

    Args:
        data: The data to save.
        file_path: The path to the JSON file.
        indent: The indentation level for the JSON file.

    Returns:
        None
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def snake_to_camel(snake_str: str) -> str:
    """Convert a snake_case string to camelCase.

    Args:
        snake_str: The snake_case string to convert.

    Returns:
        str: The camelCase string.
    """
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(camel_str: str) -> str:
    """Convert a camelCase string to snake_case.

    Args:
        camel_str: The camelCase string to convert.

    Returns:
        str: The snake_case string.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: The dictionary to flatten.
        parent_key: The parent key.
        sep: The separator to use between keys.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflatten a flattened dictionary.

    Args:
        d: The flattened dictionary.
        sep: The separator used between keys.

    Returns:
        Dict[str, Any]: The unflattened dictionary.
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def dict_to_object(d: Dict[str, Any]) -> Any:
    """Convert a dictionary to an object.

    Args:
        d: The dictionary to convert.

    Returns:
        Any: The converted object.
    """

    class DictObject:
        def __init__(self, d: Dict[str, Any]):
            for key, value in d.items():
                if isinstance(value, dict):
                    value = dict_to_object(value)
                setattr(self, key, value)

    return DictObject(d)


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.

    Args:
        s: The string to truncate.
        max_length: The maximum length of the string.
        suffix: The suffix to add to the truncated string.

    Returns:
        str: The truncated string.
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def is_valid_email(email: str) -> bool:
    """Check if a string is a valid email address.

    Args:
        email: The string to check.

    Returns:
        bool: True if the string is a valid email address, False otherwise.
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL.

    Args:
        url: The string to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    pattern = r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$"
    return bool(re.match(pattern, url))


def safe_divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Safely divide two numbers.

    Args:
        a: The numerator.
        b: The denominator.

    Returns:
        Union[int, float]: The result of the division, or 0 if the denominator is 0.
    """
    return a / b if b != 0 else 0


def clamp(value: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> Union[int, float]:
    """Clamp a value between a minimum and maximum value.

    Args:
        value: The value to clamp.
        min_value: The minimum value.
        max_value: The maximum value.

    Returns:
        Union[int, float]: The clamped value.
    """
    return max(min_value, min(value, max_value))


def format_currency(amount: Union[int, float], currency: str = "INR", locale: str = "en_IN") -> str:
    """Format a currency amount.

    Args:
        amount: The amount to format.
        currency: The currency code.
        locale: The locale to use for formatting.

    Returns:
        str: The formatted currency amount.
    """
    import locale as locale_module
    import babel.numbers

    try:
        locale_module.setlocale(locale_module.LC_ALL, locale)
        return babel.numbers.format_currency(amount, currency, locale=locale)
    except (locale_module.Error, babel.core.UnknownLocaleError):
        # Fallback to simple formatting
        return f"{currency} {amount:,.2f}"


def format_number(number: Union[int, float], decimal_places: int = 2, thousand_separator: str = ",") -> str:
    """Format a number.

    Args:
        number: The number to format.
        decimal_places: The number of decimal places to show.
        thousand_separator: The character to use as a thousand separator.

    Returns:
        str: The formatted number.
    """
    return f"{number:,.{decimal_places}f}".replace(",", thousand_separator)


def parse_number(number_str: str, decimal_separator: str = ".", thousand_separator: str = ",") -> Union[int, float]:
    """Parse a formatted number string into a number.

    Args:
        number_str: The number string to parse.
        decimal_separator: The character used as a decimal separator.
        thousand_separator: The character used as a thousand separator.

    Returns:
        Union[int, float]: The parsed number.

    Raises:
        ValueError: If the number string cannot be parsed.
    """
    # Remove thousand separators
    number_str = number_str.replace(thousand_separator, "")
    # Replace decimal separator with a dot
    if decimal_separator != ".":
        number_str = number_str.replace(decimal_separator, ".")
    # Parse the number
    try:
        return int(number_str) if "." not in number_str else float(number_str)
    except ValueError:
        raise ValueError(f"Could not parse number: {number_str}")