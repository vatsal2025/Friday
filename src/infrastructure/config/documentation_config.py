"""Documentation Configuration Module.

This module provides configuration settings for the project's documentation system,
including standards, automated generation settings, and code review guidelines.
"""

# Documentation standards
DOCUMENTATION_STANDARDS = {
    "docstring_style": "Google",  # Google, NumPy, or reStructuredText
    "required_sections": [
        "Args",
        "Returns",
        "Raises",
        "Examples",
    ],
    "optional_sections": [
        "Notes",
        "References",
        "Warnings",
    ],
    "class_documentation": True,  # Document all classes
    "method_documentation": True,  # Document all methods
    "function_documentation": True,  # Document all functions
    "variable_documentation": False,  # Document module-level variables
    "max_line_length": 88,  # Match Black formatter default
    "enforce_capitalization": True,  # First letter of docstring should be capitalized
    "enforce_period": True,  # Docstring should end with a period
}

# Automated documentation generation settings
DOCUMENTATION_GENERATION = {
    "tool": "Sphinx",
    "output_format": ["HTML", "PDF"],
    "theme": "sphinx_rtd_theme",
    "auto_generate_on_commit": False,
    "include_private_members": False,
    "include_special_methods": False,
    "include_undocumented": True,
    "generate_api_docs": True,
    "generate_module_docs": True,
    "generate_index": True,
    "generate_authors": True,
    "generate_changes": True,
}

# Code review guidelines related to documentation
CODE_REVIEW_GUIDELINES = {
    "documentation": {
        "check_docstrings": True,
        "check_comments": True,
        "check_examples": True,
        "check_spelling": True,
        "check_grammar": False,  # Requires additional tools
        "check_style_consistency": True,
        "required_coverage": 80,  # Percentage of code that must be documented
    }
}

# Documentation file paths
DOCUMENTATION_PATHS = {
    "sphinx_source": "docs/source",
    "sphinx_build": "docs/build",
    "readme": "README.md",
    "contributing": "CONTRIBUTING.md",
    "code_of_conduct": "CODE_OF_CONDUCT.md",
    "changelog": "CHANGELOG.md",
    "license": "LICENSE",
}