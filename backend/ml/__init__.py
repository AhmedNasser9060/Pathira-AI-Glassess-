"""Server-side ML inference modules.

Each capability lives in its own submodule (`ocr`, later `detect`, `face`, ...).
Models are lazy-loaded on first call (see each submodule's `_configure` /
registry helper) so application startup stays fast and tests can mock the
inference functions without paying any model-load cost.
"""
