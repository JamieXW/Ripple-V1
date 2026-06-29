"""Ripple — a hybrid graph + semantic code-intelligence engine for Python.

Ripple indexes a Python repository into two complementary structures from a single
AST parse: a call/import/inherit *graph* (answers change-impact questions via
traversal) and a *semantic index* of embedded code chunks (answers where/how
questions). A retrieval layer fuses the two; results always carry file:line citations.
"""

__version__ = "0.1.0"
