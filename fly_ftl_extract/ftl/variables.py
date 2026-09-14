"""Which ``$variables`` a Fluent entry needs from the code that formats it.

Port of ``common/src/fluent_variables.rs``:

* ``{ $name }`` in the pattern, a selector, a nested placeable or a function argument counts;
* ``{ other }`` pulls in the value of ``other``; ``{ other.attr }`` only that attribute;
* ``{ -term }`` pulls in the term, but nothing inside a term is a caller variable — the
  arguments of the reference itself are evaluated in the caller's scope and do count;
* attributes of the entry being formatted are not visited;
* every (entry, attribute, scope) triple is visited once, so cycles terminate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from fluent.syntax import ast as fl


class FluentEntries(Protocol):
    """Lookup of the messages and terms an entry may reference."""

    def message(self, name: str) -> fl.Message | None:
        """The message called ``name`` or ``None``."""
        ...

    def term(self, name: str) -> fl.Term | None:
        """The term called ``name`` (without the leading ``-``) or ``None``."""
        ...


@dataclass(frozen=True)
class UnknownReference:
    """A reference to a message or term that does not exist."""

    kind: str  # "message" | "term"
    name: str


@dataclass
class CollectedVariables:
    """Result of a traversal."""

    variables: set[str] = field(default_factory=set)
    referenced_messages: set[str] = field(default_factory=set)
    unknown_references: list[UnknownReference] = field(default_factory=list)


class _Scope(Enum):
    CALLER = 1
    TERM = 2


class _Collector:
    def __init__(self, entries: FluentEntries) -> None:
        self.entries = entries
        self.result = CollectedVariables()
        self.visited: set[tuple[str, str, str | None, _Scope]] = set()

    def visit_message(self, message: fl.Message, attribute: str | None, scope: _Scope) -> None:
        pattern = _message_pattern(message, attribute)
        if pattern is None:
            return
        marker = ("message", message.id.name, attribute, scope)
        if marker in self.visited:
            return
        self.visited.add(marker)
        self.visit_pattern(pattern, scope)

    def visit_term(self, term: fl.Term, attribute: str | None, scope: _Scope) -> None:
        pattern = _term_pattern(term, attribute)
        if pattern is None:
            return
        marker = ("term", term.id.name, attribute, scope)
        if marker in self.visited:
            return
        self.visited.add(marker)
        self.visit_pattern(pattern, scope)

    def visit_pattern(self, pattern: fl.Pattern, scope: _Scope) -> None:
        for element in pattern.elements:
            if isinstance(element, fl.Placeable):
                self.visit_expression(element.expression, scope)

    def visit_expression(self, expression: fl.Expression | fl.Placeable, scope: _Scope) -> None:
        if isinstance(expression, fl.SelectExpression):
            self.visit_inline(expression.selector, scope)
            for variant in expression.variants:
                self.visit_pattern(variant.value, scope)
        elif isinstance(expression, fl.Placeable):
            self.visit_expression(expression.expression, scope)
        else:
            self.visit_inline(expression, scope)

    def visit_inline(self, inline: fl.SyntaxNode, scope: _Scope) -> None:
        if isinstance(inline, fl.VariableReference):
            if scope is _Scope.CALLER:
                self.result.variables.add(inline.id.name)
        elif isinstance(inline, fl.MessageReference):
            self.result.referenced_messages.add(inline.id.name)
            message = self.entries.message(inline.id.name)
            attribute = inline.attribute.name if inline.attribute is not None else None
            if message is None:
                self.result.unknown_references.append(UnknownReference("message", inline.id.name))
            else:
                self.visit_message(message, attribute, scope)
        elif isinstance(inline, fl.TermReference):
            if inline.arguments is not None:
                self.visit_call_arguments(inline.arguments, scope)
            term = self.entries.term(inline.id.name)
            attribute = inline.attribute.name if inline.attribute is not None else None
            if term is None:
                self.result.unknown_references.append(UnknownReference("term", inline.id.name))
            else:
                self.visit_term(term, attribute, _Scope.TERM)
        elif isinstance(inline, fl.FunctionReference):
            self.visit_call_arguments(inline.arguments, scope)
        elif isinstance(inline, fl.Placeable):
            self.visit_expression(inline.expression, scope)

    def visit_call_arguments(self, arguments: fl.CallArguments, scope: _Scope) -> None:
        for positional in arguments.positional:
            self.visit_inline(positional, scope)
        for named in arguments.named:
            self.visit_inline(named.value, scope)


def _message_pattern(message: fl.Message, attribute: str | None) -> fl.Pattern | None:
    if attribute is None:
        return message.value
    for candidate in message.attributes:
        if candidate.id.name == attribute:
            return candidate.value
    return None


def _term_pattern(term: fl.Term, attribute: str | None) -> fl.Pattern | None:
    if attribute is None:
        return term.value
    for candidate in term.attributes:
        if candidate.id.name == attribute:
            return candidate.value
    return None


def message_variables(entries: FluentEntries, message: fl.Message) -> CollectedVariables:
    """Caller variables of ``message`` (its own attributes are not counted)."""
    collector = _Collector(entries)
    collector.visit_message(message, None, _Scope.CALLER)
    return collector.result


def term_variables(entries: FluentEntries, term: fl.Term) -> CollectedVariables:
    """Parameters of ``term``: the variables its own value uses."""
    collector = _Collector(entries)
    collector.visit_term(term, None, _Scope.CALLER)
    return collector.result
