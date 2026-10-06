"""Per-TestRun values with an explicit safe representation for reporting."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


_REDACTED = "[REDACTED]"


class RunContextValue(BaseModel):
    """One runtime value; normal model serialization preserves its value."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any = Field(repr=False)
    sensitive: bool = False
    source: str | None = None

    def __repr__(self) -> str:
        shown_value = _REDACTED if self.sensitive else repr(self.value)
        return (
            f"RunContextValue(value={shown_value}, sensitive={self.sensitive!r}, "
            f"source={self.source!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def safe_dump(self) -> dict[str, Any]:
        """Return reportable metadata without exposing a sensitive value."""
        safe_value = self.model_copy(update={
            "value": _REDACTED if self.sensitive else self.value,
        })
        return safe_value.model_dump(mode="json")


class RunContext(BaseModel):
    """Mutable state owned by one TestRun, separate from its TestCase."""

    values: dict[str, RunContextValue] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def validate_keys(self) -> "RunContext":
        for key in self.values:
            self._validate_key(key)
        return self

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("RunContext keys must be non-empty trimmed strings.")

    def set_value(
        self,
        key: str,
        value: Any,
        *,
        sensitive: bool = False,
        source: str | None = None,
    ) -> None:
        """Add a value. Duplicate keys are rejected; use replace_value explicitly."""
        self._validate_key(key)
        if key in self.values:
            raise ValueError(f"RunContext key {key!r} already exists.")
        self.values[key] = RunContextValue(
            value=value,
            sensitive=sensitive,
            source=source,
        )

    def replace_value(
        self,
        key: str,
        value: Any,
        *,
        sensitive: bool | None = None,
        source: str | None = None,
    ) -> None:
        """Replace an existing runtime value without downgrading sensitivity."""
        self._validate_key(key)
        current = self.values[key]
        self.values[key] = RunContextValue(
            value=value,
            sensitive=current.sensitive or bool(sensitive),
            source=source if source is not None else current.source,
        )

    def get_value(self, key: str) -> Any:
        """Return the actual runtime value; missing keys raise ``KeyError``."""
        self._validate_key(key)
        return self.values[key].value

    def has_value(self, key: str) -> bool:
        self._validate_key(key)
        return key in self.values

    def safe_dump(self) -> dict[str, dict[str, Any]]:
        """JSON-ready reporting view. Use ``model_dump`` for persistence only."""
        return {key: value.safe_dump() for key, value in self.values.items()}

    def __repr__(self) -> str:
        return f"RunContext(values={self.safe_dump()!r})"

    def __str__(self) -> str:
        return self.__repr__()
