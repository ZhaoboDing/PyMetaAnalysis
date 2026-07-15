"""Immutable records describing how an analysis was constructed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._version import __version__
from .config import MethodOptionValue
from .data import ColumnOrArray

PROVENANCE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class InputFieldProvenance:
    """Resolved source of one public analysis input."""

    role: str
    source: str
    column: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-compatible representation."""

        return {
            "role": self.role,
            "source": self.source,
            "column": self.column,
        }


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    """A configured data transformation and the rows it affected."""

    name: str
    parameters: tuple[tuple[str, MethodOptionValue], ...] = ()
    affected_rows: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "affected_rows": list(self.affected_rows),
        }


@dataclass(frozen=True, slots=True)
class AnalysisProvenance:
    """Versioned input and transformation provenance for one fitted result."""

    package_version: str
    schema_version: str
    analysis_type: str
    data_source: str
    input_fields: tuple[InputFieldProvenance, ...]
    row_count: int
    included_rows: tuple[int, ...]
    excluded_rows: tuple[int, ...]
    transformations: tuple[TransformationRecord, ...] = ()

    @property
    def column_mapping(self) -> Mapping[str, str]:
        """Return only inputs resolved from DataFrame columns."""

        return MappingProxyType(
            {
                field.role: field.column
                for field in self.input_fields
                if field.source == "column" and field.column is not None
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached, JSON-compatible provenance document."""

        return {
            "package_version": self.package_version,
            "schema_version": self.schema_version,
            "analysis_type": self.analysis_type,
            "data_source": self.data_source,
            "input_fields": [field.to_dict() for field in self.input_fields],
            "column_mapping": dict(self.column_mapping),
            "row_count": self.row_count,
            "included_rows": list(self.included_rows),
            "excluded_rows": list(self.excluded_rows),
            "transformations": [
                transformation.to_dict() for transformation in self.transformations
            ],
        }


def _input_field(
    role: str,
    value: ColumnOrArray | None,
    *,
    data: pd.DataFrame | None,
    is_study: bool = False,
) -> InputFieldProvenance:
    if isinstance(value, str):
        return InputFieldProvenance(role=role, source="column", column=value)
    if value is not None:
        return InputFieldProvenance(role=role, source="array")
    if is_study and data is not None:
        return InputFieldProvenance(role=role, source="dataframe_index")
    if is_study:
        return InputFieldProvenance(role=role, source="generated_row_number")
    raise RuntimeError(f"Input provenance for {role!r} requires a value.")


def build_analysis_provenance(
    *,
    analysis_type: str,
    data: pd.DataFrame | None,
    inputs: Sequence[tuple[str, ColumnOrArray]],
    study: ColumnOrArray | None,
    included: NDArray[np.bool_],
    transformations: tuple[TransformationRecord, ...] = (),
) -> AnalysisProvenance:
    """Build a provenance record after input validation and transformations."""

    fields = tuple(_input_field(role, value, data=data) for role, value in inputs) + (
        _input_field("study", study, data=data, is_study=True),
    )
    included_rows = tuple(int(row) for row in np.flatnonzero(included))
    excluded_rows = tuple(int(row) for row in np.flatnonzero(~included))
    return AnalysisProvenance(
        package_version=__version__,
        schema_version=PROVENANCE_SCHEMA_VERSION,
        analysis_type=analysis_type,
        data_source="pandas_dataframe" if data is not None else "array_like",
        input_fields=fields,
        row_count=len(included),
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        transformations=transformations,
    )


def add_input_field(
    provenance: AnalysisProvenance,
    *,
    role: str,
    value: ColumnOrArray,
    data: pd.DataFrame | None,
) -> AnalysisProvenance:
    """Add a resolved input used by a composite analysis such as subgroups."""

    field = _input_field(role, value, data=data)
    return replace(provenance, input_fields=(*provenance.input_fields, field))


def remap_provenance_rows(
    provenance: AnalysisProvenance,
    row_ids: Sequence[int],
    *,
    data_source: str = "derived_subset",
) -> AnalysisProvenance:
    """Map local row positions in a derived fit back to parent row identifiers."""

    mapped = tuple(int(row) for row in row_ids)

    def remap(rows: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(mapped[row] for row in rows)

    transformations = tuple(
        replace(record, affected_rows=remap(record.affected_rows))
        for record in provenance.transformations
    )
    return replace(
        provenance,
        data_source=data_source,
        row_count=len(mapped),
        included_rows=remap(provenance.included_rows),
        excluded_rows=remap(provenance.excluded_rows),
        transformations=transformations,
    )
