from __future__ import annotations

import inspect
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import sqlite3
from backends import DuckDBSource, SQLiteSource
from entity import EntityConfiguration
from entity_config_set import EntityConfigurationSet
from feature_builder.imports import ImportSpec, resolve_imports
from framecache import CacheConfig, FrameCache
from panel_expression import PanelExpression

from gui.plotnine_wrapper import parse_flexible_datetime

_NUMERIC_DTYPES = {
    pl.Float64,
    pl.Float32,
    pl.Int64,
    pl.Int32,
    pl.Int16,
    pl.Int8,
    pl.UInt64,
    pl.UInt32,
    pl.UInt16,
    pl.UInt8,
}

_DATE_HINTS = ("date", "datetime", "timestamp", "time", "dt", "obs_date", "period")
_ENTITY_HINTS = (
    "entity",
    "source",
    "id",
    "oid",
    "symbol",
    "ticker",
    "country",
    "issuer",
    "subject",
    "site",
    "zone",
    "name",
    "series",
)


@dataclass
class DataSourceRecord:
    """Metadata and configuration for a loaded Panelytics data source."""

    name: str
    source_type: str  # "csv" | "parquet" | "sqlite" | "duckdb" | "clipboard"
    path: str | None = None
    table_name: str | None = None
    clipboard_text: str | None = None
    date_col: str = "date"
    entity_col: str = "entity"
    default_feature: str = ""
    date_col_type: str = "date"  # "date" | "datetime"
    available_tables: list[str] = field(default_factory=list)
    available_columns: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    min_date: date | None = None
    max_date: date | None = None
    dataframe: pl.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize data source record for JSON persistence."""
        return {
            "name": self.name,
            "source_type": self.source_type,
            "path": self.path,
            "table_name": self.table_name,
            "clipboard_text": self.clipboard_text,
            "date_col": self.date_col,
            "entity_col": self.entity_col,
            "default_feature": self.default_feature,
            "date_col_type": self.date_col_type,
        }


def parse_clipboard_dataframe(text: str) -> pl.DataFrame | None:
    """Attempt to parse clipboard text (TSV, CSV, or Markdown table) into a Polars DataFrame."""
    if not text or not text.strip():
        return None

    stripped = text.strip()
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    # Check if markdown table
    if all(ln.strip().startswith("|") and ln.strip().endswith("|") for ln in lines[:2]):
        header_cells = [c.strip() for c in lines[0].strip().strip("|").split("|")]
        data_start = 2 if re.match(r"^[\s|:-]+$", lines[1]) else 1
        rows = [
            [c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in lines[data_start:]
        ]
        rows = [r for r in rows if len(r) == len(header_cells)]
        if header_cells and rows:
            tsv_buf = "\n".join(
                ["\t".join(header_cells)] + ["\t".join(r) for r in rows]
            )
            try:
                df = pl.read_csv(
                    io.StringIO(tsv_buf), separator="\t", try_parse_dates=True
                )
                if df.width >= 2 and df.height >= 1:
                    return df
            except Exception:
                pass

    # Try tab-separated first (Excel copy/paste), then comma, then semicolon, then whitespace
    for sep in ("\t", ",", ";"):
        if sep in lines[0]:
            try:
                df = pl.read_csv(
                    io.StringIO(stripped),
                    separator=sep,
                    try_parse_dates=True,
                    ignore_errors=True,
                )
                if df.width >= 2 and df.height >= 1:
                    return df
            except Exception:
                continue

    # Fallback: whitespace separated
    parts_first = lines[0].split()
    if len(parts_first) >= 2:
        all_rows = [ln.split() for ln in lines]
        if all(len(r) == len(parts_first) for r in all_rows):
            tsv_buf = "\n".join("\t".join(r) for r in all_rows)
            try:
                df = pl.read_csv(
                    io.StringIO(tsv_buf),
                    separator="\t",
                    try_parse_dates=True,
                    ignore_errors=True,
                )
                if df.width >= 2 and df.height >= 1:
                    return df
            except Exception:
                pass

    return None


def list_database_tables(path: str | Path, source_type: str) -> list[str]:
    """Return all user table names in a SQLite or DuckDB database (read-only)."""
    p = Path(path)
    if not p.is_file():
        return []

    if source_type == "sqlite":
        uri = f"file:{p.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            return [row[0] for row in cur.fetchall()]

    if source_type == "duckdb":
        with duckdb.connect(str(p), read_only=True) as conn:
            cur = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' "
                "ORDER BY table_name"
            )
            return [row[0] for row in cur.fetchall()]

    return []


def sample_database_table(
    path: str | Path, source_type: str, table_name: str, limit: int = 500
) -> pl.DataFrame:
    """Fetch a sample DataFrame from a database table in read-only mode."""
    quoted = '"' + table_name.replace('"', '""') + '"'
    sql = f"SELECT * FROM {quoted} LIMIT {int(limit)}"
    if source_type == "sqlite":
        with SQLiteSource(path, read_only=True) as src:
            return src.run_query(sql)
    if source_type == "duckdb":
        with DuckDBSource(path, read_only=True) as src:
            return src.run_query(sql)
    return pl.DataFrame()


def estimate_source_columns(
    df: pl.DataFrame,
) -> tuple[str, str, str, str]:
    """Estimate (date_col, entity_col, default_feature, date_col_type) from a DataFrame."""
    cols = df.columns
    if not cols:
        return ("date", "entity", "value", "date")

    # 1. Estimate date_col and date_col_type
    date_col: str | None = None
    date_col_type = "date"

    for c in cols:
        dtype = df.schema.get(c)
        if dtype == pl.Datetime:
            date_col = c
            date_col_type = "datetime"
            break
        if dtype == pl.Date:
            date_col = c
            date_col_type = "date"
            break

    if date_col is None:
        # Check by column name hint or parseable strings
        for c in cols:
            if c.lower() in _DATE_HINTS or any(h in c.lower() for h in ("date", "time")):
                dtype = df.schema.get(c)
                if dtype in (pl.Utf8, pl.String):
                    sample_vals = [
                        str(v) for v in df[c].head(25).to_list() if v is not None
                    ]
                    parsed = [parse_flexible_datetime(v) for v in sample_vals]
                    valid = [p for p in parsed if p is not None]
                    if valid and len(valid) >= len(sample_vals) * 0.6:
                        date_col = c
                        has_time = any(
                            p.hour != 0 or p.minute != 0 or p.second != 0 for p in valid
                        )
                        date_col_type = "datetime" if has_time else "date"
                        break

    if date_col is None:
        # Try any string column that parses as dates
        for c in cols:
            dtype = df.schema.get(c)
            if dtype in (pl.Utf8, pl.String):
                sample_vals = [
                    str(v) for v in df[c].head(25).to_list() if v is not None
                ]
                parsed = [parse_flexible_datetime(v) for v in sample_vals]
                valid = [p for p in parsed if p is not None]
                if valid and len(valid) >= max(1, len(sample_vals) * 0.7):
                    date_col = c
                    has_time = any(
                        p.hour != 0 or p.minute != 0 or p.second != 0 for p in valid
                    )
                    date_col_type = "datetime" if has_time else "date"
                    break

    if date_col is None:
        date_col = cols[0]

    # 2. Estimate entity_col
    remaining = [c for c in cols if c != date_col]
    entity_col: str | None = None

    for c in remaining:
        if c.lower() in _ENTITY_HINTS:
            entity_col = c
            break

    if entity_col is None:
        # Prefer string/categorical column
        for c in remaining:
            if df.schema.get(c) in (pl.Utf8, pl.String, pl.Categorical):
                entity_col = c
                break

    if entity_col is None:
        # Prefer integer column with low cardinality
        for c in remaining:
            if df.schema.get(c) in (pl.Int64, pl.Int32, pl.Int16, pl.UInt64, pl.UInt32):
                entity_col = c
                break

    if entity_col is None:
        entity_col = remaining[0] if remaining else cols[0]

    # 3. Estimate default_feature
    feature_candidates = [c for c in cols if c not in (date_col, entity_col)]
    default_feature: str | None = None
    for c in feature_candidates:
        if df.schema.get(c) in _NUMERIC_DTYPES:
            default_feature = c
            break

    if default_feature is None:
        default_feature = (
            feature_candidates[0]
            if feature_candidates
            else (remaining[-1] if remaining else cols[-1])
        )

    return (date_col, entity_col, default_feature, date_col_type)


def _normalize_in_memory_dates(
    df: pl.DataFrame, date_col: str, date_col_type: str
) -> pl.DataFrame:
    """Ensure date_col in an in-memory DataFrame can be cast cleanly by EntityConfiguration."""
    if date_col not in df.columns:
        return df
    dtype = df.schema.get(date_col)
    if dtype in (pl.Date, pl.Datetime):
        return df
    if dtype in (pl.Utf8, pl.String):
        parsed_col = (
            pl.col(date_col)
            .map_elements(parse_flexible_datetime, return_dtype=pl.Datetime("us"))
            .alias(date_col)
        )
        df_parsed = df.with_columns(parsed_col)
        if df_parsed[date_col].null_count() < df.height:
            if date_col_type == "date":
                return df_parsed.with_columns(pl.col(date_col).cast(pl.Date))
            return df_parsed
    return df


def populate_source_metadata(
    record: DataSourceRecord, *, estimate_missing: bool = False
) -> DataSourceRecord:
    """Load schema, entities, features, and date bounds for a DataSourceRecord."""
    df_sample: pl.DataFrame | None = None

    if record.source_type == "clipboard":
        if record.dataframe is None and record.clipboard_text:
            record.dataframe = parse_clipboard_dataframe(record.clipboard_text)
        df_sample = record.dataframe
    elif record.source_type == "csv" and record.path:
        if record.dataframe is None:
            record.dataframe = pl.read_csv(record.path, try_parse_dates=True)
        df_sample = record.dataframe
    elif record.source_type == "parquet" and record.path:
        if record.dataframe is None:
            record.dataframe = pl.read_parquet(record.path)
        df_sample = record.dataframe
    elif record.source_type in ("sqlite", "duckdb") and record.path:
        record.available_tables = list_database_tables(record.path, record.source_type)
        if not record.table_name and record.available_tables:
            record.table_name = record.available_tables[0]
        if record.table_name:
            df_sample = sample_database_table(
                record.path, record.source_type, record.table_name
            )

    if df_sample is None or df_sample.is_empty() and not df_sample.columns:
        return record

    record.available_columns = list(df_sample.columns)

    est_date, est_entity, est_feat, est_dtype = estimate_source_columns(df_sample)
    if estimate_missing or record.date_col not in record.available_columns:
        record.date_col = est_date
        record.date_col_type = est_dtype
    else:
        # Check if existing date_col is Datetime vs Date
        col_dtype = df_sample.schema.get(record.date_col)
        if col_dtype == pl.Datetime:
            record.date_col_type = "datetime"
        elif col_dtype == pl.Date:
            record.date_col_type = "date"
        else:
            record.date_col_type = est_dtype

    if estimate_missing or record.entity_col not in record.available_columns:
        record.entity_col = est_entity
    if estimate_missing or record.default_feature not in record.available_columns:
        record.default_feature = est_feat

    record.features = [
        c
        for c in record.available_columns
        if c not in (record.date_col, record.entity_col)
    ]

    if record.dataframe is not None:
        record.dataframe = _normalize_in_memory_dates(
            record.dataframe, record.date_col, record.date_col_type
        )
        df_for_stats = record.dataframe
        if record.entity_col in df_for_stats.columns:
            record.entities = sorted(
                str(x)
                for x in df_for_stats[record.entity_col].drop_nulls().unique().to_list()
            )
        if record.date_col in df_for_stats.columns:
            dmin, dmax = _extract_date_bounds(df_for_stats[record.date_col])
            record.min_date = dmin
            record.max_date = dmax
    elif record.source_type in ("sqlite", "duckdb") and record.path and record.table_name:
        quoted_table = '"' + record.table_name.replace('"', '""') + '"'
        quoted_ent = '"' + record.entity_col.replace('"', '""') + '"'
        quoted_date = '"' + record.date_col.replace('"', '""') + '"'
        backend_cls = (
            SQLiteSource if record.source_type == "sqlite" else DuckDBSource
        )
        with backend_cls(record.path, read_only=True) as src:
            ent_df = src.run_query(
                f"SELECT DISTINCT {quoted_ent} AS ent FROM {quoted_table} "
                f"WHERE {quoted_ent} IS NOT NULL LIMIT 5000"
            )
            if "ent" in ent_df.columns:
                record.entities = sorted(str(x) for x in ent_df["ent"].to_list())
            bounds_df = src.run_query(
                f"SELECT MIN({quoted_date}) AS dmin, MAX({quoted_date}) AS dmax "
                f"FROM {quoted_table}"
            )
            if not bounds_df.is_empty():
                dmin = _coerce_to_date(bounds_df["dmin"][0])
                dmax = _coerce_to_date(bounds_df["dmax"][0])
                record.min_date = dmin
                record.max_date = dmax

    return record


def _coerce_to_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    dt = parse_flexible_datetime(val)
    return dt.date() if dt is not None else None


def _extract_date_bounds(series: pl.Series) -> tuple[date | None, date | None]:
    non_null = series.drop_nulls()
    if non_null.is_empty():
        return (None, None)
    dmin = _coerce_to_date(non_null.min())
    dmax = _coerce_to_date(non_null.max())
    return (dmin, dmax)


def create_source_from_path(
    file_path: str | Path, existing_names: set[str] | None = None
) -> DataSourceRecord:
    """Create and estimate a DataSourceRecord from a dropped/selected file path."""
    p = Path(file_path).resolve()
    suffix = p.suffix.lower()
    if suffix == ".csv":
        stype = "csv"
    elif suffix in (".parquet", ".pq"):
        stype = "parquet"
    elif suffix in (".db", ".sqlite", ".sqlite3"):
        stype = "sqlite"
    elif suffix == ".duckdb":
        stype = "duckdb"
    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r}. "
            "Supported: .csv, .parquet, .pq, .db, .sqlite, .duckdb"
        )

    base_name = p.stem
    name = _unique_source_name(base_name, existing_names or set())
    record = DataSourceRecord(
        name=name,
        source_type=stype,
        path=str(p),
    )
    return populate_source_metadata(record, estimate_missing=True)


def create_source_from_clipboard(
    clipboard_text: str, existing_names: set[str] | None = None
) -> DataSourceRecord:
    """Create and estimate a DataSourceRecord from clipboard text."""
    df = parse_clipboard_dataframe(clipboard_text)
    if df is None or df.is_empty():
        raise ValueError("Clipboard does not contain a valid tabular dataset.")

    name = _unique_source_name("clipboard_data", existing_names or set())
    record = DataSourceRecord(
        name=name,
        source_type="clipboard",
        clipboard_text=clipboard_text,
        dataframe=df,
    )
    return populate_source_metadata(record, estimate_missing=True)


def _unique_source_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    idx = 2
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def build_entity_configuration_from_record(
    record: DataSourceRecord,
    *,
    entity_col_override: str | None = None,
) -> EntityConfiguration:
    """Construct a panelyzer EntityConfiguration from a DataSourceRecord."""
    populate_source_metadata(record, estimate_missing=False)
    actual_entity_col = record.entity_col
    target_entity_col = entity_col_override or actual_entity_col

    if record.dataframe is not None:
        df = record.dataframe
        if target_entity_col != actual_entity_col:
            df = df.with_columns(pl.col(actual_entity_col).alias(target_entity_col))
        return EntityConfiguration(
            entity_col=target_entity_col,
            dataframe=df,
            default_feature=record.default_feature,
            date_col=record.date_col,
            date_col_type=record.date_col_type,  # type: ignore[arg-type]
        )

    if record.source_type == "sqlite" and record.path and record.table_name:
        return EntityConfiguration.from_sqlite(
            record.path,
            table_name=record.table_name,
            entity_col=actual_entity_col,
            default_feature=record.default_feature,
            date_col=record.date_col,
            date_col_type=record.date_col_type,
        )

    if record.source_type == "duckdb" and record.path and record.table_name:
        return EntityConfiguration.from_duckdb(
            record.path,
            table_name=record.table_name,
            entity_col=actual_entity_col,
            default_feature=record.default_feature,
            date_col=record.date_col,
            date_col_type=record.date_col_type,
        )

    raise ValueError(f"Cannot build EntityConfiguration for source {record.name!r}")


def parse_import_statement(stmt: str) -> ImportSpec:
    """Parse a Python import statement or module name into a panelyzer ImportSpec.

    Supported forms:
    - ``import pkg``
    - ``import pkg.sub as alias``
    - ``from pkg.sub import fn1, fn2``
    - ``pkg.sub as alias``
    - ``pkg``
    """
    text = stmt.strip()
    if not text:
        raise ValueError("Empty import statement")

    from_match = re.match(
        r"^from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+(.+)$", text
    )
    if from_match:
        pkg = from_match.group(1).strip()
        raw_names = [n.strip() for n in from_match.group(2).split(",") if n.strip()]
        from_names = []
        for rn in raw_names:
            if " as " in rn:
                orig, _, _ = rn.partition(" as ")
                from_names.append(orig.strip())
            else:
                from_names.append(rn)
        return ImportSpec(package=pkg, from_names=from_names)

    imp_match = re.match(
        r"^import\s+([A-Za-z_][A-Za-z0-9_.]*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?$",
        text,
    )
    if imp_match:
        pkg = imp_match.group(1).strip()
        alias = imp_match.group(2).strip() if imp_match.group(2) else None
        return ImportSpec(package=pkg, as_name=alias)

    if " as " in text:
        pkg, _, alias = text.partition(" as ")
        return ImportSpec(package=pkg.strip(), as_name=alias.strip())

    return ImportSpec(package=text)


def load_user_eval_globals(
    pythonpath: list[str], import_statements: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Resolve user pythonpath and import statements into an eval_globals dict and error list."""
    specs: list[ImportSpec] = []
    errors: list[str] = []

    valid_paths = [p for p in pythonpath if Path(p).is_dir()]
    for p in pythonpath:
        if not Path(p).is_dir():
            errors.append(f"Directory not found: {p}")

    for stmt in import_statements:
        try:
            spec = parse_import_statement(stmt)
            specs.append(spec)
        except Exception as exc:
            errors.append(f"Invalid import {stmt!r}: {exc}")

    namespace: dict[str, Any] = {}
    if valid_paths:
        try:
            resolve_imports([], pythonpath=valid_paths)
        except Exception as exc:
            errors.append(str(exc))

    for spec in specs:
        try:
            part = resolve_imports([spec], pythonpath=valid_paths)
            namespace.update(part)
        except Exception as exc:
            errors.append(str(exc))

    return namespace, errors


def build_autocomplete_tokens(
    sources: list[DataSourceRecord],
    extra_eval_globals: dict[str, Any] | None = None,
) -> list[str]:
    """Generate autocomplete suggestions for `{entity}`, `{entity:feature}`, built-ins, and user functions."""
    tokens: set[str] = set()

    # 1. Entity & Feature references from all configured data sources
    for src in sources:
        for ent in src.entities:
            tokens.add(f"{{{ent}}}")
            for feat in src.features:
                tokens.add(f"{{{ent}:{feat}}}")

    # 2. Panelyzer built-in rolling & math helpers
    builtin_helpers = [
        "romean(",
        "rostd(",
        "rosum(",
        "romin(",
        "romax(",
        "romedian(",
        "rocov(",
        "rocorr(",
        "roquantile(",
        "roskew(",
        "rokurtosis(",
        "rozscore(",
        "robeta(",
        "np.log(",
        "np.exp(",
        "np.sqrt(",
        "np.abs(",
        "np.mean(",
        "np.std(",
        "np.median(",
        "np.min(",
        "np.max(",
        "np.clip(",
        "np.where(",
        "np.diff(",
        "sp.stats.zscore(",
    ]
    tokens.update(builtin_helpers)

    # 3. User-imported modules and functions
    if extra_eval_globals:
        for name, obj in extra_eval_globals.items():
            if inspect.ismodule(obj):
                tokens.add(name)
                for attr_name in dir(obj):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr = getattr(obj, attr_name)
                        if callable(attr):
                            tokens.add(f"{name}.{attr_name}(")
                    except Exception:
                        continue
            elif callable(obj):
                tokens.add(f"{name}(")
            else:
                tokens.add(name)

    return sorted(tokens)


_BRACKET_REF_RE = re.compile(r"\{([^{}]+)\}")


def formula_uses_source(formula: str, source: DataSourceRecord) -> bool:
    """Return True if any `{entity}` or `{entity:feature}` token in `formula` belongs to `source`."""
    if not formula or not source.entities:
        return False
    entities = set(source.entities)
    features = set(source.features) | ({source.default_feature} if source.default_feature else set())
    for match in _BRACKET_REF_RE.finditer(formula):
        content = match.group(1).strip()
        if not content:
            continue
        ent, _, feat = content.partition(":")
        ent = ent.strip()
        feat = feat.strip()
        if ent in entities and (not feat or not features or feat in features):
            return True
    return False


def evaluate_formula_rows(
    sources: list[DataSourceRecord],
    formula_rows: list[dict[str, Any]],
    extra_eval_globals: dict[str, Any] | None = None,
    row_errors_out: dict[int, str] | None = None,
) -> tuple[pl.DataFrame | None, str | None, list[str]]:
    """Evaluate active panelyzer formula rows across configured data sources.

    Returns
    -------
    tuple[pl.DataFrame | None, str | None, list[str]]
        (combined_dataframe, regime_column_name, list_of_warnings_or_errors).
        If ``row_errors_out`` is provided, populates ``{row_index: error_msg}``
        for any row whose formula fails evaluation or returns no rows.
    """
    IndexedRow = tuple[int, dict[str, Any]]
    active_rows: list[IndexedRow] = [
        (i, r)
        for i, r in enumerate(formula_rows)
        if str(r.get("formula", "")).strip()
    ]
    if not active_rows:
        return None, None, ["Enter at least one panelyzer formula to visualize."]
    if not sources:
        return None, None, ["Add at least one Data Source before evaluating formulas."]

    warnings_list: list[str] = []
    configs: list[EntityConfiguration] = []
    used_entity_cols: set[str] = set()
    min_dates: list[date] = []
    max_dates: list[date] = []
    any_datetime = False

    for idx, src in enumerate(sources):
        try:
            ent_col = src.entity_col
            override: str | None = None
            if ent_col in used_entity_cols:
                override = f"{ent_col}_{idx + 1}"
                used_entity_cols.add(override)
            else:
                used_entity_cols.add(ent_col)

            cfg = build_entity_configuration_from_record(
                src, entity_col_override=override
            )
            configs.append(cfg)
            if src.min_date is not None:
                min_dates.append(src.min_date)
            if src.max_date is not None:
                max_dates.append(src.max_date)
            if src.date_col_type == "datetime":
                any_datetime = True
        except Exception as exc:
            warnings_list.append(f"Source {src.name!r}: {exc}")

    if not configs:
        return None, None, warnings_list or ["No valid data sources available."]

    global_min_date = min(min_dates) if min_dates else date(1970, 1, 1)
    today = datetime.now().date()
    global_max_date = max(max_dates) if max_dates else today
    # PanelExpression._validate_from_and_to_date checks to_date <= min(max_date, today)
    panel_max_date = max(global_max_date, today)

    if any_datetime:
        from_bound: date | datetime = datetime.combine(
            global_min_date, datetime.min.time()
        )
        to_bound: date | datetime = datetime.combine(
            min(global_max_date, datetime.now().date()), datetime.max.time()
        )
    else:
        from_bound = global_min_date
        to_bound = min(global_max_date, today)

    config_set = EntityConfigurationSet(entity_configs=configs)
    fc = FrameCache.from_config(CacheConfig(backend_type="sqlite", db_path=":memory:"))

    date_col_name = "date"
    panel = PanelExpression(
        entity_config_set=config_set,
        frame_cache=fc,
        date_column_name=date_col_name,
        user_supplied_tag="panelytics_gui",
        min_date=global_min_date,
        max_date=panel_max_date,
        extra_eval_globals=extra_eval_globals or {},
        caching_enabled=False,
    )

    combined_df: pl.DataFrame | None = None
    regime_col: str | None = None
    used_col_names: set[str] = {date_col_name}

    try:
        for row_idx, row in active_rows:
            expr = str(row["formula"]).strip()
            is_regime = bool(row.get("is_regime", False))
            try:
                res_df = panel.evaluate_expression(expr, from_bound, to_bound)
            except Exception as exc:
                err_msg = f"Row {row_idx + 1} ({expr}): {exc}"
                warnings_list.append(err_msg)
                if row_errors_out is not None:
                    row_errors_out[row_idx] = str(exc)
                continue

            if res_df is None or res_df.is_empty():
                err_msg = f"Row {row_idx + 1} ({expr}): returned no rows."
                warnings_list.append(err_msg)
                if row_errors_out is not None:
                    row_errors_out[row_idx] = "Formula returned no rows."
                continue

            data_cols = [c for c in res_df.columns if c != date_col_name]
            if not data_cols:
                continue

            # Ensure unique column names across rows
            rename_map: dict[str, str] = {}
            final_cols: list[str] = []
            for c in data_cols:
                candidate = c
                suffix_idx = 2
                while candidate in used_col_names:
                    candidate = f"{c} ({suffix_idx})"
                    suffix_idx += 1
                used_col_names.add(candidate)
                if candidate != c:
                    rename_map[c] = candidate
                final_cols.append(candidate)

            if rename_map:
                res_df = res_df.rename(rename_map)

            if is_regime and final_cols:
                regime_col = final_cols[0]

            if combined_df is None:
                combined_df = res_df
            else:
                combined_df = combined_df.join(
                    res_df, on=date_col_name, how="full", coalesce=True
                )
    finally:
        for cfg in configs:
            try:
                cfg.close()
            except Exception:
                pass

    if combined_df is not None and date_col_name in combined_df.columns:
        combined_df = combined_df.sort(date_col_name)

    return combined_df, regime_col, warnings_list
