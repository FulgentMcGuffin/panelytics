from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime, time
from typing import Any

import polars as pl
from plotnine import (
    aes,
    coord_flip,
    facet_grid,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_histogram,
    geom_jitter,
    geom_line,
    geom_point,
    geom_rug,
    geom_smooth,
    geom_tile,
    geom_violin,
    ggplot,
    ggtitle,
    scale_y_continuous,
    scale_y_log10,
    scale_y_reverse,
    scale_y_sqrt,
    scale_y_symlog,
    theme,
    theme_538,
    theme_bw,
    theme_classic,
    theme_dark,
    theme_gray,
    theme_light,
    theme_linedraw,
    theme_matplotlib,
    theme_minimal,
    theme_seaborn,
    theme_tufte,
    theme_xkcd,
    ylab,
)

# theme_xkcd requests "Humor Sans"/"xkcd" fonts; when absent matplotlib logs
# one findfont warning per text element before falling back to a default font.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PLOT9_THEMES_DICT = {
    "538": theme_538,
    "5": theme_538,
    "f": theme_538,
    "five": theme_538,
    "dark": theme_dark,
    "d": theme_dark,
    "light": theme_light,
    "1": theme_light,
    "matplotlib": theme_matplotlib,
    "m": theme_matplotlib,
    "minimal": theme_minimal,
    "min": theme_minimal,
    "classic": theme_classic,
    "c": theme_classic,
    "bw": theme_bw,
    "b": theme_bw,
    "gray": theme_gray,
    "g": theme_gray,
    "seaborn": theme_seaborn,
    "s": theme_seaborn,
    "tufte": theme_tufte,
    "t": theme_tufte,
    "xkcd": theme_xkcd,
    "x": theme_xkcd,
    "linedraw": theme_linedraw,
    "ld": theme_linedraw,
}

PLOT9_GEOMS_DICT = {
    "L": geom_line,
    "P": geom_point,
    "B": geom_bar,
    "S": geom_smooth,
    "R": geom_rug,
    "J": geom_jitter,
    "T": geom_tile,
    "V": geom_violin,
    "H": geom_histogram,
    "D": geom_density,
    "X": geom_boxplot,
    "F": facet_grid,
    "W": facet_wrap,
}

PLOT9_YSCALE_DICT = {
    "linear": scale_y_continuous,
    "reverse": scale_y_reverse,
    "log": scale_y_log10,
    "log10": scale_y_log10,
    "log2": scale_y_log10,
    "sqrt": scale_y_sqrt,
    "symlog": scale_y_symlog,
}

PLOT9_POS_LEGEND_DICT = {
    "inside": "inside",
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",
    "none": "none",
    "i": "inside",
    "l": "left",
    "r": "right",
    "t": "top",
    "b": "bottom",
    "n": "none",
}

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

_INTEGER_DTYPES = {
    pl.Int64,
    pl.Int32,
    pl.Int16,
    pl.Int8,
    pl.UInt64,
    pl.UInt32,
    pl.UInt16,
    pl.UInt8,
}

_YEAR_MONTH_RE = re.compile(r"^\s*(\d{4})-(\d{1,2})\s*$")
_TIME_ONLY_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")

_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_matplotlib_date_num(x: float) -> bool:
    """Return True if x looks like a matplotlib day number."""
    return 365 <= float(x) <= 900_000


def _matplotlib_num_to_date_str(x: float) -> str | None:
    """Convert a matplotlib day number to YYYY-mm-dd, or None if not a date number."""
    if not _is_matplotlib_date_num(x):
        return None
    try:
        import matplotlib.dates as mdates

        return mdates.num2date(float(x)).strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_flexible_datetime(value: Any) -> datetime | None:
    """Parse assorted date/time strings into a Python datetime."""
    if value is None or _is_nan(value):
        return None

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            pass

    text = str(value).strip()
    if not text:
        return None

    year_month = _YEAR_MONTH_RE.match(text)
    if year_month:
        year, month = int(year_month.group(1)), int(year_month.group(2))
        if 1 <= month <= 12:
            return datetime(year, month, 15)

    time_only = _TIME_ONLY_RE.match(text)
    if time_only:
        hour = int(time_only.group(1))
        minute = int(time_only.group(2))
        second = int(time_only.group(3) or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
            return datetime.combine(date.today(), time(hour, minute, second))

    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def normalize_x_value(value: Any) -> Any:
    """Convert assorted date-like values to a Python datetime/date when possible."""
    if value is None or _is_nan(value):
        return value

    if isinstance(value, (datetime, date)):
        return value

    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            pass

    if isinstance(value, str):
        parsed = parse_flexible_datetime(value)
        if parsed is not None:
            return parsed

    return value


def format_display_date(value: Any) -> str:
    """Format a date/datetime-like value for table labels and plot tooltips."""
    if value is None or _is_nan(value):
        return ""

    normalized = normalize_x_value(value)
    if isinstance(normalized, datetime):
        if normalized.hour or normalized.minute or normalized.second:
            return normalized.strftime("%Y-%m-%d %H:%M")
        return normalized.strftime("%Y-%m-%d")
    if isinstance(normalized, date):
        return normalized.strftime("%Y-%m-%d")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        as_date = _matplotlib_num_to_date_str(float(value))
        if as_date is not None:
            return as_date

    if isinstance(value, str):
        parsed = parse_flexible_datetime(value)
        if parsed is not None:
            return format_display_date(parsed)
        return value

    return str(value)


def format_table_cell(value: Any) -> str:
    """Format a dataframe cell for display in a table."""
    if value is None or _is_nan(value):
        return ""

    if isinstance(value, (datetime, date)):
        return format_display_date(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int) and 1 <= value <= 366:
            return str(value)
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    if isinstance(value, str):
        parsed = parse_flexible_datetime(value)
        if parsed is not None:
            return format_display_date(parsed)
        return value

    return str(value)


def first_column_date_parse_rate(df: pl.DataFrame, col_name: str | None = None) -> float:
    """Fraction of non-empty column values that parse as dates."""
    if df.is_empty():
        return 0.0

    col = col_name or df.columns[0]
    dtype = df.schema.get(col)
    if dtype in (pl.Date, pl.Datetime):
        return 1.0
    if dtype not in (pl.Utf8, pl.String):
        return 0.0

    values = df[col].to_list()
    non_empty = [v for v in values if v is not None and str(v).strip()]
    if not non_empty:
        return 0.0

    parsed = sum(1 for v in non_empty if parse_flexible_datetime(v) is not None)
    return parsed / len(non_empty)


def coerce_date_columns(df: pl.DataFrame, *, threshold: float = 0.8) -> pl.DataFrame:
    """Parse the first column as dates when most values are date-like strings."""
    if df.is_empty():
        return df

    first_col = df.columns[0]
    dtype = df.schema.get(first_col)
    if dtype in (pl.Date, pl.Datetime):
        return df
    if dtype not in (pl.Utf8, pl.String):
        return df
    if first_column_date_parse_rate(df, first_col) < threshold:
        return df

    return df.with_columns(
        pl.col(first_col)
        .map_elements(parse_flexible_datetime, return_dtype=pl.Datetime("us"))
        .alias(first_col)
    )


def build_x_numeric_lookup(x_values: list) -> dict[float, str]:
    """Map matplotlib x coordinates (or categorical indices) to display labels."""
    import matplotlib.dates as mdates

    lookup: dict[float, str] = {}
    for idx, value in enumerate(x_values):
        normalized = normalize_x_value(value)
        label = format_display_date(normalized if normalized is not value else value)
        if not label and value is not None:
            label = str(value).strip()
        if not label:
            continue

        keys: list[float] = [float(idx + 1)]
        if float(idx) not in lookup:
            keys.append(float(idx))

        if isinstance(normalized, (datetime, date)):
            keys.append(float(mdates.date2num(normalized)))

        for key in keys:
            lookup[key] = label
            lookup[round(key, 6)] = label

    return lookup


def lookup_x_display(
    lookup: dict[float, str], x_num: float, tolerance: float = 0.51
) -> str | None:
    """Find a display label for a numeric x coordinate."""
    if x_num in lookup:
        return lookup[x_num]

    rounded = round(x_num, 6)
    if rounded in lookup:
        return lookup[rounded]

    best_label = None
    best_dist = tolerance
    for key, label in lookup.items():
        dist = abs(key - x_num)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _coerce_datetime_x_polars(plot_df: pl.DataFrame, x: str) -> pl.DataFrame:
    """Parse date-like string x columns so geom_line can connect points."""
    if x not in plot_df.columns:
        return plot_df

    dtype = plot_df.schema.get(x)
    if dtype in (pl.Date, pl.Datetime):
        return plot_df
    if dtype not in (pl.Utf8, pl.String):
        return plot_df

    if first_column_date_parse_rate(plot_df, x) < 0.9:
        return plot_df

    return plot_df.with_columns(
        pl.col(x)
        .map_elements(parse_flexible_datetime, return_dtype=pl.Datetime("us"))
        .alias(x)
    )


def _ensure_line_grouping(
    aes_dict_kwargs: dict,
    geoms: str,
    has_multiple_series: bool,
    is_histogram: bool,
    series_name: str,
) -> None:
    """Ensure geom_line has a stable group aesthetic to connect observations."""
    if is_histogram or "L" not in geoms:
        return

    if has_multiple_series:
        aes_dict_kwargs.setdefault("group", series_name)
    else:
        aes_dict_kwargs.setdefault("group", 1)


def _resolve_facet_var(
    plot_df: pl.DataFrame,
    x: str,
    has_multiple_series: bool,
    is_histogram: bool,
    series_name: str = "Series",
) -> str | None:
    """Pick a discrete column to facet on, if any."""
    if is_histogram:
        return None

    if has_multiple_series and series_name in plot_df.columns:
        if plot_df[series_name].n_unique() > 1:
            return series_name

    skip = {x, series_name, "Value", "_color_key"}
    for col in plot_df.columns:
        if col in skip:
            continue
        nunique = plot_df[col].n_unique()
        if nunique < 2 or nunique > 12:
            continue
        dtype = plot_df.schema.get(col)
        if dtype in _NUMERIC_DTYPES:
            if dtype in _INTEGER_DTYPES:
                return col
            continue
        return col

    return None


def create_ggplot_from_df(
    df: pl.DataFrame,
    x: str,
    y_cols: list[str],
    formula: str,
    title: str | None,
    has_multiple_series: bool,
    geoms: str,
    color: bool,
    alpha: float,
    size: bool,
    shape: bool,
    group: bool,
    theme_str: str,
    flip_coord: bool,
    y_scale: str,
    pos_legend: str,
    main_series_col: str | None = None,
):
    """Generate a ggplot from a Polars DataFrame with flexible styling options."""
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    if isinstance(alpha, type):
        alpha = 1.0
    if isinstance(color, type):
        color = False
    if isinstance(size, type):
        size = False
    if isinstance(shape, type):
        shape = False
    if isinstance(group, type):
        group = False
    if isinstance(flip_coord, type):
        flip_coord = False
    if isinstance(y_scale, type):
        y_scale = "linear"
    if isinstance(pos_legend, type):
        pos_legend = "bottom"
    if isinstance(theme_str, type):
        theme_str = "538"
    if isinstance(geoms, type):
        geoms = "LP"

    geoms = geoms.replace(" ", "").replace("_", "").strip().upper()
    is_histogram = ("D" in geoms) or ("H" in geoms)

    SERIES_NAME = "Series"
    VALUE_NAME = "Value"
    COLOR_KEY = "_color_key"

    plot_df = df.clone()
    if not is_histogram:
        if not has_multiple_series:
            aes_dict_kwargs: dict[str, Any] = {"x": x, "y": y_cols[0]}
            if color is True:
                plot_df = plot_df.with_columns(pl.lit(y_cols[0]).alias(COLOR_KEY))
                aes_dict_kwargs["color"] = COLOR_KEY
                aes_dict_kwargs["fill"] = COLOR_KEY
                aes_dict_kwargs["group"] = COLOR_KEY
            if size is True:
                aes_dict_kwargs["size"] = y_cols[0]
        else:
            aes_dict_kwargs = {"x": x, "y": VALUE_NAME}
            if color is True:
                aes_dict_kwargs["color"] = SERIES_NAME
                aes_dict_kwargs["fill"] = SERIES_NAME
            if size is True:
                aes_dict_kwargs["size"] = VALUE_NAME
            if shape is True:
                aes_dict_kwargs["shape"] = SERIES_NAME
            if group is True:
                aes_dict_kwargs["group"] = SERIES_NAME
    else:
        geoms = "".join(char for char in geoms if char in "DHRXFWB")
        aes_dict_kwargs = (
            {"x": y_cols[0]} if not has_multiple_series else {"x": VALUE_NAME}
        )
        if color is True:
            aes_dict_kwargs["color"] = (
                y_cols[0] if not has_multiple_series else SERIES_NAME
            )
            aes_dict_kwargs["fill"] = aes_dict_kwargs["color"]

    if has_multiple_series and not is_histogram:
        if SERIES_NAME not in plot_df.columns:
            plot_df = df.unpivot(
                index=[x],
                on=y_cols,
                variable_name=SERIES_NAME,
                value_name=VALUE_NAME,
            )

    plot_geoms = "".join(ch for ch in geoms if ch not in "FW")
    if not plot_geoms:
        plot_geoms = "LP"

    _ensure_line_grouping(
        aes_dict_kwargs, plot_geoms, has_multiple_series, is_histogram, SERIES_NAME
    )

    if "L" in plot_geoms and not is_histogram:
        plot_df = _coerce_datetime_x_polars(plot_df, x)
        if x in plot_df.columns:
            plot_df = plot_df.sort(x)

    p = ggplot(plot_df, aes(**aes_dict_kwargs))

    alpha_val = 1.0 if alpha is None else float(alpha)
    add_facet_wrap = "W" in geoms
    add_facet_grid = "F" in geoms

    for char in plot_geoms:
        geom_fn = PLOT9_GEOMS_DICT.get(char)
        if geom_fn is None:
            continue
        if char in ("H", "D"):
            p = p + geom_fn(alpha=alpha_val)
        elif char == "B":
            bar_geom = geom_bar if is_histogram else geom_col
            p = p + bar_geom(alpha=alpha_val)
        else:
            p = p + geom_fn(alpha=alpha_val)

    theme_key = (theme_str or "538").lower()
    theme_fn = PLOT9_THEMES_DICT.get(theme_key, theme_538)
    p = p + theme_fn()

    y_scale_key = (y_scale or "linear").lower()
    if y_scale_key == "log10":
        y_scale_key = "log"
    scale_fn = PLOT9_YSCALE_DICT.get(y_scale_key, scale_y_continuous)
    p = p + scale_fn()

    pos_key = (pos_legend or "bottom").lower()
    legend_pos = PLOT9_POS_LEGEND_DICT.get(pos_key, "bottom")
    if legend_pos != "none":
        p = p + theme(legend_position=legend_pos)

    if flip_coord:
        p = p + coord_flip()

    facet_var = _resolve_facet_var(
        plot_df, x, has_multiple_series, is_histogram, SERIES_NAME
    )
    if facet_var:
        if add_facet_wrap:
            p = p + facet_wrap(facet_var)
        elif add_facet_grid:
            p = p + facet_grid(cols=facet_var)

    if title:
        p = p + ggtitle(title)

    if main_series_col and not has_multiple_series and main_series_col in y_cols:
        p = p + ylab(main_series_col)

    return p


def process_df_for_ggplot(
    df: pl.DataFrame,
    x_col: str,
) -> tuple[pl.DataFrame, list[str], bool]:
    """Prepare a Polars DataFrame for plotnine rendering."""
    if x_col not in df.columns and len(df.columns) > 0:
        x_col = df.columns[0]

    numeric_cols = [
        c for c in df.columns if c != x_col and df.schema.get(c) in _NUMERIC_DTYPES
    ]
    y_cols = numeric_cols or [c for c in df.columns if c != x_col][:1]
    has_multiple_series = len(y_cols) > 1

    plot_df = df.clone()
    if has_multiple_series:
        plot_df = plot_df.unpivot(
            index=[x_col],
            on=y_cols,
            variable_name="Series",
            value_name="Value",
        )

    return plot_df, y_cols, has_multiple_series


def infer_plot_columns(df: pl.DataFrame) -> tuple[str, list[str]]:
    """Pick x/y columns from a Polars DataFrame."""
    if df.is_empty():
        return "index", []

    cols = df.columns
    x_col = cols[0]

    y_cols = [c for c in cols if c != x_col and df.schema.get(c) in _NUMERIC_DTYPES]
    if not y_cols:
        y_cols = [c for c in cols if c != x_col][:3]
    return x_col, y_cols
