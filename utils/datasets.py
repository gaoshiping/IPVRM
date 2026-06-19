from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Iterable, Union

from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None


LocalDataset = Union[Dataset, DatasetDict]
DatasetObject = Union[Dataset, DatasetDict, list[dict]]
_TABULAR_SUFFIXES = (".json", ".jsonl", ".parquet", ".csv", ".tsv")
_LOADABLE_SUFFIXES = _TABULAR_SUFFIXES + (".arrow",)
_LOCAL_CACHE_DIR = Path("/tmp/hf_datasets_cache")


def _normalize_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _match_suffix(path: Path, suffixes: tuple[str, ...]) -> str | None:
    name = path.name.lower()
    for suffix in suffixes:
        if name.endswith(suffix):
            return suffix
    return None


def _select_dataset_split(dataset: LocalDataset, dataset_split: str | None) -> LocalDataset | Dataset:
    if dataset_split is None or not isinstance(dataset, DatasetDict):
        return dataset
    return dataset[dataset_split]


def _default_split_name(dataset_dict: DatasetDict) -> str:
    if "train" in dataset_dict:
        return "train"
    return next(iter(dataset_dict.keys()))


def _load_tabular_files(paths: list[Path], suffix: str, dataset_split: str | None = None) -> Dataset:
    _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path_strings = [str(path) for path in paths]
    cache_dir = str(_LOCAL_CACHE_DIR)
    if suffix == ".parquet":
        if pq is not None:
            records: list[dict] = []
            for path in paths:
                table = pq.read_table(str(path), use_threads=False)
                if table.num_rows == 0:
                    continue
                records.extend(table.to_pylist())
            return Dataset.from_list(records)
        return Dataset.from_parquet(path_strings, cache_dir=cache_dir)
    if suffix in {".json", ".jsonl"}:
        return Dataset.from_json(path_strings, cache_dir=cache_dir)
    if suffix == ".csv":
        return Dataset.from_csv(path_strings, cache_dir=cache_dir)
    if suffix == ".tsv":
        return Dataset.from_csv(path_strings, cache_dir=cache_dir, delimiter="\t")
    raise RuntimeError(f"Unsupported local dataset format: {suffix}")


def _load_local_directory(dataset_dir: Path, dataset_split: str | None = None) -> LocalDataset | Dataset:
    try:
        dataset = load_from_disk(str(dataset_dir))
    except Exception as disk_error:
        files = sorted(
            child
            for child in dataset_dir.iterdir()
            if child.is_file() and _match_suffix(child, _TABULAR_SUFFIXES) is not None
        )
        if not files:
            raise RuntimeError(f"Unsupported local dataset directory: {dataset_dir}") from disk_error

        suffixes = {_match_suffix(path, _TABULAR_SUFFIXES) for path in files}
        if len(suffixes) != 1:
            raise RuntimeError(f"Mixed dataset file formats are not supported under {dataset_dir}")

        return _load_tabular_files(files, suffixes.pop(), dataset_split=dataset_split)

    return _select_dataset_split(dataset, dataset_split)


def _as_single_dataset(obj: DatasetObject, save_path: Path) -> Dataset:
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        if len(obj) != 1:
            raise TypeError(
                f"Saving a DatasetDict to a single file is ambiguous for {save_path}. "
                "Use a directory path or select a split first."
            )
        return next(iter(obj.values()))
    if isinstance(obj, list):
        return Dataset.from_list(obj)
    raise TypeError(f"Unsupported dataset object: {type(obj)!r}")


def _as_dataset_or_dict(obj: DatasetObject) -> LocalDataset:
    if isinstance(obj, (Dataset, DatasetDict)):
        return obj
    if isinstance(obj, list):
        return Dataset.from_list(obj)
    raise TypeError(f"Unsupported dataset object: {type(obj)!r}")


def _ordered_fieldnames(records: Iterable[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def load_jsonl(jsonl_path: str | Path) -> list[dict]:
    jsonl_path = _normalize_path(jsonl_path)
    rows: list[dict] = []
    with jsonl_path.open("rb") as file:
        for line in file:
            if not line.strip():
                continue
            if orjson is not None:
                rows.append(orjson.loads(line))
            else:
                rows.append(json.loads(line.decode("utf-8")))
    return rows


def load_single_dataset(dataset_path: str | Path, dataset_split: str | None = None) -> LocalDataset | Dataset:
    path = _normalize_path(dataset_path)
    if path.exists():
        if path.is_file():
            suffix = _match_suffix(path, _LOADABLE_SUFFIXES)
            if suffix == ".arrow":
                return Dataset.from_file(str(path))
            if suffix is None:
                raise RuntimeError(f"No support file type for {path.suffix}")
            return _load_tabular_files([path], suffix, dataset_split=dataset_split)
        if path.is_dir():
            return _load_local_directory(path, dataset_split=dataset_split)

    try:
        return hf_load_dataset(str(dataset_path), split=dataset_split)
    except ValueError:
        dataset = load_from_disk(str(dataset_path))
        return _select_dataset_split(dataset, dataset_split)


def load_dataset(dataset_paths: list[str] | list[Path] | str | Path, dataset_split: str | None = None) -> Dataset:
    if isinstance(dataset_paths, (str, Path)):
        dataset_paths = [dataset_paths]

    loaded_datasets: list[Dataset] = []
    for dataset_path in dataset_paths:
        try:
            dataset = load_single_dataset(dataset_path, dataset_split=dataset_split)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[_default_split_name(dataset)]
            loaded_datasets.append(dataset)
        except Exception as error:
            warnings.warn(f"Invalid dataset, dataset: {dataset_path}, error: {error}")

    if not loaded_datasets:
        raise RuntimeError("No valid dataset")
    if len(loaded_datasets) == 1:
        return loaded_datasets[0]
    return concatenate_datasets(loaded_datasets)


def load_parquet_dataset(paths: list[str]) -> Dataset:
    return _load_tabular_files([_normalize_path(path) for path in paths], ".parquet")


def load_records(data_path: str | Path, dataset_split: str | None = None) -> list[dict]:
    dataset = load_single_dataset(data_path, dataset_split=dataset_split)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[dataset_split or _default_split_name(dataset)]
    return dataset.to_list()


def save_dataset(obj: DatasetObject, save_path: str | Path) -> None:
    output_path = _normalize_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = _match_suffix(output_path, _TABULAR_SUFFIXES)
    if suffix == ".parquet":
        dataset = _as_single_dataset(obj, output_path)
        dataset.to_parquet(str(output_path))
        return

    if suffix == ".json":
        dataset = _as_single_dataset(obj, output_path)
        output_path.write_text(
            json.dumps(dataset.to_list(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    if suffix == ".jsonl":
        dataset = _as_single_dataset(obj, output_path)
        with output_path.open("w", encoding="utf-8") as file:
            for row in dataset.to_list():
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if suffix in {".csv", ".tsv"}:
        dataset = _as_single_dataset(obj, output_path)
        records = dataset.to_list()
        fieldnames = dataset.column_names or _ordered_fieldnames(records)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                delimiter="\t" if suffix == ".tsv" else ",",
                extrasaction="ignore",
            )
            if fieldnames:
                writer.writeheader()
            for row in records:
                writer.writerow(row)
        return

    dataset = _as_dataset_or_dict(obj)
    dataset.save_to_disk(str(output_path))


def save_records_as_parquet(records: list[dict], output_path: Path) -> None:
    if not records:
        raise ValueError(f"No records to save for {output_path}")
    save_dataset(records, output_path)


def preview_records(records: list[dict], count: int = 2) -> None:
    preview_count = min(count, len(records))
    for index in range(preview_count):
        print(f"\n===== Preview {index + 1} =====")
        print(json.dumps(records[index], ensure_ascii=False, indent=2))


__all__ = [
    "load_dataset",
    "load_jsonl",
    "load_parquet_dataset",
    "load_records",
    "load_single_dataset",
    "preview_records",
    "save_dataset",
    "save_records_as_parquet",
]
