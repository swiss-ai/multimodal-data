import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import img2dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "img2dataset is not installed in the active Python environment. "
        "Activate an environment that provides both img2dataset and pyarrow."
    ) from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is not installed in the active Python environment. "
        "Activate an environment that provides both pyarrow and img2dataset."
    ) from exc


DEFAULT_ROBOTS_PARQUET = "/path/to/data/users/vsabolce/apertus_v1/robotstxt/fineweb_robots_compressed.parquet"
DEFAULT_HOST_BLACKLIST = "/tmp/blacklist/hosts"


def parse_json_list(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    value = json.loads(raw_value)
    if not isinstance(value, list):
        raise ValueError("expected a JSON list")
    return value


def load_host_blacklist(path: Optional[str]) -> set[str]:
    if not path:
        return set()

    blocked_hosts = set()
    with open(path) as handle:
        for line in handle:
            host = line.strip().lower()
            if not host or host.startswith("#"):
                continue
            blocked_hosts.add(host)
    return blocked_hosts


def normalize_url(url: str) -> Optional[Tuple[str, str]]:
    parsed = urlsplit(url)
    if not parsed.hostname:
        return None
    scheme = (parsed.scheme or "https").lower()
    domain = parsed.hostname.lower()
    return scheme, domain


def parse_robots_policy(content: Optional[str]) -> Optional[RobotFileParser]:
    if not content:
        return None
    parser = RobotFileParser()
    parser.parse(content.splitlines())
    return parser


def choose_status_policy(status: str) -> bool:
    if not status:
        return True

    try:
        code = int(status)
    except ValueError:
        return True

    if 200 <= code < 300:
        return True
    if code in {404, 410}:
        return True
    if code in {401, 403, 429}:
        return False
    return True


def load_robots_records(
    robots_parquet: str, domains: Sequence[str]
) -> Dict[Tuple[str, str], Tuple[str, Optional[RobotFileParser]]]:
    if not domains:
        return {}

    robots_table = pq.read_table(
        robots_parquet,
        columns=["domain", "protocol", "status", "content"],
        filters=[("domain", "in", list(domains))],
    )

    records: Dict[Tuple[str, str], Tuple[str, Optional[RobotFileParser]]] = {}
    domain_list = robots_table.column("domain").to_pylist()
    protocol_list = robots_table.column("protocol").to_pylist()
    status_list = robots_table.column("status").to_pylist()
    content_list = robots_table.column("content").to_pylist()

    for domain, protocol, status, content in zip(domain_list, protocol_list, status_list, content_list):
        if not domain or not protocol:
            continue
        records[(protocol.lower(), domain.lower())] = (
            str(status or ""),
            parse_robots_policy(content),
        )

    return records


def collect_domains(input_file: str, url_col: str, batch_size: int = 100_000) -> List[str]:
    parquet_file = pq.ParquetFile(input_file)
    required_domains = set()

    for batch in parquet_file.iter_batches(columns=[url_col], batch_size=batch_size):
        for url in batch.column(0).to_pylist():
            if not isinstance(url, str):
                continue
            parsed = normalize_url(url)
            if parsed is not None:
                required_domains.add(parsed[1])

    return sorted(required_domains)


def collect_values(input_file: str, col_name: str, batch_size: int = 100_000) -> List[str]:
    parquet_file = pq.ParquetFile(input_file)
    values = set()

    for batch in parquet_file.iter_batches(columns=[col_name], batch_size=batch_size):
        for value in batch.column(0).to_pylist():
            if isinstance(value, str) and value:
                values.add(value)

    return sorted(values)


def load_caption_map_from_csv(
    csv_path: str, csv_key_col: str, csv_caption_col: str, required_keys: Sequence[str]
) -> Dict[str, str]:
    if not required_keys:
        return {}

    required_key_set = set(required_keys)
    caption_map: Dict[str, str] = {}

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get(csv_key_col)
            if not key or key not in required_key_set:
                continue
            caption = row.get(csv_caption_col)
            if caption is None:
                continue
            caption = caption.strip()
            if caption:
                caption_map[key] = caption

    return caption_map


def filter_batch_urls(
    urls: Sequence[Optional[str]],
    robots_records: Dict[Tuple[str, str], Tuple[str, Optional[RobotFileParser]]],
    robots_user_agent: str,
    blocked_hosts: set[str],
) -> Tuple[List[bool], int]:
    keep_mask: List[bool] = []
    kept_count = 0

    for url in urls:
        if not isinstance(url, str):
            keep_mask.append(False)
            continue

        parsed = normalize_url(url)
        if parsed is None:
            keep_mask.append(False)
            continue

        scheme, domain = parsed
        if domain in blocked_hosts:
            keep_mask.append(False)
            continue

        record = robots_records.get((scheme, domain))
        if record is None:
            record = robots_records.get(("https", domain))
        if record is None:
            record = robots_records.get(("http", domain))

        if record is None:
            allowed = True
        else:
            status, parser = record
            if parser is not None:
                allowed = parser.can_fetch(robots_user_agent, url)
            else:
                allowed = choose_status_policy(status)

        keep_mask.append(allowed)
        if allowed:
            kept_count += 1

    return keep_mask, kept_count


def filter_input_by_robots(
    input_file: str,
    output_folder: str,
    url_col: str,
    caption_col: Optional[str],
    caption_input_col: Optional[str],
    save_additional_columns: Optional[List[str]],
    robots_parquet: str,
    robots_user_agent: str,
    host_blacklist: Optional[str],
    caption_map_csv: Optional[str],
    caption_map_input_key_col: Optional[str],
    caption_map_csv_key_col: Optional[str],
    caption_map_csv_caption_col: Optional[str],
    drop_missing_captions: bool,
) -> Tuple[str, int, int, int]:
    selected_columns = [url_col]
    source_caption_col = caption_input_col or (caption_col if caption_col and not caption_map_csv else None)
    if source_caption_col:
        selected_columns.append(source_caption_col)
    if save_additional_columns:
        selected_columns.extend(save_additional_columns)
    if caption_map_input_key_col:
        selected_columns.append(caption_map_input_key_col)
    selected_columns = list(dict.fromkeys(selected_columns))

    required_domains = collect_domains(input_file, url_col)
    robots_records = load_robots_records(robots_parquet, required_domains)
    blocked_hosts = load_host_blacklist(host_blacklist)
    caption_map: Dict[str, str] = {}
    if caption_map_csv:
        if not caption_col or not caption_map_input_key_col or not caption_map_csv_key_col:
            raise ValueError("caption map arguments are incomplete")
        csv_caption_col = caption_map_csv_caption_col or caption_col
        required_keys = collect_values(input_file, caption_map_input_key_col)
        caption_map = load_caption_map_from_csv(
            csv_path=caption_map_csv,
            csv_key_col=caption_map_csv_key_col,
            csv_caption_col=csv_caption_col,
            required_keys=required_keys,
        )

    filtered_input_file = os.path.join(output_folder, "_filtered_input.parquet")

    total_count = 0
    kept_count = 0
    missing_caption_count = 0
    parquet_file = pq.ParquetFile(input_file)
    writer: Optional[pq.ParquetWriter] = None

    try:
        for batch in parquet_file.iter_batches(columns=selected_columns, batch_size=100_000):
            table = pa.Table.from_batches([batch])
            urls = table.column(url_col).to_pylist()
            keep_mask, batch_kept = filter_batch_urls(
                urls=urls,
                robots_records=robots_records,
                robots_user_agent=robots_user_agent,
                blocked_hosts=blocked_hosts,
            )
            total_count += len(urls)

            if batch_kept == 0:
                continue

            filtered_batch = table.filter(pa.array(keep_mask, type=pa.bool_()))
            if caption_map_csv:
                assert caption_col is not None
                assert caption_map_input_key_col is not None
                join_values = filtered_batch.column(caption_map_input_key_col).to_pylist()
                mapped_captions = [caption_map.get(value) for value in join_values]
                caption_keep_mask = [bool(caption) for caption in mapped_captions]
                batch_missing = len(mapped_captions) - sum(caption_keep_mask)
                missing_caption_count += batch_missing

                if drop_missing_captions and batch_missing > 0:
                    filtered_batch = filtered_batch.filter(pa.array(caption_keep_mask, type=pa.bool_()))
                    mapped_captions = [caption for caption in mapped_captions if caption]

                if filtered_batch.num_rows == 0:
                    continue

                filtered_batch = filtered_batch.append_column(caption_col, pa.array(mapped_captions, type=pa.string()))

            kept_count += filtered_batch.num_rows
            if writer is None:
                writer = pq.ParquetWriter(
                    filtered_input_file,
                    filtered_batch.schema,
                    compression="zstd",
                )
            writer.write_table(filtered_batch)
    finally:
        if writer is not None:
            writer.close()

    return filtered_input_file, total_count, kept_count, missing_caption_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--url_col", type=str, required=True)
    parser.add_argument("--caption_col", type=str, default=None)
    parser.add_argument("--caption_input_col", type=str, default=None)
    parser.add_argument("--save_additional_columns", type=str, default=None)
    parser.add_argument("--process_count", type=int, required=True)
    parser.add_argument("--thread_count", type=int, required=True)
    parser.add_argument("--number_sample_per_shard", type=int, default=100_000)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--incremental_mode", type=str, default="incremental")
    parser.add_argument("--robots_parquet", type=str, default=DEFAULT_ROBOTS_PARQUET)
    parser.add_argument("--robots_user_agent", type=str, default="img2dataset")
    parser.add_argument("--host_blacklist", type=str, default=DEFAULT_HOST_BLACKLIST)
    parser.add_argument("--caption_map_csv", type=str, default=None)
    parser.add_argument("--caption_map_input_key_col", type=str, default=None)
    parser.add_argument("--caption_map_csv_key_col", type=str, default=None)
    parser.add_argument("--caption_map_csv_caption_col", type=str, default=None)
    parser.add_argument("--drop_missing_captions", action="store_true")
    args = parser.parse_args()

    save_additional_columns = parse_json_list(args.save_additional_columns)
    os.makedirs(args.output_folder, exist_ok=True)

    filtered_input_file, input_count, kept_count, missing_caption_count = filter_input_by_robots(
        input_file=args.input_file,
        output_folder=args.output_folder,
        url_col=args.url_col,
        caption_col=args.caption_col,
        caption_input_col=args.caption_input_col,
        save_additional_columns=save_additional_columns,
        robots_parquet=args.robots_parquet,
        robots_user_agent=args.robots_user_agent,
        host_blacklist=args.host_blacklist,
        caption_map_csv=args.caption_map_csv,
        caption_map_input_key_col=args.caption_map_input_key_col,
        caption_map_csv_key_col=args.caption_map_csv_key_col,
        caption_map_csv_caption_col=args.caption_map_csv_caption_col,
        drop_missing_captions=args.drop_missing_captions,
    )

    if kept_count == 0:
        raise SystemExit(f"Robots filtering removed all rows from {args.input_file}; nothing to download.")

    print(f"Robots filter kept {kept_count} / {input_count} rows from {args.input_file} -> {filtered_input_file}")
    if args.caption_map_csv:
        print(f"Caption map missing matches for {missing_caption_count} rows")

    img2dataset.download(
        url_list=filtered_input_file,
        output_folder=args.output_folder,
        input_format="parquet",
        output_format="webdataset",
        url_col=args.url_col,
        caption_col=args.caption_col,
        save_additional_columns=save_additional_columns,
        processes_count=args.process_count,
        thread_count=args.thread_count,
        resize_mode="no",
        skip_reencode=True,
        number_sample_per_shard=args.number_sample_per_shard,
        timeout=args.timeout,
        retries=args.retries,
        compute_hash=None,
        incremental_mode=args.incremental_mode,
        enable_wandb=False,
    )
