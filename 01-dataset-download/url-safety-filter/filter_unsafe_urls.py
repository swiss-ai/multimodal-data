"""Pre-filter URLs in parquet datasets against cybersecurity blocklists.

Three-layer filtering:
  1. Domain blocklists  — URLhaus (malware), Phishing Army, Phishing.Database,
                          Maltrail, stamparm/blackbook, scam blocklist
  2. IP blocklists      — FireHOL level1 (botnet/C2/spyware CIDRs):
                          resolve each unique domain → check IP against CIDRs
  3. Private / bogon IP — reject URLs pointing to RFC-1918 / loopback / link-local

Usage:
    python filter_unsafe_urls.py \
        --input  /path/to/dataset.parquet \
        --output /path/to/dataset_safe.parquet \
        --url-column url \
        --workers 64

    # Or process a whole directory of parquets:
    python filter_unsafe_urls.py \
        --input-dir  /path/to/parquets/ \
        --output-dir /path/to/parquets_safe/ \
        --url-column url \
        --workers 64
"""

import argparse
import asyncio
import csv
import ipaddress
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import pyarrow as pa

# ---------------------------------------------------------------------------
# Blocklist sources (freely downloadable, no auth required)
# ---------------------------------------------------------------------------
BLOCKLIST_SOURCES = {
    # Domain lists (one domain per line, # comments)
    "phishing_army": {
        "url": "https://phishing.army/download/phishing_army_blocklist_extended.txt",
        "type": "domain",
    },
    "phishing_database": {
        "url": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt",
        "type": "domain",
    },
    "maltrail_malware": {
        "url": "https://raw.githubusercontent.com/stamparm/aux/master/maltrail-malware-domains.txt",
        "type": "domain",
    },
    "blackbook": {
        "url": "https://raw.githubusercontent.com/stamparm/blackbook/master/blackbook.txt",
        "type": "domain",
    },
    "scam_blocklist": {
        "url": "https://raw.githubusercontent.com/durablenapkin/scamblocklist/master/hosts.txt",
        "type": "hosts",  # 0.0.0.0 domain format
    },
    # URL list (CSV with url column)
    "urlhaus_online": {
        "url": "https://urlhaus.abuse.ch/downloads/csv_online/",
        "type": "urlhaus_csv",
    },
    # IP/CIDR list
    "firehol_level1": {
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset",
        "type": "cidr",
    },
    "firehol_level2": {
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level2.netset",
        "type": "cidr",
    },
}

BLOCKLIST_FILE_HEADER = "# url-safety-filter blocklist v1"
ALLOWED_URL_SCHEMES = {"http", "https"}


def normalize_host(host: str | None) -> str | None:
    """Normalize a hostname or IP literal for matching."""
    if host is None:
        return None

    host = host.strip().strip('"').rstrip(".").lower()
    if not host:
        return None

    if host.startswith("*."):
        host = host[2:]
    host = host.lstrip(".")
    if not host:
        return None

    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass

    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def parse_domain_entry(raw_entry: str) -> str | None:
    """Extract a normalized hostname from a blocklist entry."""
    entry = raw_entry.strip()
    if not entry or entry.startswith("#"):
        return None

    entry = entry.split("#", 1)[0].strip().strip('"')
    if not entry:
        return None

    if "://" in entry:
        return normalize_host(urlparse(entry).hostname)

    return normalize_host(entry)


def is_ip_literal(host: str) -> bool:
    """Return whether host is an IP literal."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def iter_domain_candidates(host: str):
    """Yield a hostname and its parent domains for suffix matching."""
    yield host
    if is_ip_literal(host):
        return

    labels = host.split(".")
    for start in range(1, max(len(labels) - 1, 1)):
        yield ".".join(labels[start:])


def is_domain_blocked(host: str, blocked_domains: set[str]) -> bool:
    """Check a host against an exact-or-parent domain blocklist."""
    return any(candidate in blocked_domains for candidate in iter_domain_candidates(host))


def is_non_public_ip(addr: ipaddress._BaseAddress) -> bool:
    """Reject private, loopback, link-local, multicast, reserved, and unspecified IPs."""
    return not addr.is_global


def collapse_cidrs(cidrs: list) -> list:
    """Collapse IPv4 and IPv6 CIDRs separately."""
    grouped = {4: [], 6: []}
    for net in cidrs:
        grouped[net.version].append(net)

    collapsed = []
    for version in (4, 6):
        if grouped[version]:
            collapsed.extend(ipaddress.collapse_addresses(grouped[version]))
    return collapsed


def save_blocklist(path: Path, blocked_domains: set[str], blocked_cidrs: list) -> None:
    """Save blocked domains and CIDRs to a tagged text file."""
    lines = [BLOCKLIST_FILE_HEADER]
    lines.extend(f"DOMAIN\t{domain}" for domain in sorted(blocked_domains))
    lines.extend(f"CIDR\t{cidr}" for cidr in blocked_cidrs)
    path.write_text("\n".join(lines) + "\n")


def load_blocklist(path: Path) -> tuple[set[str], list]:
    """Load blocked domains and CIDRs from a saved blocklist file."""
    blocked_domains = set()
    blocked_cidrs = []
    saw_tagged_entries = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("DOMAIN\t"):
            saw_tagged_entries = True
            domain = normalize_host(line.split("\t", 1)[1])
            if domain:
                blocked_domains.add(domain)
            continue

        if line.startswith("CIDR\t"):
            saw_tagged_entries = True
            try:
                blocked_cidrs.append(ipaddress.ip_network(line.split("\t", 1)[1], strict=False))
            except ValueError:
                pass
            continue

        # Legacy plain-domain format.
        domain = normalize_host(line)
        if domain:
            blocked_domains.add(domain)

    if not saw_tagged_entries and blocked_domains:
        print(
            "WARNING: Loaded legacy domain-only blocklist file; direct non-public IPs remain blocked, "
            "but regenerate the file to preserve external CIDR blocklists."
        )

    return blocked_domains, collapse_cidrs(blocked_cidrs)


def resolve_output_path(
    pq_path: Path,
    input_base: Path | None,
    output_base: Path | None,
    explicit_output_path: Path | None = None,
) -> Path:
    """Resolve the destination parquet path for a filtering task."""
    if explicit_output_path is not None:
        return Path(explicit_output_path)

    rel = pq_path.relative_to(input_base) if input_base else Path(pq_path.name)
    if output_base is not None:
        return output_base / rel
    return pq_path.parent / f"{pq_path.stem}_safe.parquet"


def should_block_host(host: str | None, blocked_domains: set[str], blocked_cidrs: list) -> bool:
    """Return whether a normalized host should be filtered."""
    if host is None:
        return True

    if is_domain_blocked(host, blocked_domains):
        return True

    if is_ip_literal(host):
        return check_ip_against_cidrs(host, blocked_cidrs)

    return False


def should_block_url(url: str | None, blocked_domains: set[str], blocked_cidrs: list) -> bool:
    """Return whether a URL should be filtered."""
    if url is None:
        return True
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        return True
    return should_block_host(normalize_host(parsed.hostname), blocked_domains, blocked_cidrs)


def download_blocklist(name: str, info: dict) -> tuple[str, set, list, str | None]:
    """Download a single blocklist. Returns (name, domain_set, cidr_list, error)."""
    import urllib.request

    domains = set()
    cidrs = []

    print(f"  Downloading {name} from {info['url']} ...")
    try:
        req = urllib.request.Request(info["url"], headers={"User-Agent": "url-safety-filter/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return name, domains, cidrs, str(e)

    if info["type"] == "domain":
        for line in raw.splitlines():
            domain = parse_domain_entry(line)
            if domain:
                domains.add(domain)

    elif info["type"] == "hosts":
        # Format: 0.0.0.0 domain or 127.0.0.1 domain
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    d = normalize_host(parts[1])
                    if d and d not in ("localhost", "localhost.localdomain"):
                        domains.add(d)

    elif info["type"] == "urlhaus_csv":
        # CSV with # comment header, fields: id,dateadded,url,url_status,...
        lines = [l for l in raw.splitlines() if l and not l.startswith("#")]
        reader = csv.reader(lines)
        for row in reader:
            if len(row) >= 3:
                try:
                    parsed = urlparse(row[2].strip('"'))
                    host = normalize_host(parsed.hostname)
                    if host:
                        domains.add(host)
                except Exception:
                    pass

    elif info["type"] == "cidr":
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    cidrs.append(ipaddress.ip_network(line, strict=False))
                except ValueError:
                    pass

    return name, domains, cidrs, None


def download_all_blocklists(allow_partial: bool = False) -> tuple[set, list]:
    """Download all blocklists and merge. Returns (blocked_domains, blocked_cidrs)."""
    from concurrent.futures import ThreadPoolExecutor

    all_domains = set()
    all_cidrs = []
    failures = []

    print("Downloading blocklists...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(download_blocklist, name, info): name
            for name, info in BLOCKLIST_SOURCES.items()
        }
        for future in as_completed(futures):
            name, domains, cidrs, error = future.result()
            if error is not None:
                failures.append((name, error))
                print(f"  ERROR: Failed to download {name}: {error}")
                continue
            print(f"  {name}: {len(domains)} domains, {len(cidrs)} CIDRs")
            all_domains |= domains
            all_cidrs.extend(cidrs)

    if failures and not allow_partial:
        failed_names = ", ".join(name for name, _ in sorted(failures))
        raise RuntimeError(
            "Failed to download required blocklists: "
            f"{failed_names}. Re-run when the sources are reachable, or pass --allow-partial-blocklists "
            "to proceed with reduced coverage."
        )

    all_cidrs = collapse_cidrs(all_cidrs)

    print(f"\nTotal blocked domains: {len(all_domains):,}")
    print(f"Total blocked CIDRs:  {len(all_cidrs):,}")
    return all_domains, all_cidrs


def resolve_domains_async(domains: list[str], max_concurrent: int = 500) -> dict[str, list[str]]:
    """Resolve a list of domains to IPs using async DNS. Returns {domain: [ip, ...]}."""

    async def resolve_one(sem, domain):
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                infos = await loop.getaddrinfo(domain, None, family=socket.AF_UNSPEC)
                ips = list({info[4][0] for info in infos})
                return domain, ips
            except (socket.gaierror, OSError):
                return domain, []

    async def resolve_all(domains_list):
        sem = asyncio.Semaphore(max_concurrent)
        tasks = [resolve_one(sem, d) for d in domains_list]
        results = {}
        done = 0
        for coro in asyncio.as_completed(tasks):
            domain, ips = await coro
            results[domain] = ips
            done += 1
            if done % 10000 == 0 or done == len(domains_list):
                print(f"  Resolved {done:,}/{len(domains_list):,} domains")
        return results

    return asyncio.run(resolve_all(domains))


def check_ip_against_cidrs(ip_str: str, cidrs: list) -> bool:
    """Check if an IP falls within any blocked CIDR range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if is_non_public_ip(addr):
        return True
    for net in cidrs:
        if addr in net:
            return True
    return False


def build_blocked_domain_set(
    blocked_domains: set,
    blocked_cidrs: list,
    all_domains: list[str],
    resolve_dns: bool = True,
) -> set:
    """Build the full set of domains to block (blocklist + IP-resolved)."""
    # Start with direct domain matches
    final_blocked = set(blocked_domains)

    if not resolve_dns:
        return final_blocked

    # Resolve domains NOT already blocked to check their IPs
    to_resolve = [d for d in all_domains if d not in final_blocked]
    print(f"\nResolving {len(to_resolve):,} domains for IP-based filtering...")
    t0 = time.time()
    dns_results = resolve_domains_async(to_resolve)
    dt = time.time() - t0
    print(f"DNS resolution took {dt:.1f}s")

    # Check resolved IPs against CIDR blocklists
    ip_blocked = 0
    unresolvable = 0
    for domain, ips in dns_results.items():
        if not ips:
            unresolvable += 1
            continue
        for ip in ips:
            if check_ip_against_cidrs(ip, blocked_cidrs):
                final_blocked.add(domain)
                ip_blocked += 1
                break

    print(f"IP-blocked domains:   {ip_blocked:,}")
    print(f"Unresolvable domains: {unresolvable:,}")
    print(f"Total blocked:        {len(final_blocked):,}")
    return final_blocked


def extract_domain(url: str) -> str | None:
    """Extract lowercase domain from a URL string."""
    try:
        parsed = urlparse(url)
        return normalize_host(parsed.hostname)
    except Exception:
        return None


def filter_parquet(args: tuple) -> tuple[str, int, int]:
    """Filter a single parquet file. Returns (rel_path, original_rows, kept_rows)."""
    pq_path, input_base, output_base, explicit_output_path, url_column, blocked_domains, blocked_cidrs = args

    rel = pq_path.relative_to(input_base) if input_base else pq_path.name
    out_path = resolve_output_path(pq_path, input_base, output_base, explicit_output_path)

    table = pq.read_table(pq_path)
    n_original = len(table)

    urls = table.column(url_column).to_pylist()
    keep_mask = []
    for url in urls:
        keep_mask.append(not should_block_url(url, blocked_domains, blocked_cidrs))

    mask_array = pa.array(keep_mask, type=pa.bool_())
    filtered = table.filter(mask_array)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, out_path)

    return str(rel), n_original, len(filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Filter URLs in parquet datasets against cybersecurity blocklists"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single parquet file")
    group.add_argument("--input-dir", type=Path, help="Directory of parquet files")
    parser.add_argument("--output", type=Path, help="Output parquet (for --input)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (for --input-dir)")
    parser.add_argument("--url-column", default="url", help="Column containing URLs (default: url)")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers for parquet processing")
    parser.add_argument("--no-dns", action="store_true", help="Skip DNS resolution (only use domain blocklists)")
    parser.add_argument(
        "--save-blocklist",
        type=Path,
        help="Save merged blocked domains and CIDRs to file",
    )
    parser.add_argument(
        "--load-blocklist",
        type=Path,
        help="Load pre-computed blocked domains and CIDRs from file",
    )
    parser.add_argument(
        "--allow-partial-blocklists",
        action="store_true",
        help="Proceed even if one or more blocklist sources fail to download",
    )
    args = parser.parse_args()

    if args.input and args.output_dir:
        parser.error("--output-dir may only be used with --input-dir")
    if args.input_dir and args.output:
        parser.error("--output may only be used with --input")

    # Collect parquet files
    if args.input:
        parquets = [args.input]
        input_base = args.input.parent
        output_base = None
    else:
        parquets = sorted(args.input_dir.rglob("*.parquet"))
        input_base = args.input_dir
        output_base = args.output_dir
        if not parquets:
            print(f"No parquet files found in {args.input_dir}")
            sys.exit(1)

    print(f"Found {len(parquets)} parquet file(s)")

    # --- Step 1: Build blocked domain set ---
    if args.load_blocklist:
        print(f"Loading pre-computed blocklist from {args.load_blocklist}")
        blocked_domains, blocked_cidrs = load_blocklist(args.load_blocklist)
        print(f"Loaded {len(blocked_domains):,} blocked domains")
        print(f"Loaded {len(blocked_cidrs):,} blocked CIDRs")
    else:
        # Download blocklists
        domain_blocklist, blocked_cidrs = download_all_blocklists(
            allow_partial=args.allow_partial_blocklists
        )

        # Extract unique domains from all parquets
        print(f"\nExtracting unique domains from {len(parquets)} parquet file(s)...")
        all_domains = set()
        for pq_path in parquets:
            table = pq.read_table(pq_path, columns=[args.url_column])
            urls = table.column(args.url_column).to_pylist()
            for url in urls:
                if url:
                    d = extract_domain(url)
                    if d:
                        all_domains.add(d)
        print(f"Unique domains: {len(all_domains):,}")

        # Build full blocked set (domain blocklist + IP check)
        blocked_domains = build_blocked_domain_set(
            domain_blocklist, blocked_cidrs, list(all_domains), resolve_dns=not args.no_dns
        )

        if args.save_blocklist:
            args.save_blocklist.parent.mkdir(parents=True, exist_ok=True)
            save_blocklist(args.save_blocklist, blocked_domains, blocked_cidrs)
            print(
                f"Saved {len(blocked_domains):,} blocked domains and "
                f"{len(blocked_cidrs):,} CIDRs to {args.save_blocklist}"
            )

    # --- Step 2: Filter parquets ---
    print(f"\nFiltering {len(parquets)} parquet file(s) with {args.workers} workers...")

    if args.input and args.output:
        tasks = [
            (args.input, input_base, output_base, args.output, args.url_column, blocked_domains, blocked_cidrs)
        ]
    else:
        tasks = [
            (pq_path, input_base, output_base, None, args.url_column, blocked_domains, blocked_cidrs)
            for pq_path in parquets
        ]

    total_original = 0
    total_kept = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(filter_parquet, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            rel, n_orig, n_kept = future.result()
            total_original += n_orig
            total_kept += n_kept
            done += 1
            removed = n_orig - n_kept
            if done % 20 == 0 or done == len(parquets):
                print(f"  [{done}/{len(parquets)}] {rel}: {n_orig:,} → {n_kept:,} (-{removed:,})")

    total_removed = total_original - total_kept
    pct = (total_removed / total_original * 100) if total_original else 0
    print(f"\nDone. {total_original:,} rows → {total_kept:,} rows ({total_removed:,} removed, {pct:.2f}%)")


if __name__ == "__main__":
    main()
