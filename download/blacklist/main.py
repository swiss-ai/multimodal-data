#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import feed_parsers
from common import (
    collapse_ip_networks,
    counts,
    empty_indicators,
    merge_indicators,
    render_domains,
    write_text_file_atomic,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SQUID_BINARY = "/users/tchu/.squid/sbin/squid"
USER_AGENT = "cscs-blacklist-builder/1.0"


FEEDS = [
    {
        "name": "abuseipdb_s100_1d_ipv4",
        "url": "https://raw.githubusercontent.com/borestad/blocklist-abuseipdb/main/abuseipdb-s100-1d.ipv4",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "alienvault_reputation_generic",
        "url": "https://reputation.alienvault.com/reputation.generic",
        "parser": feed_parsers.parse_alienvault,
    },
    {
        "name": "binarydefense_banlist",
        "url": "https://www.binarydefense.com/banlist.txt",
        "parser": feed_parsers.parse_binarydefense,
    },
    {
        "name": "firehol_level1",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "firehol_abusers_30d",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_abusers_30d.netset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "firehol_webclient",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_webclient.netset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "firehol_webserver",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_webserver.netset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "firehol_bitcoin_nodes_1d",
        "url": "https://iplists.firehol.org/files/bitcoin_nodes_1d.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "stamparm_blackbook",
        "url": "https://raw.githubusercontent.com/stamparm/blackbook/master/blackbook.csv",
        "parser": feed_parsers.parse_blackbook_csv,
    },
    {
        "name": "blackhole_monster_today",
        "url": "https://blackhole.monster/blackhole-today",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "blocklist_de_all",
        "url": "https://lists.blocklist.de/lists/all.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "firehol_botscout_1d",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/botscout_1d.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "danger_rulez_bruteforceblocker",
        "url": "https://danger.rulez.sk/projects/bruteforceblocker/blist.php",
        "parser": feed_parsers.parse_bruteforceblocker,
    },
    {
        "name": "cinsscore_badguys",
        "url": "https://cinsscore.com/list/ci-badguys.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "firehol_cleantalk_1d",
        "url": "https://iplists.firehol.org/files/cleantalk_1d.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "foxit_cobaltstrike_servers",
        "url": "https://raw.githubusercontent.com/fox-it/cobaltstrike-extraneous-space/master/cobaltstrike-servers.csv",
        "parser": feed_parsers.parse_cobaltstrike_csv,
    },
    {
        "name": "firehol_dshield_top_1000",
        "url": "https://iplists.firehol.org/files/dshield_top_1000.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "emergingthreats_botcc_rules",
        "url": "https://rules.emergingthreats.net/open/suricata/rules/botcc.rules",
        "parser": feed_parsers.parse_suricata_ip_rules,
    },
    {
        "name": "emergingthreats_compromised_ips",
        "url": "https://rules.emergingthreats.net/open/suricata/rules/compromised-ips.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "emergingthreats_emerging_malware_rules",
        "url": "https://rules.emergingthreats.net/open/suricata/rules/emerging-malware.rules",
        "parser": feed_parsers.parse_suricata_ip_rules,
    },
    {
        "name": "firehol_gpf_comics",
        "url": "https://iplists.firehol.org/files/gpf_comics.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "greensnow_blacklist",
        "url": "https://blocklist.greensnow.co/greensnow.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "sekuripy_blacklist",
        "url": "https://www.sekuripy.hr/blacklist.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "firehol_maxmind_proxy_fraud",
        "url": "https://iplists.firehol.org/files/maxmind_proxy_fraud.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "minerchk_hostslist",
        "url": "https://raw.githubusercontent.com/Hestat/minerchk/master/hostslist.txt",
        "parser": feed_parsers.parse_domain_list,
    },
    {
        "name": "urlhaus_hostfile",
        "url": "https://urlhaus.abuse.ch/downloads/hostfile/",
        "parser": feed_parsers.parse_hostfile_domains,
    },
    {
        "name": "myip_ms_latest_blacklist",
        "url": "https://myip.ms/files/blacklist/htaccess/latest_blacklist.txt",
        "fallback_url": "http://myip.ms/files/blacklist/htaccess/latest_blacklist.txt",
        "parser": feed_parsers.parse_myip_ms,
    },
    {
        "name": "openphish_feed",
        "url": "https://openphish.com/feed.txt",
        "parser": feed_parsers.parse_generic_url_list,
    },
    {
        "name": "policeman_simple_domains_blacklist",
        "url": "https://raw.githubusercontent.com/futpib/policeman-rulesets/master/examples/simple_domains_blacklist.txt",
        "parser": feed_parsers.parse_domain_list,
    },
    {
        "name": "rutgers_drop_attackers",
        "url": "https://report.cs.rutgers.edu/DROP/attackers",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "sblam_blacklist",
        "url": "https://sblam.com/blacklist.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "scriptzteam_badips",
        "url": "https://raw.githubusercontent.com/scriptzteam/badIPS/main/ips.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "firehol_socks_proxy_7d",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/socks_proxy_7d.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "firehol_sslproxies_1d",
        "url": "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/sslproxies_1d.ipset",
        "parser": feed_parsers.parse_firehol_ipset,
    },
    {
        "name": "stamparm_maltrail_static_trails",
        "url": "https://raw.githubusercontent.com/stamparm/aux/master/maltrail-static-trails.txt",
        "parser": feed_parsers.parse_maltrail_static,
    },
    {
        "name": "tor_bulk_exit_list",
        "url": "https://check.torproject.org/cgi-bin/TorBulkExitList.py?ip=1.1.1.1",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "turris_greylist_latest",
        "url": "https://view.sentinel.turris.cz/greylist-data/greylist-latest.csv",
        "parser": feed_parsers.parse_turris_csv,
    },
    {
        "name": "urlhaus_text_urls",
        "url": "https://urlhaus.abuse.ch/downloads/text/",
        "parser": feed_parsers.parse_generic_url_list,
    },
    {
        "name": "spamhaus_dropv6",
        "url": "https://www.spamhaus.org/drop/dropv6.txt",
        "parser": feed_parsers.parse_ip_list,
    },
    {
        "name": "viriback_dump",
        "url": "https://tracker.viriback.com/dump.php",
        "parser": feed_parsers.parse_viriback_csv,
    },
]


def fetch_feed(feed, download_dir, timeout, allow_plain_http_fallback):
    destination = os.path.join(download_dir, "{}.download".format(feed["name"]))
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--max-time",
        str(timeout),
        "--user-agent",
        USER_AGENT,
        "--output",
        destination,
        feed["url"],
    ]

    warnings = []
    try:
        subprocess.check_call(command)
        return {
            "path": destination,
            "fetched_url": feed["url"],
            "transport": "https" if feed["url"].startswith("https://") else "http",
            "warnings": warnings,
        }
    except subprocess.CalledProcessError:
        fallback_url = feed.get("fallback_url")
        if not fallback_url or not allow_plain_http_fallback:
            raise
        fallback_command = list(command)
        fallback_command[-1] = fallback_url
        subprocess.check_call(fallback_command)
        warnings.append(
            "used plain-http fallback because the HTTPS endpoint failed strict validation"
        )
        return {
            "path": destination,
            "fetched_url": fallback_url,
            "transport": "http-fallback",
            "warnings": warnings,
        }


def validate_required_output(path):
    if not os.path.exists(path):
        raise RuntimeError("missing required output {}".format(path))
    if os.path.getsize(path) == 0:
        raise RuntimeError("required output is empty {}".format(path))


def validate_squid_config(output_dir, squid_binary):
    with tempfile.TemporaryDirectory(prefix="blacklist-squid-") as proxy_dir:
        config_path = os.path.join(proxy_dir, "squid.conf")
        access_log_path = os.path.join(proxy_dir, "access.log")
        subprocess.check_call(
            [
                os.path.join(SCRIPT_DIR, "proxy.sh"),
                "render-config",
                output_dir,
                config_path,
                access_log_path,
                "127.0.0.1",
                "3128",
            ]
        )
        process = subprocess.run(
            [squid_binary, "-k", "parse", "-f", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if process.returncode != 0:
            message = process.stdout or "squid parse failed without output"
            raise RuntimeError(message)


def collect_indicators(args):
    merged = empty_indicators()
    manifest_feeds = []
    warnings = []

    with tempfile.TemporaryDirectory(prefix="blacklist-feeds-") as download_dir:
        for feed in FEEDS:
            fetched = fetch_feed(
                feed=feed,
                download_dir=download_dir,
                timeout=args.timeout,
                allow_plain_http_fallback=args.allow_plain_http_fallback,
            )
            with open(fetched["path"], "rb") as handle:
                raw_bytes = handle.read()
            parsed = feed["parser"](raw_bytes)
            parsed_counts = counts(parsed)
            if sum(parsed_counts.values()) == 0:
                raise RuntimeError(
                    "feed {} produced zero indicators; refusing to publish".format(
                        feed["name"]
                    )
                )

            merge_indicators(merged, parsed)
            manifest_feeds.append(
                {
                    "name": feed["name"],
                    "requested_url": feed["url"],
                    "fetched_url": fetched["fetched_url"],
                    "transport": fetched["transport"],
                    "bytes": len(raw_bytes),
                    "counts": parsed_counts,
                    "warnings": list(fetched["warnings"]),
                }
            )
            warnings.extend(fetched["warnings"])

    return merged, manifest_feeds, warnings


def build_blacklist(args):
    output_dir = os.path.abspath(args.output_dir or SCRIPT_DIR)
    merged, manifest_feeds, warnings = collect_indicators(args)

    ip_lines = collapse_ip_networks(merged["ip_networks"])
    domain_lines = render_domains(merged["domains"])
    url_lines = sorted(merged["url_regexes"])
    url_path_lines = sorted(merged["url_path_regexes"])

    write_text_file_atomic(os.path.join(output_dir, "blocked_ip.txt"), ip_lines)
    write_text_file_atomic(
        os.path.join(output_dir, "blocked_domains.txt"), domain_lines
    )
    write_text_file_atomic(os.path.join(output_dir, "blocked_urls.txt"), url_lines)
    write_text_file_atomic(
        os.path.join(output_dir, "blocked_url_paths.txt"), url_path_lines
    )

    manifest = {
        "built_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "warnings": warnings,
        "totals": {
            "blocked_ip.txt": len(ip_lines),
            "blocked_domains.txt": len(domain_lines),
            "blocked_urls.txt": len(url_lines),
            "blocked_url_paths.txt": len(url_path_lines),
        },
        "feeds": manifest_feeds,
    }
    write_text_file_atomic(
        os.path.join(output_dir, "manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True).splitlines(),
    )

    validate_required_output(os.path.join(output_dir, "blocked_ip.txt"))
    validate_required_output(os.path.join(output_dir, "blocked_domains.txt"))
    validate_required_output(os.path.join(output_dir, "blocked_urls.txt"))
    if not args.skip_squid_parse:
        validate_squid_config(output_dir, args.squid_binary)

    print(json.dumps(manifest["totals"], sort_keys=True))
    if warnings:
        for warning in warnings:
            print("warning: {}".format(warning), file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Squid ACL files from public threat-intelligence feeds."
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output-dir")
    build_parser.add_argument("--timeout", type=int, default=120)
    build_parser.add_argument("--allow-plain-http-fallback", action="store_true")
    build_parser.add_argument("--skip-squid-parse", action="store_true")
    build_parser.add_argument("--squid-binary", default=DEFAULT_SQUID_BINARY)

    args = parser.parse_args()
    if args.command is None:
        args.command = "build"
        args.output_dir = None
        args.timeout = 120
        args.allow_plain_http_fallback = False
        args.skip_squid_parse = False
        args.squid_binary = DEFAULT_SQUID_BINARY
    return args


def main():
    args = parse_args()
    if args.command == "build":
        build_blacklist(args)
    else:
        raise SystemExit("unsupported command {}".format(args.command))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
