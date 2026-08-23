#!/usr/bin/env python3

import csv
import io

from common import (
    add_domain,
    add_exact_url,
    add_host_token,
    add_ip,
    add_url_path,
    csv_dict_reader,
    empty_indicators,
    iter_data_lines,
    iter_text_lines,
    strip_inline_comment,
    suricata_header_ip_literals,
)


def parse_ip_list(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        add_ip(indicators, strip_inline_comment(line))
    return indicators


def parse_alienvault(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        add_ip(indicators, line.split()[0])
    return indicators


def parse_binarydefense(raw_bytes):
    return parse_ip_list(raw_bytes)


def parse_firehol_ipset(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        parts = line.split()
        candidate = parts[-1]
        add_ip(indicators, candidate)
    return indicators


def parse_bruteforceblocker(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        add_ip(indicators, line.split()[0])
    return indicators


def parse_blackbook_csv(raw_bytes):
    indicators = empty_indicators()
    for row in csv_dict_reader(raw_bytes):
        try:
            add_domain(indicators, row["Domain"])
        except ValueError:
            continue
    return indicators


def parse_cobaltstrike_csv(raw_bytes):
    indicators = empty_indicators()
    for row in csv_dict_reader(raw_bytes):
        add_ip(indicators, row["ip"])
    return indicators


def parse_generic_url_list(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        try:
            add_exact_url(indicators, line)
        except ValueError:
            continue
    return indicators


def parse_viriback_csv(raw_bytes):
    indicators = empty_indicators()
    for row in csv_dict_reader(raw_bytes):
        try:
            add_exact_url(indicators, row["URL"])
        except ValueError:
            pass
        if row.get("IP"):
            try:
                add_ip(indicators, row["IP"])
            except ValueError:
                pass
    return indicators


def parse_domain_list(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        try:
            add_domain(indicators, strip_inline_comment(line))
        except ValueError:
            continue
    return indicators


def parse_hostfile_domains(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            add_domain(indicators, parts[1])
        except ValueError:
            continue
    return indicators


def parse_myip_ms(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        if not line.lower().startswith("deny from "):
            continue
        add_ip(indicators, line.split(None, 2)[2])
    return indicators


def parse_turris_csv(raw_bytes):
    indicators = empty_indicators()
    lines = []
    for line in iter_text_lines(raw_bytes):
        if not line.strip() or line.startswith("#"):
            continue
        lines.append(line)
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        try:
            add_ip(indicators, row["Address"])
        except (KeyError, ValueError):
            continue
    return indicators


def parse_maltrail_static(raw_bytes):
    indicators = empty_indicators()
    for line in iter_data_lines(raw_bytes):
        indicator = line.split(",", 1)[0].strip()
        if not indicator:
            continue
        try:
            if indicator.startswith("/"):
                add_url_path(indicators, indicator)
            elif "://" in indicator:
                add_exact_url(indicators, indicator)
            elif ":" in indicator and "/" not in indicator:
                add_host_token(indicators, indicator)
            else:
                add_ip(indicators, indicator)
        except ValueError:
            try:
                add_domain(indicators, indicator)
            except ValueError:
                continue
    return indicators


def parse_suricata_ip_rules(raw_bytes):
    indicators = empty_indicators()
    for line in iter_text_lines(raw_bytes):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "(" not in line:
            continue
        for literal in suricata_header_ip_literals(line):
            add_ip(indicators, literal)
    return indicators
