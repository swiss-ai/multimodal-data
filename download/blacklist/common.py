#!/usr/bin/env python3

import csv
import io
import ipaddress
import os
import re
import tempfile
import urllib.parse

DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def empty_indicators():
    return {
        "ip_networks": set(),
        "domains": set(),
        "url_regexes": set(),
        "url_path_regexes": set(),
    }


def counts(indicators):
    return {
        "ip_networks": len(indicators["ip_networks"]),
        "domains": len(indicators["domains"]),
        "url_regexes": len(indicators["url_regexes"]),
        "url_path_regexes": len(indicators["url_path_regexes"]),
    }


def merge_indicators(target, source):
    for key in target:
        target[key].update(source[key])


def decode_text(raw_bytes):
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def iter_text_lines(raw_bytes):
    for line in decode_text(raw_bytes).replace("\r", "").splitlines():
        yield line


def iter_data_lines(raw_bytes, comment_markers=("#", ";")):
    for raw_line in iter_text_lines(raw_bytes):
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(marker) for marker in comment_markers):
            continue
        yield line


def strip_inline_comment(value, markers=("#", ";")):
    stripped = value
    for marker in markers:
        marker_index = stripped.find(marker)
        if marker_index != -1:
            stripped = stripped[:marker_index]
    return stripped.strip()


def normalize_ip_network(value):
    network = ipaddress.ip_network(value.strip(), strict=False)
    if network.prefixlen == network.max_prefixlen:
        return str(network.network_address)
    return network.with_prefixlen


def normalize_domain(value):
    domain = value.strip().strip(".").lower()
    if not domain:
        raise ValueError("empty domain")
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        pass
    else:
        raise ValueError("IP literal is not a domain")

    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("invalid IDNA domain {!r}".format(domain)) from exc
    labels = ascii_domain.split(".")
    if len(labels) < 2:
        raise ValueError("expected a multi-label hostname")
    if len(ascii_domain) > 253:
        raise ValueError("domain too long")
    for label in labels:
        if not DOMAIN_LABEL_RE.match(label):
            raise ValueError("invalid domain label {!r}".format(label))
    return ascii_domain


def _normalize_url_host(host):
    try:
        return normalize_domain(host)
    except ValueError:
        address = ipaddress.ip_address(host)
        return str(address)


def normalize_exact_url(value):
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("unsupported URL scheme {!r}".format(parsed.scheme))
    if not parsed.netloc:
        raise ValueError("URL is missing a netloc")
    host = parsed.hostname
    if not host:
        raise ValueError("URL is missing a hostname")

    normalized_host = _normalize_url_host(host)
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = "[{}]".format(normalized_host)

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += ":" + parsed.password
        userinfo += "@"

    netloc = userinfo + normalized_host
    if parsed.port is not None:
        netloc += ":{}".format(parsed.port)

    path = parsed.path or "/"
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), netloc, path, parsed.query, "")
    )


def regex_escape(value):
    return re.escape(value)


def add_ip(indicators, value):
    indicators["ip_networks"].add(normalize_ip_network(value))


def add_domain(indicators, value):
    indicators["domains"].add(normalize_domain(value))


def add_exact_url(indicators, value):
    normalized_url = normalize_exact_url(value)
    indicators["url_regexes"].add("^{}$".format(regex_escape(normalized_url)))
    # An explicit proxy sees only host:port for HTTPS CONNECT requests. Promote
    # the destination to a host-level ACL so HTTPS indicators remain effective
    # without TLS interception. This intentionally blocks the whole host.
    parsed = urllib.parse.urlsplit(normalized_url)
    classify_host(indicators, parsed.hostname)


def add_url_regex(indicators, regex_value):
    indicators["url_regexes"].add(regex_value)


def add_url_path(indicators, path_value):
    if not path_value.startswith("/"):
        raise ValueError("expected an absolute URL path")
    indicators["url_path_regexes"].add("^{}([?#].*)?$".format(regex_escape(path_value)))


def classify_host(indicators, value):
    try:
        add_ip(indicators, value)
        return "ip"
    except ValueError:
        add_domain(indicators, value)
        return "domain"


def add_host_token(indicators, value):
    parsed = urllib.parse.urlsplit("http://" + value.strip())
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("missing host token")
    classify_host(indicators, parsed.hostname)


def collapse_ip_networks(values):
    networks_v4 = []
    networks_v6 = []
    for value in values:
        network = ipaddress.ip_network(value, strict=False)
        if network.version == 4:
            networks_v4.append(network)
        else:
            networks_v6.append(network)

    collapsed = []
    collapsed.extend(ipaddress.collapse_addresses(networks_v4))
    collapsed.extend(ipaddress.collapse_addresses(networks_v6))

    normalized = []
    for network in collapsed:
        if network.prefixlen == network.max_prefixlen:
            normalized.append(str(network.network_address))
        else:
            normalized.append(network.with_prefixlen)
    normalized.sort(
        key=lambda value: (
            ipaddress.ip_network(value, strict=False).version,
            int(ipaddress.ip_network(value, strict=False).network_address),
            ipaddress.ip_network(value, strict=False).prefixlen,
        )
    )
    return normalized


def render_domains(values):
    rendered = []
    for domain in collapse_domains(values):
        rendered.append(".{}".format(domain))
    return rendered


def collapse_domains(values):
    accepted = set()
    ordered = sorted(values, key=lambda value: (len(value.split(".")), value))
    for domain in ordered:
        labels = domain.split(".")
        covered = domain in accepted
        if not covered:
            for index in range(1, len(labels)):
                suffix = ".".join(labels[index:])
                if suffix in accepted:
                    covered = True
                    break
        if not covered:
            accepted.add(domain)
    return sorted(accepted)


def csv_dict_reader(raw_bytes):
    return csv.DictReader(io.StringIO(decode_text(raw_bytes)))


def write_text_file_atomic(path, lines):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)

    handle = None
    tmp_path = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            delete=False,
        )
        tmp_path = handle.name
        if lines:
            handle.write("\n".join(lines))
            handle.write("\n")
        handle.close()
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def suricata_header_ip_literals(line):
    header = line.split("(", 1)[0]
    tokens = re.split(r"[\s,\[\]!]+", header)
    results = set()
    for token in tokens:
        token = token.strip()
        if not token or token.startswith("$"):
            continue
        if "/" not in token and "." not in token and ":" not in token:
            continue
        try:
            results.add(normalize_ip_network(token))
        except ValueError:
            continue
    return results
