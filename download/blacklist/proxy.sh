#!/usr/bin/env bash

# Source this file from a Slurm job, then call blacklist_proxy_start.

BLACKLIST_PROXY_RUNTIME_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

blacklist_proxy_stop() {
  if [[ -n "${BLACKLIST_PROXY_PID:-}" ]] && kill -0 "$BLACKLIST_PROXY_PID" 2>/dev/null; then
    kill "$BLACKLIST_PROXY_PID" 2>/dev/null || true
    wait "$BLACKLIST_PROXY_PID" 2>/dev/null || true
  fi
  unset BLACKLIST_PROXY_PID
}

blacklist_proxy_export_env() {
  local proxy_url=$1

  export HTTP_PROXY="$proxy_url"
  export HTTPS_PROXY="$proxy_url"
  export http_proxy="$proxy_url"
  export https_proxy="$proxy_url"
  unset ALL_PROXY all_proxy NO_PROXY no_proxy
}

blacklist_proxy_wait_for_port() {
  local host=$1
  local port=$2
  local attempt

  for ((attempt = 0; attempt < 50; attempt++)); do
    if command -v nc >/dev/null 2>&1; then
      nc -z "$host" "$port" >/dev/null 2>&1 && return 0
    elif python3 - "$host" "$port" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 0.2
  done

  return 1
}

blacklist_proxy_render_config() {
  local blacklist_dir=$1
  local config_path=$2
  local access_log_path=$3
  local listen_host=$4
  local listen_port=$5
  local optional_path_file="$blacklist_dir/blocked_url_paths.txt"
  local proxy_dir tmp_config

  proxy_dir=$(dirname -- "$access_log_path")
  tmp_config=$(mktemp "${config_path}.tmp.XXXXXX")

  {
    printf 'http_port %s:%s\n' "$listen_host" "$listen_port"
    printf '%s\n' 'pid_filename none'
    printf '%s\n' 'visible_hostname cscs-blacklist-proxy'
    printf '%s\n' 'logformat cscs_squid %ts.%03tu %6tr %>a %Ss/%03>Hs %<st %rm %ru %[un %Sh/%<a %mt %{blacklist_hit}note %{blacklist_reason}note'
    printf 'access_log stdio:%s logformat=cscs_squid\n' "$access_log_path"
    printf '%s\n' 'cache_store_log none'
    printf '%s\n' 'cache_log /dev/null'
    printf 'coredump_dir %s\n' "$proxy_dir"
    printf '%s\n' 'cache deny all'
    printf '%s\n' 'cache_mem 0 MB'
    printf '%s\n' 'memory_cache_mode never'
    printf '%s\n' 'shutdown_lifetime 0 seconds'
    printf '%s\n' 'via off'
    printf '%s\n' 'forwarded_for delete'
    printf 'acl blocked_ip dst "%s"\n' "$blacklist_dir/blocked_ip.txt"
    printf 'acl blocked_domains dstdomain "%s"\n' "$blacklist_dir/blocked_domains.txt"
    printf 'acl blocked_urls url_regex "%s"\n' "$blacklist_dir/blocked_urls.txt"
    printf '%s\n' 'acl SSL_ports port 443'
    printf '%s\n' 'acl Safe_ports port 80'
    printf '%s\n' 'acl Safe_ports port 443'
    printf '%s\n' 'acl CONNECT method CONNECT'
    printf '%s\n' 'acl mark_blacklist_hit annotate_transaction blacklist_hit=CSCS_BLACKLIST_HIT'
    printf '%s\n' 'acl mark_blacklist_reason_ip annotate_transaction blacklist_reason=ip'
    printf '%s\n' 'acl mark_blacklist_reason_domain annotate_transaction blacklist_reason=domain'
    printf '%s\n' 'acl mark_blacklist_reason_url annotate_transaction blacklist_reason=url'
    if [[ -s "$optional_path_file" ]]; then
      printf 'acl blocked_url_paths urlpath_regex "%s"\n' "$optional_path_file"
      printf '%s\n' 'acl mark_blacklist_reason_urlpath annotate_transaction blacklist_reason=urlpath'
    fi
    printf '%s\n' 'http_access deny !Safe_ports'
    printf '%s\n' 'http_access deny CONNECT !SSL_ports'
    printf '%s\n' 'http_access deny blocked_ip mark_blacklist_hit mark_blacklist_reason_ip'
    printf '%s\n' 'http_access deny blocked_domains mark_blacklist_hit mark_blacklist_reason_domain'
    printf '%s\n' 'http_access deny blocked_urls mark_blacklist_hit mark_blacklist_reason_url'
    if [[ -s "$optional_path_file" ]]; then
      printf '%s\n' 'http_access deny blocked_url_paths mark_blacklist_hit mark_blacklist_reason_urlpath'
    fi
    printf '%s\n' 'http_access allow all'
  } >"$tmp_config"

  mv -- "$tmp_config" "$config_path"
}

blacklist_proxy_start() {
  local work_dir=${1:-${BLACKLIST_PROXY_WORK_DIR:-${SLURM_TMPDIR:-$PWD}/blacklist-proxy}}
  local listen_host=${BLACKLIST_PROXY_HOST:-127.0.0.1}
  local listen_port=${BLACKLIST_PROXY_PORT:-3128}
  local squid_bin=${BLACKLIST_SQUID_BIN:-/users/tchu/.squid/sbin/squid}
  local proxy_url="http://${listen_host}:${listen_port}"
  local config_path access_log_path parse_log_path stdout_log_path required_file

  if [[ -n "${BLACKLIST_PROXY_PID:-}" ]] && kill -0 "$BLACKLIST_PROXY_PID" 2>/dev/null; then
    echo "blacklist proxy is already running (pid $BLACKLIST_PROXY_PID)" >&2
    return 1
  fi
  if [[ ! -x "$squid_bin" ]]; then
    echo "missing Squid binary: $squid_bin" >&2
    return 1
  fi
  for required_file in blocked_ip.txt blocked_domains.txt blocked_urls.txt; do
    if [[ ! -s "$BLACKLIST_PROXY_RUNTIME_DIR/$required_file" ]]; then
      echo "missing or empty blacklist: $BLACKLIST_PROXY_RUNTIME_DIR/$required_file" >&2
      return 1
    fi
  done

  mkdir -p -- "$work_dir"
  config_path="$work_dir/squid.conf"
  access_log_path="$work_dir/access.log"
  parse_log_path="$work_dir/squid.parse.log"
  stdout_log_path="$work_dir/squid.stdout.log"

  blacklist_proxy_render_config \
    "$BLACKLIST_PROXY_RUNTIME_DIR" "$config_path" "$access_log_path" \
    "$listen_host" "$listen_port"

  if ! "$squid_bin" -k parse -f "$config_path" >"$parse_log_path" 2>&1; then
    cat "$parse_log_path" >&2
    return 1
  fi

  "$squid_bin" -N -f "$config_path" >"$stdout_log_path" 2>&1 &
  BLACKLIST_PROXY_PID=$!
  export BLACKLIST_PROXY_PID

  if ! blacklist_proxy_wait_for_port "$listen_host" "$listen_port"; then
    cat "$stdout_log_path" >&2 || true
    blacklist_proxy_stop
    echo "Squid did not start on ${listen_host}:${listen_port}" >&2
    return 1
  fi

  export BLACKLIST_PROXY_URL="$proxy_url"
  export BLACKLIST_PROXY_ACCESS_LOG="$access_log_path"
  export BLACKLIST_PROXY_PARSE_LOG="$parse_log_path"
  export BLACKLIST_PROXY_STDOUT_LOG="$stdout_log_path"
  blacklist_proxy_export_env "$proxy_url"
  trap blacklist_proxy_stop EXIT
}

# Compatibility with the original positional interface.
blacklist_proxy_use() {
  local work_dir=${1:?proxy work directory is required}
  local blacklist_dir=${2:-$BLACKLIST_PROXY_RUNTIME_DIR}

  BLACKLIST_PROXY_RUNTIME_DIR=$blacklist_dir
  BLACKLIST_PROXY_HOST=${3:-${BLACKLIST_PROXY_HOST:-127.0.0.1}}
  BLACKLIST_PROXY_PORT=${4:-${BLACKLIST_PROXY_PORT:-3128}}
  blacklist_proxy_start "$work_dir"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${1:-}" == "render-config" && $# -eq 6 ]]; then
    blacklist_proxy_render_config "$2" "$3" "$4" "$5" "$6"
  else
    echo "usage: $0 render-config BLACKLIST_DIR CONFIG ACCESS_LOG HOST PORT" >&2
    echo "normally, source this file and call blacklist_proxy_start [WORK_DIR]" >&2
    exit 2
  fi
fi
