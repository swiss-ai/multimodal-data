# Squid Blacklist

Builds public threat-intelligence feeds into ACL files for a local, non-caching Squid proxy.

## Use in a Slurm job

```bash
source /capstor/store/cscs/swissai/infra01/users/tchu/blacklist/proxy.sh
blacklist_proxy_start

srun python download.py
```

The proxy runs in the background, exports the standard HTTP(S) proxy variables,
and stops when the job shell exits. Set `BLACKLIST_SQUID_BIN` to override the
Squid executable.

## Build

```bash
# generate blacklist files (download live threat feeds)
./main.py build

# build and share on infra01
./deploy.sh /capstor/store/cscs/swissai/infra01/users/tchu/blacklist
```
