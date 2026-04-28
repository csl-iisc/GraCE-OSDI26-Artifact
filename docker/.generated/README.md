# Generated Docker run metadata

This directory is intentionally empty in the repository.

The host tuning scripts may optionally write `docker_cpuset.env` here so that
`docker/run_cuda124.sh` and `docker/run_cuda128.sh` can automatically discover
Docker cpuset settings without reading `/run/grace_docker_cpuset_backup/state.env`.

Expected generated file shape:

```bash
BENCH_CPUS='8-31,64-95'
BENCH_MEMS='0'
BENCH_SLICE='bench.slice'
CGROUP_PARENT='bench.slice'
USE_NUMACTL='0'
```
