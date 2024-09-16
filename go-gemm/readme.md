```bash
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses \
	go run main.go -K 4096 -M 2048 -N 3072 -type matMultSync \
	> _profiler_logs/matMultSync.log 2>&1
```

```go
go run main.go --help
```
