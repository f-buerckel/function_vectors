# Shut Windows down 5 minutes after the queue finishes, or after 8 hours at the latest (in case a run hangs).
start=$(date +%s)
while [ ! -f ALL_DONE ] && [ $(( $(date +%s) - start )) -lt 28800 ]; do sleep 60; done
echo "$(date): shutting down (ALL_DONE: $(test -f ALL_DONE && echo yes || echo no))" >> shutdown_watch.log
/mnt/c/Windows/System32/shutdown.exe /s /t 300 /c "Research runs finished"
