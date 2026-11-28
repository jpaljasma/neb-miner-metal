#!/bin/zsh
# Start the Suprnova adapter and open its live dashboard in macOS Terminal.
set -eu
umask 077
sky_args=()
while (( $# )); do
  case "$1" in
    --sky|--sky-threads|--sky-address|--sky-device)
      (( $# >= 2 )) || { print -u2 "Missing value for $1"; exit 1; }
      sky_args+=("$1" "$2")
      shift 2 ;;
    -h|--help)
      print 'start-neb.command [--sky http://HOST:PORT [--sky-threads N] [--sky-address ADDRESS] [--sky-device TEXT]]'
      exit 0 ;;
    *) print -u2 "Unknown option: $1"; exit 1 ;;
  esac
done
root=${0:A:h}
cd "$root"
python=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
[[ -x "$python" ]] || python=$(command -v python3)
[[ -s .neb-payout-address && -s .neb-pool-url && -s build/neb-datasets.bin && -x node/target/release/neb-miner-metal ]] || {
  print -u2 'Missing local payout address, pool URL, dataset, or release miner.'
  exit 1
}
pool=$(<.neb-pool-url)
worker=neb-miner-metal
mkdir -p build
zmodload zsh/system
: >> build/metal-launch.lock
zsystem flock -t 15 build/metal-launch.lock || { print -u2 'Another launch is in progress.'; exit 1; }

terminal() {
  /usr/bin/osascript - "$@" <<'APPLESCRIPT' >/dev/null
on run argv
    set windowTitle to item 1 of argv
    set cmd to "exec"
    repeat with arg in items 2 thru -1 of argv
        set cmd to cmd & " " & quoted form of (contents of arg)
    end repeat
    tell application "Terminal"
        set newTab to do script cmd
        set custom title of newTab to windowTitle
        activate
    end tell
end run
APPLESCRIPT
}

# An existing supervisor owns the writable log even while its GPU child reconnects.
# Only attach to this checkout and the configured pool.
log=''
for pid in ${(f)"$(/usr/bin/pgrep -U "$EUID" -f '[/ ]test_metal_stratum[.]py[[:space:]]' || true)"}; do
  [[ -n "$pid" ]] || continue
  active=$(/usr/sbin/lsof -nP -a -p "$pid" -Ffan 2>/dev/null | /usr/bin/awk -v dir="$root/build/" '
    /^f/ {w=0} /^a/ {w=($0=="aw" || $0=="au")}
    /^n/ && w && index(substr($0,2),dir)==1 && /[.]jsonl$/ {print substr($0,2); exit}')
  [[ -n "$active" ]] || continue
  cmd=$(/bin/ps -ww -p "$pid" -o command=)
  if [[ " $cmd " != *" --pool $pool "* ]]; then
    print -u2 'A session is already running on another pool or port. Stop that session before launching this one.'
    exit 1
  fi
  if (( ${#sky_args} )); then
    print -u2 'A worker is already running. Stop it before launching with sky options, or omit the options to open its dashboard.'
    exit 1
  fi
  log=$active
  break
done

if [[ -z "$log" ]]; then
  dir=$(/usr/bin/mktemp -d "$root/build/pool-XXXXXXXX")
  log="$dir/events.jsonl"
  terminal "NEB · Pool worker" /usr/bin/caffeinate -i "$python" -u "$root/tools/test_metal_stratum.py" \
    --pool "$pool" --worker "$worker" --selective --tess-ahead --duration 86400 \
    --pause-file "$dir/PAUSE" --log "$log" "${sky_args[@]}"
  for attempt in {1..100}; do
    [[ -s "$log" ]] && break
    sleep 0.1
  done
  [[ -s "$log" ]] || { print -u2 'The miner did not create its log. Check the miner window.'; exit 1; }
fi
terminal "NEB · Kepler and TESS" "$python" "$root/tools/metal_tui.py" "$log"
