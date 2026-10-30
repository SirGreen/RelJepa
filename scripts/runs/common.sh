# Shared helpers for the job scripts (sourced). Runs inside the container.
# set_commit: record the repository commit at the start of EACH job, marked
# "-dirty" if code, configs or scripts have uncommitted changes (paper/docs edits do not count).
set_commit() {
  local g="git -c safe.directory=/workspace -C /workspace" c
  c="$($g rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$($g status --porcelain --untracked-files=no -- third_party src configs scripts tests docker 2>/dev/null)" ]]; then
    c="${c}-dirty"
  fi
  export GIT_COMMIT="$c"
}
# sweep_failed LOG: TVE's sweep runner (train_baseline.py -s) catches exceptions and still exits 0,
# so a crashed sweep must be detected from its log.
sweep_failed() { grep -q "An error occurred" "$1" 2>/dev/null; }
# sweep_ready CONFIG: the sweep runner sets the learning rate only through the {{ lr }} Jinja variable
# and silently keeps a hard-coded value, so never start a sweep on a config without it.
sweep_ready() { grep -q "{{ lr }}" "$1" 2>/dev/null; }
