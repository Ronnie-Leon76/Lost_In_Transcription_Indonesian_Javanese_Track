#!/usr/bin/env bash
# Export every LIT_* / HF_* variable this repo reads, then check the development
# set is actually on disk.
#
# Must be sourced, or the exports die with the process:
#
#   source scripts/setup_env.sh
#   . scripts/setup_env.sh
#
# Order of precedence, highest first:
#   1. variables already in the environment (RunPod secrets, a prior export)
#   2. repo-root .env
#   3. defaults below
#
# HF_TOKEN is never printed.

if [ -n "${ZSH_VERSION:-}" ]; then
  _LIT_THIS="${(%):-%x}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
  _LIT_THIS="${BASH_SOURCE[0]}"
else
  echo "setup_env.sh: source this file from bash or zsh" >&2
  return 1 2>/dev/null || exit 1
fi

# Executed rather than sourced: the exports would be discarded.
if [ "${0##*/}" = "setup_env.sh" ] && [ -z "${ZSH_EVAL_CONTEXT:-}" ]; then
  if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "source this file — do not execute it:" >&2
    echo "  source scripts/setup_env.sh" >&2
    exit 1
  fi
fi

_LIT_ROOT="$(cd "$(dirname "$_LIT_THIS")/.." && pwd)"
unset _LIT_THIS

_lit_is_set() {
  eval "[ -n \"\${$1+x}\" ]"
}

_lit_strip() {
  # trim whitespace and one layer of matching quotes
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [ "${#value}" -ge 2 ]; then
    local first="${value%"${value#?}"}"
    local last="${value#"${value%?}"}"
    if [ "$first" = "$last" ] && { [ "$first" = "'" ] || [ "$first" = '"' ]; }; then
      value="${value#?}"
      value="${value%?}"
    fi
  fi
  printf '%s' "$value"
}

_lit_load_dotenv() {
  local file="$1"
  [ -f "$file" ] || return 0
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|\#*) continue ;;
    esac
    case "$line" in
      *=*) ;;
      *) continue ;;
    esac
    key="$(_lit_strip "${line%%=*}")"
    value="$(_lit_strip "${line#*=}")"
    case "$key" in
      ''|\#*) continue ;;
    esac
    if _lit_is_set "$key"; then
      continue
    fi
    export "$key=$value"
  done < "$file"
}

_lit_default() {
  local key="$1"
  local value="$2"
  if ! _lit_is_set "$key"; then
    export "$key=$value"
  fi
}

_lit_load_dotenv "$_LIT_ROOT/.env"

_lit_default LIT_DEV_DIR "$_LIT_ROOT/indonesian_dev"
_lit_default LIT_JEMBER "$_LIT_ROOT/jember"
_lit_default LIT_NAHUATL_DIR "$_LIT_ROOT/nahuatl_dev"
_lit_default LIT_SEED 3407
_lit_default LIT_DETERMINISTIC 0
_lit_default LIT_MODEL "Qwen/Qwen3-ASR-1.7B"
_lit_default LIT_BASE_MODEL "Qwen/Qwen3-ASR-1.7B"
_lit_default LIT_HF_PRIVATE 1
_lit_default LIT_HF_UPLOAD_MERGED 0
_lit_default LIT_SKIP_DECODE 0
_lit_default LIT_SKIP_TRAIN 0
_lit_default LIT_TEST_CLIPS 1000

# Optional keys are left unset unless .env or the shell provided them:
#   HF_TOKEN, LIT_HF_REPO, LIT_HF_CHECKPOINT, LIT_HF_SUBFOLDER,
#   LIT_HF_REVISION, LIT_CLIPS_PER_S
# Container-only keys are never defaulted here:
#   LIT_DATA_DIR, LIT_SUBMISSION_PATH, LIT_CONFIG_PATH

_LIT_META="$LIT_DEV_DIR/metadata.tsv"
_LIT_CLIPS="$LIT_DEV_DIR/clips"

echo "LIT_ROOT              $_LIT_ROOT"
echo "LIT_DEV_DIR           $LIT_DEV_DIR"
echo "LIT_JEMBER            $LIT_JEMBER"
echo "LIT_NAHUATL_DIR       ${LIT_NAHUATL_DIR:-<unset>}"
echo "MDC_API_KEY           $(_lit_is_set MDC_API_KEY && echo present || echo absent)"
echo "LIT_SEED              $LIT_SEED"
echo "LIT_MODEL             $LIT_MODEL"
echo "LIT_BASE_MODEL        $LIT_BASE_MODEL"
echo "LIT_HF_REPO           ${LIT_HF_REPO:-<unset>}"
echo "HF_TOKEN              $(_lit_is_set HF_TOKEN && echo present || echo absent)"
echo "LIT_SKIP_DECODE/TRAIN $LIT_SKIP_DECODE / $LIT_SKIP_TRAIN"

if [ ! -f "$_LIT_META" ]; then
  echo >&2
  echo "setup_env.sh: development set not found at $_LIT_META" >&2
  echo "indonesian_dev/ must contain metadata.tsv and clips/." >&2
  echo >&2
  echo "On a private repo the folder is meant to be tracked. From your laptop:" >&2
  echo "  git add indonesian_dev" >&2
  echo "  git commit -m \"Add development set\"" >&2
  echo "  git push" >&2
  echo "then on this box: git pull" >&2
  echo >&2
  echo "Or copy it once:" >&2
  echo "  rsync -avP indonesian_dev/ <host>:$_LIT_ROOT/indonesian_dev/" >&2
  unset _LIT_ROOT _LIT_META _LIT_CLIPS
  unset -f _lit_is_set _lit_strip _lit_load_dotenv _lit_default
  return 1 2>/dev/null || exit 1
fi

_LIT_N_CLIPS=0
if [ -d "$_LIT_CLIPS" ]; then
  _LIT_N_CLIPS="$(find "$_LIT_CLIPS" -type f \( -name '*.mp3' -o -name '*.wav' -o -name '*.flac' \) | wc -l | tr -d ' ')"
fi
echo "dev metadata          $_LIT_META"
echo "dev clips             ${_LIT_N_CLIPS} files"

if [ "$_LIT_N_CLIPS" -eq 0 ]; then
  echo "setup_env.sh: metadata.tsv is present but clips/ is empty" >&2
  unset _LIT_ROOT _LIT_META _LIT_CLIPS _LIT_N_CLIPS
  unset -f _lit_is_set _lit_strip _lit_load_dotenv _lit_default
  return 1 2>/dev/null || exit 1
fi

unset _LIT_ROOT _LIT_META _LIT_CLIPS _LIT_N_CLIPS
unset -f _lit_is_set _lit_strip _lit_load_dotenv _lit_default
