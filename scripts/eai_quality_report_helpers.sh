#!/usr/bin/env bash
# 采集脚本引用：解析 validate_bag.py 并输出质量页约定的单行：
#   EAI_VALIDATION_REPORT_JSON:{...}
# source 示例：
#   _H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eai_quality_report_helpers.sh"
#   [ -f "$_H" ] && . "$_H"

eai_resolve_validate_bag_py() {
  local repo_root_hint="${1:-}"
  local candidates=(
    "${EAI_VALIDATE_BAG_PY:-}"
    "${repo_root_hint}/agent/validate_bag.py"
    "${repo_root_hint}/validate_bag.py"
    "/opt/eai-agent/agent/validate_bag.py"
    "/opt/eai-agent/validate_bag.py"
    "${HOME}/eai-agent/agent/validate_bag.py"
    "${HOME}/eai-agent/validate_bag.py"
  )
  local c
  for c in "${candidates[@]}"; do
    [ -n "${c:-}" ] || continue
    [ -f "$c" ] || continue
    printf '%s' "$c"
    return 0
  done
  return 1
}

# 若为目录且其下仅有 1 个 *.mcap / *.db3，则用该文件
eai_pick_bag_target_for_validate() {
  local raw="${1:?}"
  if [ -f "$raw" ]; then
    printf '%s' "$raw"
    return 0
  fi
  if [ ! -d "$raw" ]; then
    printf '%s' "$raw"
    return 0
  fi

  local n
  n="$(find "$raw" -maxdepth 1 \( -name '*.mcap' -o -name '*.db3' \) -type f 2>/dev/null | wc -l)"
  n="$(echo "$n" | tr -d ' ')"
  if [ "${n:-0}" -eq 1 ]; then
    find "$raw" -maxdepth 1 \( -name '*.mcap' -o -name '*.db3' \) -type f 2>/dev/null | head -n 1
    return 0
  fi

  printf '%s' "$raw"
}

eai_emit_validation_report() {
  local bag_raw="${1:?}"
  local duration_sec="${2:-30}"
  local repo_root_hint="${3:-}"

  command -v python3 >/dev/null 2>&1 || return 1

  local vb
  vb="$(eai_resolve_validate_bag_py "$repo_root_hint")" || return 1

  local bag_abs="$bag_raw"
  if [ -e "$bag_abs" ]; then
    bag_abs="$(cd "$(dirname "$bag_abs")" && pwd)/$(basename "$bag_abs")"
  fi
  local bag_use
  bag_use="$(eai_pick_bag_target_for_validate "$bag_abs")"

  export PYTHONIOENCODING=utf-8
  local tmp_out tmp_err
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  set +e
  python3 -u "$vb" "$bag_use" "${duration_sec}" raw >"$tmp_out" 2>"$tmp_err"
  local ec=$?
  set -e

  local report_line
  report_line="$(tail -n 1 "$tmp_out" | tr -d '\r')"
  rm -f "$tmp_out"

  if [ "$ec" -ne 0 ] || [ -z "$report_line" ]; then
    rm -f "$tmp_err"
    return 1
  fi
  rm -f "$tmp_err"

  case "$report_line" in
  \{*) ;;
  *)
    return 1
    ;;
  esac

  printf '%s\n' "EAI_VALIDATION_REPORT_JSON:${report_line}"
  return 0
}
