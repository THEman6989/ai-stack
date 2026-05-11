#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hermes_dir="${HERMES_PATCH_TARGET_DIR:-${repo_root}/hermes-agent}"

if [[ ! -f "${hermes_dir}/hermes_cli/kanban_db.py" ]]; then
  echo "Hermes target is missing hermes_cli/kanban_db.py: ${hermes_dir}" >&2
  exit 1
fi

patch_file="${repo_root}/patches/hermes-agent/kanban-db-duplicate-column-guard.patch"
target_file="${hermes_dir}/hermes_cli/kanban_db.py"

if grep -q "duplicate column name" "${target_file}"; then
  echo "Hermes kanban duplicate-column patch already present in ${target_file}."
elif git -C "${hermes_dir}" apply --check "${patch_file}" 2>/dev/null; then
  git -C "${hermes_dir}" apply "${patch_file}"
  echo "Applied Hermes kanban duplicate-column patch to ${hermes_dir}."
else
  echo "Hermes kanban duplicate-column patch is not applicable to ${hermes_dir}." >&2
  echo "Review ${patch_file} against ${target_file} before starting Hermes." >&2
  exit 1
fi

echo "Applied AlphaRavis Hermes-Agent local patches."
