#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hermes_dir="${repo_root}/hermes-agent"

if [[ ! -d "${hermes_dir}/.git" && ! -f "${hermes_dir}/.git" ]]; then
  echo "hermes-agent submodule is missing: ${hermes_dir}" >&2
  exit 1
fi

git -C "${hermes_dir}" apply --check "${repo_root}/patches/hermes-agent/kanban-db-duplicate-column-guard.patch" \
  && git -C "${hermes_dir}" apply "${repo_root}/patches/hermes-agent/kanban-db-duplicate-column-guard.patch" \
  || echo "Hermes kanban duplicate-column patch already applied or not applicable."

if [[ -d "${hermes_dir}/tinker-atropos" ]]; then
  git -C "${hermes_dir}/tinker-atropos" fetch origin ac3a1650e33ab5bfeee2df32f511120a259d667f
  git -C "${hermes_dir}/tinker-atropos" checkout ac3a1650e33ab5bfeee2df32f511120a259d667f
  git -C "${hermes_dir}" add tinker-atropos
fi

echo "Applied AlphaRavis Hermes-Agent local patches."
