#!/usr/bin/env bash
# migrate_subprojects.sh
# Splits each subproject into its own GitHub repo, then re-adds as a submodule.
# Preserves full git history for each subproject.
#
# Prerequisites:
#   - gh auth login (already done)
#   - Run from the root of the deepMRI repo

set -euo pipefail

GITHUB_USER="sunhongfu"
DEEPMRI_DIR="$(pwd)"
WORK_DIR="/tmp/deepmri_migration"

# Subprojects: folder name = new repo name
SUBPROJECTS=(
  "iQSM"
  "iQSM_Plus"
  "xQSM"
  "DCRNet"
  "BFRnet"
  "AFTER-QSM"
  "MoDIP"
  "DIP-UP"
)

# ─── CHECKS ──────────────────────────────────────────────────────────────────
if [[ "$(basename "$DEEPMRI_DIR")" != "deepMRI" ]]; then
  echo "ERROR: run this script from the root of the deepMRI repo."
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean. Commit or stash changes first."
  exit 1
fi

mkdir -p "$WORK_DIR"

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
for FOLDER in "${SUBPROJECTS[@]}"; do
  echo ""
  echo "════════════════════════════════════════"
  echo "Processing: ${FOLDER}"
  echo "════════════════════════════════════════"

  # 1. Create the GitHub repo (skip if already exists)
  echo "  → Creating github.com/${GITHUB_USER}/${FOLDER}..."
  gh repo create "${GITHUB_USER}/${FOLDER}" --public --description "Part of the deepMRI project" || echo "  ! Repo may already exist, continuing..."

  # 2. Split the subfolder history into a temporary branch
  echo "  → Splitting git history for ${FOLDER}..."
  git subtree split --prefix="${FOLDER}" --branch "split/${FOLDER}"
  echo "  ✓ History split."

  # 3. Push split branch to new standalone repo
  CLONE_DIR="${WORK_DIR}/${FOLDER}"
  rm -rf "$CLONE_DIR"
  git clone --branch "split/${FOLDER}" "$DEEPMRI_DIR" "$CLONE_DIR"
  pushd "$CLONE_DIR" > /dev/null
  git remote add upstream "https://github.com/${GITHUB_USER}/${FOLDER}.git"
  echo "  → Pushing to github.com/${GITHUB_USER}/${FOLDER}..."
  git push upstream HEAD:master
  echo "  ✓ Pushed."
  popd > /dev/null

  # 4. Remove folder from deepMRI and re-add as submodule
  echo "  → Converting to submodule..."
  git rm -r "${FOLDER}"
  git submodule add "https://github.com/${GITHUB_USER}/${FOLDER}.git" "${FOLDER}"
  git commit -m "Convert ${FOLDER} to submodule → github.com/${GITHUB_USER}/${FOLDER}"
  echo "  ✓ Done: ${FOLDER}"

  # 5. Clean up temp branch
  git branch -D "split/${FOLDER}"
done

rm -rf "$WORK_DIR"

echo ""
echo "════════════════════════════════════════"
echo "All subprojects migrated!"
echo "Review, then push:"
echo "  git push origin master"
echo "════════════════════════════════════════"
