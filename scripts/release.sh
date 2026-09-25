#!/usr/bin/env bash
#
# release.sh — cut an Andai release in one command.
#
# Bumps the version in every manifest (scripts/version.mjs), runs the full
# check suite, commits, creates an annotated tag carrying the release notes,
# and pushes branch + tag atomically. Pushing the tag triggers
# .github/workflows/release.yml, which re-verifies, builds the macOS app for
# Apple Silicon and Intel, and publishes the GitHub Release. The tag always
# equals the version the built Andai.app reports.
#
# Usage:
#   scripts/release.sh                  # bump patch (0.1.4 -> 0.1.5)
#   scripts/release.sh minor            # 0.1.4 -> 0.2.0   (also: major)
#   scripts/release.sh 0.3.0            # explicit version
#   scripts/release.sh 0.3.0 --yes      # no confirmation prompt (agents / CI)
#   scripts/release.sh --dry-run        # show the plan and notes, change nothing
#   scripts/release.sh --watch          # after pushing, follow the release workflow
#
# Flags:
#   -y, --yes          skip the confirmation prompt
#   -n, --dry-run      print the plan; no files changed, nothing pushed
#   -w, --watch        wait for the release workflow and report its result (needs gh)
#       --skip-checks  don't run `npm run check` first (CI still verifies)
#       --allow-dirty  include uncommitted changes in the release commit
#   -h, --help         this text
#
# Safety: refuses to run with uncommitted changes (unless --allow-dirty), when
# the branch is behind origin, or when the tag already exists. Nothing leaves
# the machine until the final confirmation.
#
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "error: not inside a git repository" >&2; exit 1; }
cd "$ROOT"

# --- args -------------------------------------------------------------------
BUMP="patch"; ASSUME_YES=0; DRY_RUN=0; WATCH=0; SKIP_CHECKS=0; ALLOW_DIRTY=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes)       ASSUME_YES=1 ;;
    -n|--dry-run)   DRY_RUN=1 ;;
    -w|--watch)     WATCH=1 ;;
    --skip-checks)  SKIP_CHECKS=1 ;;
    --allow-dirty)  ALLOW_DIRTY=1 ;;
    -h|--help)      sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*)             echo "error: unknown flag: $arg (see --help)" >&2; exit 1 ;;
    *)              BUMP="$arg" ;;
  esac
done

say()  { printf '%s\n' "$*"; }
step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- versions ----------------------------------------------------------------
CUR_VERSION="$(node scripts/version.mjs)" || die "manifests disagree on the current version — fix with: node scripts/version.mjs set <X.Y.Z>"
IFS='.' read -r MA MI PA <<<"$CUR_VERSION"
case "$BUMP" in
  patch) NEW_VERSION="${MA}.${MI}.$((PA + 1))" ;;
  minor) NEW_VERSION="${MA}.$((MI + 1)).0" ;;
  major) NEW_VERSION="$((MA + 1)).0.0" ;;
  *)     NEW_VERSION="${BUMP#v}" ;;
esac
[[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$BUMP' is not patch|minor|major or an X.Y.Z version"
[[ "$NEW_VERSION" != "$CUR_VERSION" ]] || die "new version equals current version ($CUR_VERSION)"
# refuse to go backwards
if [[ "$(printf '%s\n%s\n' "$CUR_VERSION" "$NEW_VERSION" | sort -V | tail -1)" != "$NEW_VERSION" ]]; then
  die "$NEW_VERSION is lower than the current $CUR_VERSION"
fi

TAG="v${NEW_VERSION}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REMOTE_URL="$(git remote get-url origin 2>/dev/null)" || die "no 'origin' remote"
REPO_SLUG="$(sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##' <<<"$REMOTE_URL")"

# --- preflight -----------------------------------------------------------------
step "Preflight"
[[ "$BRANCH" != "HEAD" ]] || die "detached HEAD — check out a branch first"
[[ "$BRANCH" == "main" ]] || say "warning: releasing from '${BRANCH}', not 'main'"

DIRTY="$(git status --porcelain)"
if [[ -n "$DIRTY" && "$ALLOW_DIRTY" -ne 1 ]]; then
  git status --short >&2
  die "working tree has uncommitted changes — commit them first, or pass --allow-dirty to include them"
fi

git fetch --quiet --tags origin || die "could not reach origin"
if git rev-parse -q --verify "origin/${BRANCH}" >/dev/null; then
  BEHIND="$(git rev-list --count "HEAD..origin/${BRANCH}")"
  [[ "$BEHIND" -eq 0 ]] || die "${BRANCH} is ${BEHIND} commit(s) behind origin/${BRANCH} — pull first"
fi
git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null && die "tag ${TAG} already exists locally"
git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1 && die "tag ${TAG} already exists on origin"
say "  ok: $([[ -n "$DIRTY" ]] && echo "uncommitted changes allowed" || echo "clean tree"), in sync with origin/${BRANCH}, ${TAG} is free"

# --- release notes (grouped conventional commits since the last tag) ---------------
PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
RANGE="${PREV_TAG:+${PREV_TAG}..}HEAD"
RELEASE_NOTES="$(git log "$RANGE" --no-merges --format='%s' | awk '
  /^release:/ { next }
  { m = $0 }
  sub(/^feat(\([^)]*\))?!?:[[:space:]]*/, "", m)     { f = f "* " m "\n"; next }
  sub(/^fix(\([^)]*\))?!?:[[:space:]]*/, "", m)      { b = b "* " m "\n"; next }
  sub(/^perf(\([^)]*\))?!?:[[:space:]]*/, "", m)     { p = p "* " m "\n"; next }
  sub(/^refactor(\([^)]*\))?!?:[[:space:]]*/, "", m) { r = r "* " m "\n"; next }
  sub(/^(docs|test|build|ci|chore|style)(\([^)]*\))?!?:[[:space:]]*/, "", m) { o = o "* " m "\n"; next }
  { o = o "* " m "\n" }
  END {
    if (f) printf "### Features\n%s\n", f
    if (b) printf "### Bug Fixes\n%s\n", b
    if (p) printf "### Performance\n%s\n", p
    if (r) printf "### Refactors\n%s\n", r
    if (o) printf "### Other\n%s\n", o
  }')"
[[ -n "$RELEASE_NOTES" ]] || RELEASE_NOTES="No changes since ${PREV_TAG:-the beginning}."

step "Release plan"
say "  version:  ${CUR_VERSION}  ->  ${NEW_VERSION}"
say "  tag:      ${TAG}   (previous: ${PREV_TAG:-none})"
say "  push:     origin ${BRANCH} + ${TAG}  (atomic)"
say "  publish:  https://github.com/${REPO_SLUG}/actions/workflows/release.yml"
[[ -n "$DIRTY" ]] && { say "  includes uncommitted changes:"; git status --short | sed 's/^/    /'; }
step "Release notes"
say "$RELEASE_NOTES"

if [[ "$DRY_RUN" -eq 1 ]]; then
  say ""
  say "[dry-run] nothing changed, nothing pushed. (Checks would run: $([[ $SKIP_CHECKS -eq 1 ]] && echo no || echo 'npm run check').)"
  exit 0
fi

# --- checks (before touching any file) --------------------------------------------
if [[ "$SKIP_CHECKS" -ne 1 ]]; then
  step "npm run check"
  npm run --silent check || die "checks failed — nothing was changed. Fix them, or pass --skip-checks (CI still verifies)."
fi

# --- bump ------------------------------------------------------------------------------
step "Bumping manifests"
node scripts/version.mjs set "$NEW_VERSION"
git --no-pager diff --stat -- package.json package-lock.json src-tauri/tauri.conf.json src-tauri/Cargo.toml src-tauri/Cargo.lock

revert_bump() {
  git checkout -- package.json package-lock.json src-tauri/tauri.conf.json src-tauri/Cargo.toml src-tauri/Cargo.lock
  say "Reverted the version bump; nothing was committed."
}

# --- confirm before anything leaves the machine ---------------------------------------
if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p $'\nCommit, tag '"${TAG}"', and push to origin (this publishes a release)? [y/N] ' reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) revert_bump; exit 1 ;;
  esac
fi

# --- commit, tag, push -------------------------------------------------------------------
step "Commit, tag, push"
git add -A
git commit --quiet -m "release: ${TAG}" -m "${RELEASE_NOTES}"
git tag -a "${TAG}" -m "${TAG}" -m "${RELEASE_NOTES}"
git push --atomic origin "${BRANCH}" "${TAG}"

say ""
say "✔ Pushed ${TAG}."
say "  workflow: https://github.com/${REPO_SLUG}/actions/workflows/release.yml"
say "  release:  https://github.com/${REPO_SLUG}/releases/tag/${TAG}  (appears when the workflow finishes)"

# --- optionally follow the workflow ----------------------------------------------------
if [[ "$WATCH" -eq 1 ]]; then
  command -v gh >/dev/null || die "--watch needs the gh CLI"
  step "Waiting for the release workflow"
  RUN_ID=""
  for _ in $(seq 1 30); do
    RUN_ID="$(gh run list --repo "$REPO_SLUG" --workflow release.yml --branch "$TAG" --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"
    [[ -n "$RUN_ID" ]] && break
    sleep 4
  done
  [[ -n "$RUN_ID" ]] || die "release workflow run for ${TAG} did not appear; check the Actions tab"
  if gh run watch "$RUN_ID" --repo "$REPO_SLUG" --exit-status --interval 20; then
    say ""
    say "✔ Released: https://github.com/${REPO_SLUG}/releases/tag/${TAG}"
    gh release view "$TAG" --repo "$REPO_SLUG" --json assets -q '.assets[].name' | sed 's/^/    /'
  else
    die "release workflow failed — see: gh run view ${RUN_ID} --repo ${REPO_SLUG} --log-failed"
  fi
fi
