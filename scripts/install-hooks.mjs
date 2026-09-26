// Points git at .githooks/ so the pre-push check runs before anything reaches
// origin (AGENTS.md → Git). Idempotent, and never overrides a hooksPath the
// developer set on purpose (e.g. another tool's hooks).
import { execSync } from 'node:child_process';

const WANTED = '.githooks';

const current = (() => {
  try {
    return execSync('git config --get core.hooksPath', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return ''; // unset
  }
})();

if (current === WANTED) process.exit(0);
if (current) {
  console.warn(`hooks: core.hooksPath is already "${current}"; not overriding. Run manually if wanted: git config core.hooksPath .githooks`);
  process.exit(0);
}

try {
  execSync(`git config core.hooksPath ${WANTED}`, { stdio: 'ignore' });
  console.log('hooks: pre-push check installed (git config core.hooksPath .githooks)');
} catch {
  console.warn('hooks: no git repository here, skipped');
}
