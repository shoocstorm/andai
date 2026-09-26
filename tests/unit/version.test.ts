// All four manifests must carry the same version (the release workflow
// refuses a tag that disagrees with them). Fix with: bun run version set X.Y.Z
import { describe, expect, it } from 'vitest';
// @ts-expect-error — plain ESM script without type declarations
import { readVersions } from '../../scripts/version.mjs';

describe('app version', () => {
  it('is identical in every manifest', () => {
    const versions: Record<string, string | undefined> = readVersions();
    expect(Object.keys(versions)).toHaveLength(4);
    expect(new Set(Object.values(versions)).size, JSON.stringify(versions, null, 2)).toBe(1);
  });
  it('is plain semver X.Y.Z', () => {
    expect(readVersions()['package.json']).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
