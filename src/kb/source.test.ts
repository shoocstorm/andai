import { describe, expect, it } from 'vitest';
import { pdfPages, readStructure } from './source';

// Shape of `ug file_context file:operations.md --json` (ug 0.1.22), trimmed.
const report = {
  files: [
    {
      file: 'operations.md',
      facts: { language: 'markdown', classification: 'documentation', lines: 25, is_test: false, symbols: 3 },
      items: [
        { role: 'outline', id: 'heading:operations.md:Fleet', name: 'Fleet', node_type: 'Concept', start_line: 3, end_line: 9, doc: '| Vessel |\n  Passengers |' },
        { role: 'outline', id: 'heading:operations.md:Handbook', name: 'Handbook', node_type: 'Concept', start_line: 1, end_line: 25 },
        { role: 'outline', id: 'heading:operations.md:Kestrel', name: 'Kestrel', node_type: 'Concept', start_line: 5, end_line: 6 },
        { role: 'outline', id: 'heading:operations.md:Routes', name: 'Routes', node_type: 'Concept', start_line: 10, end_line: 14 },
        { role: 'import', why: 'this file imports it', name: 'fares.ts', file: 'fares.ts' },
        { role: 'sibling', name: 'refund-policy.md', file: 'refund-policy.md' },
        { role: 'sibling', name: 'refund-policy.md', file: 'refund-policy.md' },
      ],
    },
  ],
};

describe('readStructure', () => {
  it('reads the facts and nests the outline by line containment, in line order', () => {
    const s = readStructure(report);
    expect(s).toMatchObject({ language: 'markdown', classification: 'documentation', lines: 25, symbols: 3, isTest: false });
    expect(s.outline.map((i) => [i.name, i.depth])).toEqual([
      ['Handbook', 0],
      ['Fleet', 1],
      ['Kestrel', 2],
      ['Routes', 1],
    ]);
    expect(s.outline[1].doc).toBe('| Vessel | Passengers |');
  });

  it('groups related files by role in ug’s order, without repeats', () => {
    expect(readStructure(report).related).toEqual([
      { role: 'import', label: 'Imports', items: [{ name: 'fares.ts', file: 'fares.ts', why: 'this file imports it' }] },
      { role: 'sibling', label: 'Same folder', items: [{ name: 'refund-policy.md', file: 'refund-policy.md', why: '' }] },
    ]);
  });

  it('tolerates any shape', () => {
    for (const junk of [null, 'text', 42, [], { files: 'x' }, { files: [{ items: [null, 3, { role: 'outline' }] }] }]) {
      expect(readStructure(junk)).toEqual({ language: null, classification: null, lines: null, symbols: null, isTest: false, outline: [], related: [] });
    }
  });
});

describe('pdfPages', () => {
  it('keeps the outline items that carry text', () => {
    const s = readStructure({
      files: [{ items: [
        { role: 'outline', name: 'p.1 · Intro', node_type: 'Concept', start_line: 1, end_line: 1, doc: 'Hello PDF' },
        { role: 'outline', name: 'p.2', node_type: 'Concept', start_line: 2, end_line: 2 },
      ] }],
    });
    expect(pdfPages(s.outline)).toEqual([{ title: 'p.1 · Intro', text: 'Hello PDF' }]);
  });
});
