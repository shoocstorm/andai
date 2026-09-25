// Simulated data for the Workflows screens. Tool routing and workflow
// execution are mocked in this build; Knowledge Search is the only live tool.

export type Approval = {
  id: string;
  tool: string;
  icon: 'mail' | 'bank' | 'db';
  severity: 'critical' | 'review';
  body: string;
  target?: string;
  tail?: string;
};

export const APPROVALS: Approval[] = [
  {
    id: 'ap-1',
    tool: 'EMAIL_DISPATCH',
    icon: 'mail',
    severity: 'critical',
    body: 'Requesting authority to send encrypted summary to',
    target: 'executive_board@corp.net',
    tail: 'Payload size: 4.2MB.',
  },
  {
    id: 'ap-2',
    tool: 'WIRE_TRANSFER',
    icon: 'bank',
    severity: 'review',
    body: 'Validation required for API key rotation on secondary ledger.',
  },
  {
    id: 'ap-3',
    tool: 'SQL_WRITE',
    icon: 'db',
    severity: 'review',
    body: 'Agent proposes UPDATE on 1,204 rows in `customer_segments`. Dry-run diff attached.',
  },
];

export type ActiveWorkflow = {
  id: string;
  name: string;
  body: string;
  status: 'nominal' | 'idle' | 'running';
  icon: 'network' | 'loop' | 'digest';
};

export const WORKFLOWS: ActiveWorkflow[] = [
  {
    id: 'wf-market',
    name: 'Market Research Flow',
    body: 'Autonomous synthesis of global market trends via 12 source vectors.',
    status: 'nominal',
    icon: 'network',
  },
  {
    id: 'wf-support',
    name: 'Auto-Support Loop',
    body: 'Continuous monitoring of inbound triage and sentiment classification.',
    status: 'idle',
    icon: 'loop',
  },
];

export type Tool = {
  id: string;
  name: string;
  body: string;
  icon: 'globe' | 'chart' | 'mail' | 'db' | 'sync' | 'graph';
  accent: string;
  enabled: boolean;
  live?: boolean;
};

export const TOOLS: Tool[] = [
  { id: 'kb', name: 'Knowledge Search', body: 'GraphRAG over your ug knowledge graphs. Runs on-device.', icon: 'graph', accent: 'var(--violet)', enabled: true, live: true },
  { id: 'web', name: 'Web Search', body: 'Real-time crawling and indexing of public web domains.', icon: 'globe', accent: 'var(--blue)', enabled: true },
  { id: 'data', name: 'Data Analysis', body: 'Statistical modeling and anomaly detection on JSON datasets.', icon: 'chart', accent: 'var(--violet)', enabled: true },
  { id: 'mail', name: 'Email Dispatcher', body: 'Secure SMTP/IMAP protocol for automated communication.', icon: 'mail', accent: 'var(--amber)', enabled: true },
  { id: 'sql', name: 'SQL Connector', body: 'Relational database bridge for enterprise data retrieval.', icon: 'db', accent: 'var(--text-4)', enabled: false },
  { id: 's3', name: 'S3 Sync', body: 'Direct integration with S3 buckets for large file handling.', icon: 'sync', accent: 'var(--blue)', enabled: true },
];

export type NodeKind = 'trigger' | 'skill' | 'rag' | 'approval' | 'llm' | 'upload';
export type FlowNode = {
  id: string;
  hex: string;
  kind: NodeKind;
  title: string;
  x: number;
  y: number;
};

export const NODE_W = 300;
export const NODE_H = 150;

export const FLOW_NODES: FlowNode[] = [
  { id: 'n1', hex: '0x442', kind: 'trigger', title: 'Schedule Trigger', x: 0, y: 160 },
  { id: 'n2', hex: '0x881', kind: 'skill', title: 'Extract Data', x: 390, y: 160 },
  { id: 'n3', hex: '0x3A7', kind: 'rag', title: 'Knowledge Search', x: 780, y: 20 },
  { id: 'n4', hex: '0x219', kind: 'approval', title: 'Human Approval', x: 780, y: 300 },
  { id: 'n5', hex: '0x5C1', kind: 'llm', title: 'Summarize', x: 1170, y: 160 },
  { id: 'n6', hex: '0x7F0', kind: 'upload', title: 'Upload to S3', x: 1560, y: 160 },
];

export const FLOW_EDGES: [string, string][] = [
  ['n1', 'n2'],
  ['n2', 'n3'],
  ['n2', 'n4'],
  ['n3', 'n5'],
  ['n4', 'n5'],
  ['n5', 'n6'],
];

export const RUN_SCRIPT: Record<string, { working: string; done: string; ms: number }> = {
  n1: { working: 'Firing cron 0 9 * * MON…', done: 'Fired · weekly digest', ms: 700 },
  n2: { working: 'Parsing JSON payload…', done: 'Parsed 4.2 MB · 1,904 records', ms: 2200 },
  n3: { working: 'Querying knowledge graph…', done: '8 passages · 41 ms', ms: 1400 },
  n4: { working: 'Waiting for @admin signature…', done: 'Approved by @admin_core', ms: 0 },
  n5: { working: 'Generating summary…', done: '612 tokens · 29 tok/s', ms: 2600 },
  n6: { working: 'Uploading digest.pdf…', done: 's3://andai-digests/2026-09-25.pdf', ms: 1500 },
};
