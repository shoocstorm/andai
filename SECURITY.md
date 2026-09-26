# Security

Andai is a local-first app: documents, chats and models stay on your Mac. The
only network traffic is a model download you start, from Hugging Face.

## How that is enforced

- **Egress is blocked by policy.** A content security policy lets the app
  reach only itself and Hugging Face. The end-to-end test proves it with a
  local canary server that must receive zero requests.
- **Model output is inert.** Answers never load images, and links can only be
  copied. A document crafted to manipulate the model can't use them to send
  data out.
- **The app shell is the trust boundary.** Rust validates every request from
  the UI. It only reads files you dropped or picked, and it neutralizes
  arguments to the `ug` CLI that could be parsed as flags.
- **Andai only loads a UI it serves itself.** If another process holds its
  local port (14230), Andai shows an error instead of starting.
- **Models are verified.** Downloads come from a fixed Hugging Face commit and
  must match a recorded sha256 before they load.
- **Documents are data, not instructions.** Retrieved passages reach the model
  fenced and marked untrusted.
- **Dependencies are audited.** `bun audit` and `cargo audit` run in CI, and a
  release can't ship with a known vulnerability.

The full user-facing explanation, including what isn't protected yet, is in
[docs/security.md](docs/security.md). The threat model and the rules
contributors follow are in [AGENTS.md §9](AGENTS.md#9-security).

## Reporting a vulnerability

Please report it privately through GitHub's **Security → Report a
vulnerability** on this repository. Don't open a public issue. Include steps
to reproduce and the Andai version (select Andai.app in Finder, then
**File → Get Info**). We'll acknowledge the
report, and credit you in the release notes unless you'd rather not be named.
