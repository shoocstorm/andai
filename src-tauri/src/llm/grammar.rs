//! GBNF grammars for the native engine, the way wllama's llama.cpp honors them.
//!
//! - A choice between literals, `root ::= "A" | "B" | "C"` (what decisions
//!   send, llm/decide.ts), is read as a list: engine.rs restricts the reply
//!   to those strings and reads the letters' log-probabilities.
//! - Any other grammar without recursion (argument filling's JSON objects,
//!   agent/tools/validate.ts `schemaGrammar`) is compiled to one anchored
//!   regex and a lazy DFA (regex-automata, already in the tree through
//!   Tauri). At each step only tokens whose bytes keep the DFA alive may be
//!   picked, and the end only where it matches.
//!
//! The grammar matters beyond validity: without it Qwen3 1.7B wrote argument
//! JSON its own way and, in that context, chose `"scope": "broad"` for every
//! search where the grammar-held reply chose `"focused"`, and the answers got
//! worse (agent eval, 2026-09-27; AGENTS.md §2).

use regex_automata::hybrid::dfa::{Cache, DFA};
use regex_automata::hybrid::LazyStateID;
use regex_automata::util::start;
use regex_automata::{Anchored, MatchKind};
use std::collections::HashMap;

/// The alternatives of a `root ::= "a" | "b"` grammar, or `None` for any
/// other grammar. Literals may use `\"` and `\\` escapes.
pub fn choices(grammar: &str) -> Option<Vec<String>> {
    let body = grammar.trim().strip_prefix("root")?.trim_start().strip_prefix("::=")?;
    let mut out = Vec::new();
    let mut chars = body.trim().chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next()? != '"' {
            return None;
        }
        let mut lit = String::new();
        loop {
            match chars.next()? {
                '"' => break,
                '\\' => match chars.next()? {
                    c @ ('"' | '\\') => lit.push(c),
                    _ => return None,
                },
                '\n' => return None,
                c => lit.push(c),
            }
        }
        if lit.is_empty() {
            return None;
        }
        out.push(lit);
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        match chars.next() {
            None => return Some(out),
            Some('|') => continue,
            Some(_) => return None,
        }
    }
}

// ── GBNF → regex ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Alt(Vec<Expr>),
    Seq(Vec<Expr>),
    Lit(String),
    /// Negated?, then inclusive char ranges.
    Class(bool, Vec<(char, char)>),
    Ref(String),
    Repeat(Box<Expr>, u32, Option<u32>),
}

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl Parser<'_> {
    /// Skips spaces, newlines and `#` comments between items.
    fn skip(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
            } else if c == '#' {
                while self.chars.next().is_some_and(|c| c != '\n') {}
            } else {
                break;
            }
        }
    }

    fn escape(&mut self) -> Result<char, String> {
        let hex = |p: &mut Self, n: usize| -> Result<char, String> {
            let digits: String = (0..n).filter_map(|_| p.chars.next()).collect();
            u32::from_str_radix(&digits, 16).ok().and_then(char::from_u32).ok_or_else(|| format!("bad escape \\x{digits}"))
        };
        match self.chars.next().ok_or("grammar ends in an escape")? {
            'n' => Ok('\n'),
            't' => Ok('\t'),
            'r' => Ok('\r'),
            'x' => hex(self, 2),
            'u' => hex(self, 4),
            'U' => hex(self, 8),
            c => Ok(c),
        }
    }

    fn literal(&mut self) -> Result<Expr, String> {
        let mut s = String::new();
        loop {
            match self.chars.next().ok_or("unterminated string in grammar")? {
                '"' => return Ok(Expr::Lit(s)),
                '\\' => s.push(self.escape()?),
                c => s.push(c),
            }
        }
    }

    fn class(&mut self) -> Result<Expr, String> {
        let negated = self.chars.peek() == Some(&'^');
        if negated {
            self.chars.next();
        }
        let mut items = Vec::new();
        loop {
            let lo = match self.chars.next().ok_or("unterminated character class in grammar")? {
                ']' => return Ok(Expr::Class(negated, items)),
                '\\' => self.escape()?,
                c => c,
            };
            let mut hi = lo;
            if self.chars.peek() == Some(&'-') {
                self.chars.next();
                if self.chars.peek() == Some(&']') {
                    // `[a-]`: a literal dash at the end
                    items.extend([(lo, lo), ('-', '-')]);
                    continue;
                }
                hi = match self.chars.next().ok_or("unterminated character class in grammar")? {
                    '\\' => self.escape()?,
                    c => c,
                };
            }
            if hi < lo {
                return Err(format!("bad range {lo:?}-{hi:?} in grammar"));
            }
            items.push((lo, hi));
        }
    }

    fn number(&mut self) -> Result<u32, String> {
        let mut n = String::new();
        while let Some(&c) = self.chars.peek().filter(|c| c.is_ascii_digit()) {
            n.push(c);
            self.chars.next();
        }
        n.parse().map_err(|_| "expected a number in a {m,n} repeat".to_string())
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.chars.next().ok_or("expected an item in grammar")? {
            '"' => self.literal(),
            '[' => self.class(),
            '(' => {
                let e = self.alt()?;
                self.skip();
                if self.chars.next() != Some(')') {
                    return Err("unclosed ( in grammar".into());
                }
                Ok(e)
            }
            c if c.is_ascii_alphabetic() => {
                let mut name = c.to_string();
                while let Some(&c) = self.chars.peek().filter(|c| c.is_ascii_alphanumeric() || **c == '-' || **c == '_') {
                    name.push(c);
                    self.chars.next();
                }
                Ok(Expr::Ref(name))
            }
            c => Err(format!("unexpected {c:?} in grammar")),
        }
    }

    fn item(&mut self) -> Result<Expr, String> {
        let mut e = self.atom()?;
        loop {
            let (lo, hi) = match self.chars.peek() {
                Some('?') => (0, Some(1)),
                Some('*') => (0, None),
                Some('+') => (1, None),
                Some('{') => {
                    self.chars.next();
                    let lo = self.number()?;
                    let hi = if self.chars.peek() == Some(&',') {
                        self.chars.next();
                        if self.chars.peek() == Some(&'}') {
                            None
                        } else {
                            Some(self.number()?)
                        }
                    } else {
                        Some(lo)
                    };
                    if self.chars.next() != Some('}') {
                        return Err("unclosed { in grammar".into());
                    }
                    e = Expr::Repeat(Box::new(e), lo, hi);
                    continue;
                }
                _ => return Ok(e),
            };
            self.chars.next();
            e = Expr::Repeat(Box::new(e), lo, hi);
        }
    }

    fn seq(&mut self) -> Result<Expr, String> {
        let mut items = Vec::new();
        loop {
            self.skip();
            match self.chars.peek() {
                None | Some('|') | Some(')') => break,
                _ => items.push(self.item()?),
            }
        }
        Ok(Expr::Seq(items))
    }

    fn alt(&mut self) -> Result<Expr, String> {
        let mut alts = vec![self.seq()?];
        while self.chars.peek() == Some(&'|') {
            self.chars.next();
            alts.push(self.seq()?);
        }
        Ok(if alts.len() == 1 { alts.pop().unwrap() } else { Expr::Alt(alts) })
    }
}

/// `name ::= body` rules; a body runs until the next line that starts a rule.
fn rules(gbnf: &str) -> Result<HashMap<String, Expr>, String> {
    let starts_rule = |line: &str| {
        let t = line.trim_start();
        let name_len = t.find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_')).unwrap_or(t.len());
        name_len > 0 && t[..name_len].starts_with(|c: char| c.is_ascii_alphabetic()) && t[name_len..].trim_start().starts_with("::=")
    };
    let mut bodies: Vec<(String, String)> = Vec::new();
    for line in gbnf.lines() {
        if starts_rule(line) {
            let (name, body) = line.split_once("::=").unwrap();
            bodies.push((name.trim().to_string(), body.to_string()));
        } else if let Some((_, body)) = bodies.last_mut() {
            body.push('\n');
            body.push_str(line);
        } else if !line.trim().is_empty() && !line.trim_start().starts_with('#') {
            return Err("grammar text before the first rule".into());
        }
    }
    let mut out = HashMap::new();
    for (name, body) in bodies {
        let mut p = Parser { chars: body.chars().peekable() };
        let e = p.alt()?;
        p.skip();
        if let Some(c) = p.chars.next() {
            return Err(format!("unexpected {c:?} in rule {name}"));
        }
        if out.insert(name.clone(), e).is_some() {
            return Err(format!("rule {name} is defined twice"));
        }
    }
    Ok(out)
}

fn class_char(c: char) -> String {
    format!("\\x{{{:X}}}", c as u32)
}

fn emit(e: &Expr, rules: &HashMap<String, Expr>, stack: &mut Vec<String>, out: &mut String) -> Result<(), String> {
    match e {
        Expr::Lit(s) => {
            out.push_str("(?:");
            for c in s.chars() {
                out.push_str(&class_char(c));
            }
            out.push(')');
        }
        Expr::Class(negated, items) => {
            out.push('[');
            if *negated {
                out.push('^');
            }
            for (lo, hi) in items {
                out.push_str(&class_char(*lo));
                if hi != lo {
                    out.push('-');
                    out.push_str(&class_char(*hi));
                }
            }
            out.push(']');
        }
        Expr::Seq(items) => {
            out.push_str("(?:");
            for i in items {
                emit(i, rules, stack, out)?;
            }
            out.push(')');
        }
        Expr::Alt(alts) => {
            out.push_str("(?:");
            for (n, a) in alts.iter().enumerate() {
                if n > 0 {
                    out.push('|');
                }
                emit(a, rules, stack, out)?;
            }
            out.push(')');
        }
        Expr::Repeat(inner, lo, hi) => {
            out.push_str("(?:");
            emit(inner, rules, stack, out)?;
            out.push(')');
            match hi {
                Some(hi) => out.push_str(&format!("{{{lo},{hi}}}")),
                None => out.push_str(&format!("{{{lo},}}")),
            }
        }
        Expr::Ref(name) => {
            if stack.contains(name) {
                return Err(format!("rule {name} is recursive; only grammars without recursion are supported"));
            }
            let body = rules.get(name).ok_or_else(|| format!("rule {name} is not defined"))?;
            stack.push(name.clone());
            emit(body, rules, stack, out)?;
            stack.pop();
        }
    }
    Ok(())
}

/// A GBNF grammar without recursion as one regex matching exactly `root`.
pub fn to_regex(gbnf: &str) -> Result<String, String> {
    let rules = rules(gbnf)?;
    let root = rules.get("root").ok_or("the grammar has no root rule")?;
    let mut out = String::new();
    emit(root, &rules, &mut vec!["root".into()], &mut out)?;
    Ok(out)
}

// ── the constraint ───────────────────────────────────────────────────────

/// Each token id's raw bytes; `None` for a special or added token, never allowed.
pub type TokenBytes = Vec<Option<Box<[u8]>>>;

/// A grammar being followed: a lazy DFA and where the reply has got to.
pub struct Constraint {
    dfa: DFA,
    cache: Cache,
    state: LazyStateID,
    /// Allowed token ids per state; a DFA cache clear renumbers states.
    masks: HashMap<LazyStateID, std::sync::Arc<Vec<u32>>>,
    clears: usize,
}

const DFA_CACHE: usize = 64 << 20;

impl Constraint {
    pub fn new(gbnf: &str) -> Result<Self, String> {
        let re = to_regex(gbnf)?;
        let dfa = DFA::builder()
            .configure(DFA::config().match_kind(MatchKind::All).cache_capacity(DFA_CACHE))
            .build(&re)
            .map_err(|e| format!("the grammar can't be compiled: {e}"))?;
        let mut cache = dfa.create_cache();
        let state = dfa.start_state(&mut cache, &start::Config::new().anchored(Anchored::Yes)).map_err(|e| e.to_string())?;
        Ok(Self { dfa, cache, state, masks: HashMap::new(), clears: 0 })
    }

    fn walk(&mut self, from: LazyStateID, bytes: &[u8]) -> Result<Option<LazyStateID>, String> {
        let mut s = from;
        for &b in bytes {
            s = self.dfa.next_state(&mut self.cache, s, b).map_err(|e| e.to_string())?;
            if s.is_dead() || s.is_quit() {
                return Ok(None);
            }
        }
        Ok(Some(s))
    }

    /// Consumes a picked token's bytes; errors if the grammar doesn't allow them.
    pub fn advance(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.state = self.walk(self.state, bytes)?.ok_or("the reply left its grammar")?;
        Ok(())
    }

    /// Whether the reply may end here.
    pub fn can_end(&mut self) -> Result<bool, String> {
        Ok(self.dfa.next_eoi_state(&mut self.cache, self.state).map_err(|e| e.to_string())?.is_match())
    }

    /// Token ids whose bytes the grammar allows next (`tokens[id]`; `None` is
    /// a special token, never allowed). Memoized per DFA state.
    pub fn allowed(&mut self, tokens: &[Option<Box<[u8]>>]) -> Result<std::sync::Arc<Vec<u32>>, String> {
        if self.cache.clear_count() != self.clears {
            self.clears = self.cache.clear_count();
            self.masks.clear();
        }
        if let Some(m) = self.masks.get(&self.state) {
            return Ok(m.clone());
        }
        let from = self.state;
        let mut ids = Vec::new();
        for (id, bytes) in tokens.iter().enumerate() {
            if let Some(bytes) = bytes {
                if !bytes.is_empty() && self.walk(from, bytes)?.is_some() {
                    ids.push(id as u32);
                }
            }
        }
        // Walking may have cleared the cache (states renumbered): don't key the result on a stale id.
        let ids = std::sync::Arc::new(ids);
        if self.cache.clear_count() == self.clears {
            self.masks.insert(from, ids.clone());
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_decision_grammar() {
        assert_eq!(choices(r#"root ::= "A" | "B" | "C""#), Some(vec!["A".into(), "B".into(), "C".into()]));
        assert_eq!(choices("root::=\"yes\"|\"no\"\n"), Some(vec!["yes".into(), "no".into()]));
        assert_eq!(choices(r#"root ::= "say \"hi\"""#), Some(vec![r#"say "hi""#.into()]));
    }

    #[test]
    fn anything_else_is_not_a_choice() {
        for g in [
            "",
            r#"root ::= "{" ws "}""#,
            "root ::= \"A\" | \"B\"\nws ::= [ ]*",
            r#"root ::= [A-C]"#,
            r#"root ::= "A" |"#,
            r#"root ::= """#,
            r#"other ::= "A""#,
            r#"root ::= "A\n""#,
        ] {
            assert_eq!(choices(g), None, "{g:?}");
        }
    }

    fn grammars() -> serde_json::Map<String, serde_json::Value> {
        let path = format!("{}/tests/fixtures/llm/grammars.json", env!("CARGO_MANIFEST_DIR"));
        serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(path).unwrap()).unwrap().as_object().unwrap().clone()
    }

    /// Whole-string match, byte by byte, as the engine walks tokens.
    fn matches(c: &mut Constraint, text: &str) -> bool {
        let start = c.state;
        let ok = c.walk(start, text.as_bytes()).unwrap().is_some_and(|s| {
            c.state = s;
            c.can_end().unwrap()
        });
        c.state = start;
        ok
    }

    #[test]
    fn compiles_every_argument_grammar_and_holds_replies_to_it() {
        let g = grammars();
        assert!(g.len() >= 8, "fixture has the tools' grammars");
        for (tool, gbnf) in &g {
            Constraint::new(gbnf.as_str().unwrap()).unwrap_or_else(|e| panic!("{tool}: {e}"));
        }
        let mut search = Constraint::new(g["kb_search"].as_str().unwrap()).unwrap();
        for ok in [r#"{"query": "group discount", "scope": "focused"}"#, "{\"query\":\"x\",\"scope\":\"broad\"}", "{ \"query\" :\n\"Ünïcödé ✓ 日本\" , \"scope\": \"broad\" }", r#"{"query": "say \"hi\"", "scope": "broad"}"#] {
            assert!(matches(&mut search, ok), "{ok}");
        }
        for bad in [
            r#"{"query": "x", "scope": "wide"}"#,
            r#"{"scope": "broad", "query": "x"}"#,
            r#"{"query": "x"}"#,
            r#"```json {"query": "x", "scope": "broad"}"#,
            r#"{"query": "x", "scope": "broad"} and more"#,
            "{\"query\": \"a\u{1}b\", \"scope\": \"broad\"}",
        ] {
            assert!(!matches(&mut search, bad), "{bad}");
        }
        let mut names = Constraint::new(g["kb_find_symbols"].as_str().unwrap()).unwrap();
        assert!(matches(&mut names, r#"{"names": ["computeFare", "withRetry"], "node_type": "Function"}"#));
        assert!(!matches(&mut names, r#"{"names": ["a","b","c","d","e","f"], "node_type": "any"}"#), "at most 5 names");
    }

    #[test]
    fn a_prefix_stays_alive_and_the_end_only_matches_when_complete() {
        let g = grammars();
        let mut c = Constraint::new(g["kb_search"].as_str().unwrap()).unwrap();
        c.advance(br#"{"query": "fare"#).unwrap();
        assert!(!c.can_end().unwrap());
        assert!(c.advance(b"\x01").is_err(), "a control character leaves the grammar");
        let mut c = Constraint::new(g["kb_search"].as_str().unwrap()).unwrap();
        c.advance(br#"{"query": "fare", "scope": "focused"}"#).unwrap();
        assert!(c.can_end().unwrap());
    }

    #[test]
    fn masks_tokens_by_their_bytes() {
        let g = grammars();
        let mut c = Constraint::new(g["kb_search"].as_str().unwrap()).unwrap();
        let tokens: Vec<Option<Box<[u8]>>> = ["{", "{\"", "```", " ", "{\"query", "\"", "x", ""].iter().map(|t| Some(t.as_bytes().into())).chain([None]).collect();
        assert_eq!(*c.allowed(&tokens).unwrap(), vec![0, 1, 4], "only an opening brace (with what may follow it)");
        c.advance(b"{\"query\": \"").unwrap();
        let inside = c.allowed(&tokens).unwrap();
        assert!([2, 3, 5, 6].iter().all(|i| inside.contains(i)), "any text, or the closing quote, inside a string: {inside:?}");
        assert!(!inside.contains(&8) && !inside.contains(&7), "never a special or empty token");
    }

    #[test]
    fn refuses_what_it_cannot_follow() {
        for (g, why) in [
            ("expr ::= \"x\"", "no root"),
            ("root ::= a\na ::= \"(\" a \")\" | \"x\"", "recursive"),
            ("root ::= missing", "not defined"),
            ("root ::= \"x\"\nroot ::= \"y\"", "twice"),
            ("root ::= (\"x\"", "unclosed"),
            ("root ::= [z-a]", "bad range"),
            ("root ::= \"x\"{2", "unclosed"),
        ] {
            let err = Constraint::new(g).err().unwrap_or_else(|| panic!("{g:?} compiled"));
            assert!(err.contains(why), "{g:?}: {err}");
        }
        // GBNF features the parser does follow
        let mut c = Constraint::new("# comment\nroot ::= item+ # trailing\n  \"!\"\nitem ::= [a-c] | \"-\"").unwrap();
        assert!(matches(&mut c, "ab-c!") && !matches(&mut c, "!") && !matches(&mut c, "abd!"));
    }
}
