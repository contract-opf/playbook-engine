# Security Policy

## Reporting a code vulnerability

Please report vulnerabilities in the engine or its toolchain **privately**
via GitHub's private vulnerability reporting: **Security tab → "Report a
vulnerability"** on this repository. Do not open a public issue for a
vulnerability.

We aim to acknowledge reports within 3 business days. Please include a
reproduction (fixture corpus + command) where possible.

GitHub private vulnerability reporting is the authoritative channel. If a
dedicated security contact address is published here later, it is an
addition, never a replacement.

## I found confidential, corpus, or PII material

This project compiles playbooks from **private legal agreements**. The most
likely real incident here is not a code vulnerability — it is confidential
material escaping into an artifact: a real party name in an example, a
clause quotation in a committed playbook, PII in a fixture, corpus content
in git history.

If you find anything like that, in any artifact of this repository:

1. **Report it privately, immediately**, via the same channel (Security
   tab → "Report a vulnerability"), marked **"confidential material"**.
   Do NOT quote the material in a public issue, PR, or discussion — that
   spreads it.
2. **Takedown first, questions later.** We will remove or rewrite the
   material (including history rewrites where needed) before debating
   whether it was truly sensitive. Erring toward removal is the policy.

The engine's own safeguards (the born-safe entity registry, the
pseudonymization pass, the export profile's residue judging) exist to make
this class of incident impossible; a report that one of them failed is
treated as a high-severity engineering bug in addition to the takedown.

## Supported versions

Pre-1.0: only the latest release/main receives fixes.
