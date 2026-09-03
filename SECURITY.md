# Security Policy

## Reporting a vulnerability

Email **gbranaa4@gmail.com** with "SECURITY" in the subject. Please do not open
a public issue for a security report.

Include: what you found, how to reproduce it, and the impact you think it has.
You'll get an acknowledgement within a few days. This is a single-maintainer
project with no bug-bounty budget — what you get is a fix, credit in the
changelog if you want it, and a straight answer.

## Scope

OBSERVE runs entirely on your machine and makes no outbound network calls after
the one-time model download. The things worth reporting:

- a path that lets indexed code or the local index leave the machine
- the MCP server exposing more than the documented tools, or accepting input it
  shouldn't
- the one-line installer (`install.sh` / `install.ps1`) doing something outside
  `~/.observe`
- arbitrary code execution from opening/indexing a hostile repository

## Supported versions

The latest tagged release is supported. Older tags are not patched.
