# Security Policy

## Supported versions

Security fixes target the latest release on `main`.

## Reporting a vulnerability

Do not publish credentials, private prompts, provider responses, or exploit details in a public issue. Open a minimal GitHub issue requesting a private contact channel, without sensitive details.

Include affected version, reproduction steps, impact, and suggested mitigation when possible.

## Data handling

Universal-Zero sends prompts to endpoints configured by the operator. Result exports may contain complete prompts and model responses. Keep result files private when they contain sensitive material. API keys are read from CLI/environment and are never intentionally written to result files.
