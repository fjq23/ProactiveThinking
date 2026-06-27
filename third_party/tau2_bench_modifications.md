# Tau2 Bench Modifications

## Changed file

- [third_party/tau2-bench/src/tau2/user/user_simulator.py](./src/tau2/user/user_simulator.py:231)

## What was changed

- Changed the user simulator from single-shot generation to retry-based generation.
- Added validation before accepting the generated user message.
- Added up to 4 retries when the simulator returns an invalid or empty message.
- Added a final fallback to `STOP` if all retries fail.

## Why it was changed

- Some providers occasionally return empty user messages.
- Without a fallback, a single bad generation can break or corrupt a benchmark run.
- The patch makes the user simulator more robust and prevents the entire experiment from failing because of one empty output.