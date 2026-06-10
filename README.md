# BMG Project für Fortgeschrittene Methoden

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Objectives

1. Compare the definition of best matches and weak best matches:
   1. Check whether every best match in (N, σ) is also a weak best match. Test this computationally, but also try to give a formal argument.
   2. Check if a modified version of the BIC-cherry + expansion procedureyields networks that explain weak best match graphs.
2. Find a way to edit the explaining networks for (weak) best matches to become more tree like.

## 📌 Tasks

| Todo                                                            | Name    | Tested | Done  |
| --------------------------------------------------------------- | ------- | ------ | ----- |
| **LEAH GITHUB SSH AUTH**                                        | **ALL** | ❗ YES | ✅    |
| Keep Distance attribute in gene tree                            | Lenny   | ❗ YES | ✅    |
| Hybrid-Node-Insertion                                           | Lenny   | ❗ NO  | ❗ NO |
| BIC Cherry Expansion                                            | Leah    | ❗ NO  | ❗ NO |
| GraphOperations: "Contract", "Extend", "Delete redundant nodes" | OPEN    | ❗ NO  | ❗ NO |
| Test (bm = weak bm?) with python                                | Leo    | ❗ NO  | ❗ NO |
| Implement wbmg_from_network() | Leo |  YES(1)  | ✅ |
| Implement bmg_from_network() | Leo |  YES(1)  | ✅ |

## Help

- how to run project: [uv](https://docs.astral.sh/uv/guides/scripts/)

```
uv run main.py
```
or
```
uv run pytest
```
for testing
