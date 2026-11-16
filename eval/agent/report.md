# Agent evaluation (synthetic world)

240 questions built from the generator's ground truth: 40 guidance, 40 reaction, 80 disclosure_yes, 80 disclosure_no. Agent: offline extractive. Intervals are 95% bootstrap intervals over questions.

| System | Guidance, dated (right paragraph cited and direction stated) | Stock reaction, dated (right return given) | Disclosure made (a stating paragraph cited) | No such disclosure (says so) | Mean of the four | Citations valid |
|---|---|---|---|---|---|---|
| offline, bm25 | 0.625 [0.475, 0.775] | 0.850 [0.725, 0.950] | 1.000 [1.000, 1.000] | 0.150 [0.075, 0.225] | 0.656 | 1.000 |
| offline, bm25, dated questions scoped | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.150 [0.075, 0.225] | 0.787 | 1.000 |
| offline, dense | 0.050 [0.000, 0.125] | 0.900 [0.800, 0.975] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.488 | 1.000 |
| offline, dense, dated questions scoped | 0.475 [0.325, 0.625] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.619 | 1.000 |
| offline, hybrid | 0.075 [0.000, 0.175] | 0.900 [0.800, 0.975] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.494 | 1.000 |
| offline, hybrid, dated questions scoped | 0.925 [0.825, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.731 | 1.000 |
| offline, hybrid, dated questions scoped, company name kept | 0.600 [0.450, 0.750] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.650 | 1.000 |

| System | Guidance: right paragraph cited | Guidance: cited first | Disclosure: share of citations that state it | Latency p50 / p95 (s) | Cost (USD) | Errors |
|---|---|---|---|---|---|---|
| offline, bm25 | 0.625 | 0.125 | 0.642 | 0.006 / 0.024 | 0.00 | 0 |
| offline, bm25, dated questions scoped | 1.000 | 0.225 | 0.642 | 0.006 / 0.017 | 0.00 | 0 |
| offline, dense | 0.050 | 0.025 | 0.537 | 0.008 / 0.037 | 0.00 | 0 |
| offline, dense, dated questions scoped | 0.475 | 0.150 | 0.537 | 0.005 / 0.021 | 0.00 | 0 |
| offline, hybrid | 0.075 | 0.025 | 0.537 | 0.008 / 0.027 | 0.00 | 0 |
| offline, hybrid, dated questions scoped | 0.925 | 0.150 | 0.537 | 0.010 / 0.024 | 0.00 | 0 |
| offline, hybrid, dated questions scoped, company name kept | 0.600 | 0.000 | 0.525 | 0.023 / 0.073 | 0.00 | 0 |

| Paired difference on the same questions | guidance | reaction | disclosure_yes | disclosure_no |
|---|---|---|---|---|
| offline, bm25, dated questions scoped minus offline, bm25 | +0.375 [+0.225, +0.525] | +0.150 [+0.050, +0.275] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| offline, dense minus offline, bm25 | -0.575 [-0.750, -0.400] | +0.050 [-0.100, +0.200] | +0.000 [+0.000, +0.000] | -0.150 [-0.225, -0.075] |
| offline, dense, dated questions scoped minus offline, bm25 | -0.150 [-0.350, +0.050] | +0.150 [+0.050, +0.275] | +0.000 [+0.000, +0.000] | -0.150 [-0.225, -0.075] |
| offline, hybrid minus offline, bm25 | -0.550 [-0.725, -0.375] | +0.050 [-0.100, +0.200] | +0.000 [+0.000, +0.000] | -0.150 [-0.225, -0.075] |
| offline, hybrid, dated questions scoped minus offline, bm25 | +0.300 [+0.125, +0.475] | +0.150 [+0.050, +0.275] | +0.000 [+0.000, +0.000] | -0.150 [-0.225, -0.075] |
| offline, hybrid, dated questions scoped, company name kept minus offline, bm25 | -0.025 [-0.225, +0.175] | +0.150 [+0.050, +0.275] | +0.000 [+0.000, +0.000] | -0.150 [-0.225, -0.075] |

Every check is mechanical: citations against the paragraphs the generator wrote, returns against the event table, the direction and a statement of absence by pattern. The patterns are strict for the offline agent, which quotes filing sentences; a model that words an answer freely can be marked wrong for phrasing, so read its direction and absence columns with answers.jsonl next to them.
