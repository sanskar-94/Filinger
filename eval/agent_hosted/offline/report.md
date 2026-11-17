# Agent evaluation (synthetic world)

60 questions built from the generator's ground truth: 10 guidance, 10 reaction, 20 disclosure_yes, 20 disclosure_no. Agent: offline extractive. Intervals are 95% bootstrap intervals over questions.

| System | Guidance, dated (right paragraph cited and direction stated) | Stock reaction, dated (right return given) | Disclosure made (a stating paragraph cited) | No such disclosure (says so) | Mean of the four | Citations valid |
|---|---|---|---|---|---|---|
| offline, bm25 | 0.700 [0.400, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.100 [0.000, 0.250] | 0.700 | 1.000 |
| offline, bm25, dated questions scoped | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.100 [0.000, 0.250] | 0.775 | 1.000 |
| offline, hybrid | 0.100 [0.000, 0.300] | 0.600 [0.300, 0.900] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.425 | 1.000 |
| offline, hybrid, dated questions scoped | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.750 | 1.000 |
| offline, hybrid, dated questions scoped, company name kept | 0.900 [0.700, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.725 | 1.000 |

| System | Guidance: right paragraph cited | Guidance: cited first | Disclosure: share of citations that state it | Turns per question | Latency p50 / p95 (s) | Cost (USD) | Asked again after an API error | Errors left | Empty answers |
|---|---|---|---|---|---|---|---|---|---|
| offline, bm25 | 0.700 | 0.100 | 0.600 | 1.0 | 0.005 / 0.007 | 0.00 | 0 | 0 | 0 |
| offline, bm25, dated questions scoped | 1.000 | 0.200 | 0.600 | 1.0 | 0.005 / 0.006 | 0.00 | 0 | 0 | 0 |
| offline, hybrid | 0.100 | 0.000 | 0.450 | 1.0 | 0.015 / 0.027 | 0.00 | 0 | 0 | 0 |
| offline, hybrid, dated questions scoped | 1.000 | 0.400 | 0.450 | 1.0 | 0.008 / 0.009 | 0.00 | 0 | 0 | 0 |
| offline, hybrid, dated questions scoped, company name kept | 0.900 | 0.000 | 0.450 | 1.0 | 0.016 / 0.019 | 0.00 | 0 | 0 | 0 |

| Paired difference on the same questions | guidance | reaction | disclosure_yes | disclosure_no |
|---|---|---|---|---|
| offline, bm25, dated questions scoped minus offline, bm25 | +0.300 [+0.000, +0.600] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| offline, hybrid minus offline, bm25 | -0.600 [-1.000, -0.100] | -0.400 [-0.700, -0.100] | +0.000 [+0.000, +0.000] | -0.100 [-0.250, +0.000] |
| offline, hybrid, dated questions scoped minus offline, bm25 | +0.300 [+0.000, +0.600] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.100 [-0.250, +0.000] |
| offline, hybrid, dated questions scoped, company name kept minus offline, bm25 | +0.200 [+0.000, +0.402] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.100 [-0.250, +0.000] |

Every check is mechanical: citations against the paragraphs the generator wrote, returns against the event table, the direction and a statement of absence by pattern. The patterns are strict for the offline agent, which quotes filing sentences; a model that words an answer freely can be marked wrong for phrasing, so read its direction and absence columns with answers.jsonl next to them.
