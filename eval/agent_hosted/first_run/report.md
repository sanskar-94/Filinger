# Agent evaluation (synthetic world)

60 questions built from the generator's ground truth: 10 guidance, 10 reaction, 20 disclosure_yes, 20 disclosure_no. Agent: hosted model through an OpenAI compatible API (nvidia/nemotron-3-ultra-550b-a55b). Intervals are 95% bootstrap intervals over questions.

| System | Guidance, dated (right paragraph cited and direction stated) | Stock reaction, dated (right return given) | Disclosure made (a stating paragraph cited) | No such disclosure (says so) | Mean of the four | Citations valid |
|---|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b, bm25 | 1.000 [1.000, 1.000] | 0.300 [0.100, 0.600] | 0.800 [0.600, 0.950] | 0.900 [0.750, 1.000] | 0.750 | 1.000 |
| nvidia/nemotron-3-ultra-550b-a55b, hybrid | 1.000 [1.000, 1.000] | 0.300 [0.100, 0.600] | 0.050 [0.000, 0.150] | 0.800 [0.600, 0.950] | 0.538 | 1.000 |

| System | Guidance: right paragraph cited | Guidance: cited first | Disclosure: share of citations that state it | Turns per question | Latency p50 / p95 (s) | Cost (USD) | Errors | Empty answers |
|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b, bm25 | 1.000 | 1.000 | 0.800 | 5.3 | 37.629 / 75.565 | 0.00 | 0 | 0 |
| nvidia/nemotron-3-ultra-550b-a55b, hybrid | 1.000 | 1.000 | 0.050 | 4.1 | 65.158 / 283.962 | 0.00 | 24 | 24 |

| Paired difference on the same questions | guidance | reaction | disclosure_yes | disclosure_no |
|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b, hybrid minus nvidia/nemotron-3-ultra-550b-a55b, bm25 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.750 [-0.900, -0.550] | -0.100 [-0.300, +0.100] |

Every check is mechanical: citations against the paragraphs the generator wrote, returns against the event table, the direction and a statement of absence by pattern. The patterns are strict for the offline agent, which quotes filing sentences; a model that words an answer freely can be marked wrong for phrasing, so read its direction and absence columns with answers.jsonl next to them.
