---
title: Filings Intelligence Lab
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Filings Intelligence Lab: live demo

Ask a question about a company's filings and get an answer in which every statement links to the paragraph
it came from. The data is a synthetic EDGAR shaped world, built into the image, so nothing here is a claim about
real companies or markets. Code, evaluations and what has and has not been run:
https://github.com/sanskar-94/Filinger

## Putting this demo up, free

1. Make a free account at huggingface.co and choose New Space. Pick the Docker SDK, the blank template, the free
   CPU basic hardware, and make it public.
2. In the Space's Files tab, upload this folder's `Dockerfile` and this `README.md` (the block at the top of this
   file is the Space's settings).
3. Wait for the build, about ten minutes: it clones the repository, builds the demo data and the web app.
4. Optional: to answer with a hosted model, add a secret named `NVIDIA_API_KEY` in the Space's Settings (a free
   key from build.nvidia.com). Without it the agent answers in its offline mode.

To pick up new commits later, use Factory rebuild in the Space's Settings.
