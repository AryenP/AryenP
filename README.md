Data Science student at UC San Diego's Halıcıoğlu Data Science Institute (expected June 2028). I work on LLM inference and evaluation: I've served Qwen3-8B on vLLM and measured what 4-bit quantization buys, written speculative decoding from scratch on GPT-2, and I'm a student researcher on Engineers for Exploration's mangrove monitoring team. Looking for a Summer 2027 internship.

[Portfolio](https://aryenp.github.io/portfolio/) · [Résumé](https://aryenp.github.io/portfolio/resume/)

### Experience

**Student Researcher**, Engineers for Exploration, UC San Diego · Sept 2026 to present
Mangrove Monitoring, with Scripps Institution of Oceanography. Working on machine learning over satellite and drone imagery to estimate mangrove coverage in Baja California, on a team run with the Aburto Lab.

**Founder and Quantitative Developer**, Stealth · June 2024 to present
Designed Python trading algorithms (ARIMA, Bollinger Bands, VWAP, linear regression) on S&P 500 equities and integrated Polygon.io real-time market feeds with automated reporting.

### Selected work

#### 01 / LLM Inference & Evaluation Platform · [Repo](https://github.com/AryenP/llm-inference-platform)

**The Problem:** Self-hosting a model is supposed to be faster and cheaper than an API, but the answer depends on quantization, load, and how busy the GPU stays, and most write-ups skip the measurement.

**The Solution:** Qwen3-8B on vLLM behind a FastAPI gateway, with hybrid retrieval over PostgreSQL and pgvector and an evaluation suite that scores retrieval and answer quality. I benchmarked 16 configurations across input length and concurrency with prefix caching disabled and 10 warmup requests discarded per run, and the recorder rejects any row missing its hardware, quantization kernel, sampler or cache state. Serving a 4-bit AWQ checkpoint in place of BF16 on one L40S cut time-to-first-token from 165 ms to 107 ms and inter-token latency from 28.8 ms to 10.3 ms, raised sustained throughput from 6.31 to 7.62 requests/sec, and cut cost per 1,000 queries from $0.048 to $0.040. Priced against hosted Qwen3-8B on 6 Oct 2026, that is $0.040 self-hosted at 64 concurrent requests against $0.178 for the cheapest published API, a 4.5x gap that inverts below 1.70 requests/sec, where the idle GPU costs more than the API. Hybrid BM25 plus dense retrieval did not beat dense alone (recall@5 0.700 against 0.693, one question in 150), and that null result stays in the README. A CLI gate exits non-zero when recall@1 drops more than 0.01 below the recorded baseline.

**Tech Stack:** `Python` `vLLM` `PostgreSQL` `pgvector` `FastAPI` `Docker` `GitHub Actions`

#### 02 / spec-decode · [Repo](https://github.com/AryenP/spec-decode)

**The Problem:** Speculative decoding is described as a free speedup. What it buys depends on the hardware, the draft model and how long the generation runs, and the output is supposed to stay identical.

**The Solution:** GPT-2 decoding written from scratch: the forward pass, a KV cache with per-row rollback, and the decoders in about 600 lines, with Hugging Face used only for the weights and tokenizer. gpt2-xl is the target and gpt2 the draft. At 128 new tokens, speculative decoding runs 1.90x faster than a plain KV cache on an M2 Pro and 1.47x on an RTX 4090. At 512 tokens, prompt lookup with no draft model reaches 2.63x and 2.69x. In fp32 the output is byte-identical to the large model decoding alone. Every number has its full configuration in a row under `results/`, and the two bugs the numbers caught are kept there too.

**Tech Stack:** `Python` `PyTorch` `RunPod`

#### 03 / Griddy · [Repo](https://github.com/AryenP/griddy)

**The Problem:** When the power grid runs short, the usual plan for AI data centers is to switch to diesel generators.

**The Solution:** Griddy watches the live grid (prices, the day-ahead forecast, the fuel mix and emergency alerts) and plans every AI job on the machine for the next 24 hours. When the grid needs power back, it pauses or slows the work that can wait, keeps the services people are using running, and asks a job's owner before pushing back their deadline. We built it as a team of four for the Microsoft Challenge at Cal Hacks 13; I ran the demo machine and built the runtime, integration, frontend and agent layer. Replaying Winter Storm Uri from the grid operators' EIA-930 records, the machine dropped from 28.2 W to 10.9 W when ERCOT declared EEA 3, a measured 61% cut, and `make verify` recomputes every published number from the raw logs.

**Tech Stack:** `Python` `JavaScript` `WebSockets`

#### 04 / Runway · [Repo](https://github.com/AryenP/academic-dashbaord) · [Live site](https://aryenp.github.io/academic-dashbaord/)

**The Problem:** Exam dates live in course calendars and syllabi, study time lives nowhere, and nothing compares the plan with what actually happened.

**The Solution:** Runway reads Canvas, Google, iCloud and Outlook calendar feeds every 15 minutes, finds the courses, exams and deadlines, and draws every day up to the last exam with study blocks in course colors. Title rules read most events; the ones they're unsure about go to Claude, which sees titles, times and places and nothing else, and any change you make wins over the AI. It drafts study blocks before each exam as proposals you accept, draws the time you actually studied beside the plan, and gives each exam a readiness score. Session timers come from timestamps in the database rather than a ticking clock, so they stay right across reloads, sleep and devices. Everything sits in Supabase Postgres behind Row Level Security, and you can download or delete all of it.

**Tech Stack:** `TypeScript` `Supabase` `PostgreSQL` `GitHub Actions` `Claude API`

#### 05 / Afterglow · [Repo](https://github.com/AryenP/afterglow)

**The Problem:** Music on a Mac plays in a window nobody looks at, so the lyrics and the album's colors stay hidden behind whatever you're working on.

**The Solution:** Afterglow is a macOS menu-bar app that lights the rim of the display in colors pulled from the album art, pulsing on every kick drum, and shows lyrics one word at a time, timed to the song, on the desktop or the lock screen. It has five lyric styles, one of them under a Metal glass shader, and a glass now-playing card under the lock-screen clock. It works with Spotify and Apple Music on Apple silicon.

**Tech Stack:** `Swift` `SwiftUI` `AppKit` `Metal`

#### 06 / GPU MODE kernels · [Repo](https://github.com/AryenP/gpu-mode-kernels)

**The Problem:** A kernel that beats the PyTorch reference can still sit far down the leaderboard, because the reference is a weak bar.

**The Solution:** CUDA kernels submitted to GPU MODE's ranked leaderboards and run on their B200s. After six tuning batches, `grayscale_v2` runs at 598 µs (9.3x the reference, 12th of 45) and `vectorsum_v2` at 42.3 µs (6.9x, 24th of 79), as of 17 Oct 2026. The first submissions, in Triton, ranked 34th and 67th. The README keeps the framing I got wrong: measuring headroom against the reference instead of the board, where the top 30 on grayscale already sit at the roofline.

**Tech Stack:** `CUDA` `Triton` `Python`

### More projects

**PhotoRevive AI:** CNN photo restoration that repairs scratches and colorizes black-and-white photographs. I curated the dataset, built the interface, and field-tested it with the Sunnyvale Senior Center.

**PlateWise AI:** Nutrition estimation from food images, built on Azure Cognitive Services.

### Awards

1st Place, De Anza Annual Hackathon (OpenAI-sponsored), 400+ competitors
