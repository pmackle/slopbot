# Slopbot

A Slack bot that tells you whether a message or image is AI-generated, and why, in a line or two of light sarcasm. Named in honor of AI slop.

Example reply (illustrative):

> 🤖 **91% slop.** This post has more em dashes than a Victorian novel.
> • Opens with "In today's fast-paced world"
> • "It's not just a tool, it's a mindset", the classic two-step
> • Ends with an unsolicited three-bullet recap
>
> `👍 Nailed it` `👎 Wrong`

Verdicts are probabilistic. They're for fun and for raising an eyebrow, not for disciplining anyone.

## Using it in Slack

| You do | Slopbot judges |
|---|---|
| `@Slopbot` in a thread reply | the thread's parent message |
| `@Slopbot <some text>` or `@Slopbot` with an image attached | that text and those images |
| `@Slopbot` on its own, at the top level | the message posted just before |
| DM Slopbot | whatever you send (text and/or images) |
| ⋯ menu on any message → **Slop check** | that message, replying in its thread |

The 👍/👎 buttons send your rating to Braintrust, so the bot's real-world accuracy is tracked.

## How it works

Each check is a **single Claude call** (`claude-opus-5-5`) that returns a structured verdict:

```json
{"verdict": "ai | human | mixed | unsure", "ai_probability": 0-100, "reasons": ["..."], "quip": "..."}
```

- **Prompt** ([src/slopbot/prompts.py](src/slopbot/prompts.py)): the persona, a rubric of text and image tells, and calibration rules (short messages → `unsure`; good grammar alone isn't evidence). The judged content is wrapped in tags and treated as data. A message that tries to instruct the bot gets cited as evidence instead.
- **Detector** ([src/slopbot/detector.py](src/slopbot/detector.py)): structured output via `client.beta.messages.parse`, effort set explicitly, and the server-side refusal fallback (`fallbacks: "default"`) on. If Claude declines, the API retries on a fallback model, and the model that actually answered is recorded in the trace.
- **Rendering** ([src/slopbot/render.py](src/slopbot/render.py)): code, not the model, formats the reply. That keeps it short (≤3 reasons, clipped), escapes Slack control characters so quoted text can't `@channel` anyone, and adds the feedback buttons.
- **Images**: downloaded with the bot token, kept byte-for-byte when possible (compression artifacts matter), and downscaled only when larger than Claude would use anyway.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/pmackle/slopbot && cd slopbot
uv sync
cp .env.example .env     # add ANTHROPIC_API_KEY (or run `ant auth login`)

uv run slopbot check "In today's fast-paced world, it's important to delve into synergy."
uv run slopbot check --image suspicious-headshot.png
pbpaste | uv run slopbot check -          # judge your clipboard
```

## Slack setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From a manifest**, pick your workspace, and paste [slack/manifest.yaml](slack/manifest.yaml).
2. **Basic Information → App-Level Tokens** → generate a token with the `connections:write` scope. That's `SLACK_APP_TOKEN` (`xapp-…`).
3. **Install App** to the workspace (an admin may need to approve). The **Bot User OAuth Token** is `SLACK_BOT_TOKEN` (`xoxb-…`).
4. Put both in `.env`, then run:

   ```bash
   uv run slopbot serve
   ```

5. `/invite @Slopbot` to any channel where you want mentions to work. DMs and the shortcut work without inviting it, though the shortcut answers privately in channels it hasn't joined.

It uses Socket Mode, so there's no public URL or port to expose. Anywhere that can make outbound connections can host it (laptop, VM, container).

## Braintrust

Set `BRAINTRUST_API_KEY` and everything below lights up in the `Slopbot` project.

### Production tracing and feedback
Every check is logged as a trace: input, verdict, model, effort, token usage, latency, the Slack surface it came from, and whether a fallback model answered. Clicks on 👍/👎 attach a `slack_thumbs` score to that trace. Set `SLOPBOT_BRAINTRUST_LOGGING=off` to keep the key for evals without logging live traffic.

### Dataset
```bash
uv run python evals/build_dataset.py              # build and upload "slop-detection-v1"
uv run python evals/build_dataset.py --no-generate --no-upload   # free, local only
```
It builds a balanced 80-row set over 40 questions from [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) `reddit_eli5`:

| Label | Source |
|---|---|
| human (40) | Real Reddit answers written before ChatGPT existed |
| ai (20) | HC3's ChatGPT (Dec 2022) answers to the same questions |
| ai (20) | Fresh `claude-opus-5-5` answers to the same questions: half in default assistant style, half told to pass as a Redditor |

The builder removes shortcuts that would let a detector score well without judging the writing. HC3's human text is tokenized (`ca n't`, `word ,`) and its ChatGPT text isn't, so the human text is detokenized and any row with leftover artifacts is dropped. Each human answer is also length-matched to its AI counterpart. The majority-class baseline is 50%.

**Add your own examples** in `evals/data/custom.jsonl` (format in [custom.example.jsonl](evals/data/custom.example.jsonl)): real Slack messages and images whose origin you actually know. That file is gitignored, so workplace messages never land in this public repo. Rebuild to include them.

### Running the eval
```bash
uv run braintrust eval evals/eval_slopbot.py
SLOPBOT_EFFORT=low uv run braintrust eval evals/eval_slopbot.py   # compare effort levels
SLOPBOT_SKIP_JUDGES=1 uv run braintrust eval evals/eval_slopbot.py # code scorers only, cheaper
```
The eval calls the same `detect()` and `render()` code the bot uses. API failures are recorded as errors, not wrong answers.

| Scorer | Kind | Measures |
|---|---|---|
| `verdict_correct` | code | exact label match (`mixed`/`unsure` count as misses) |
| `committed` | code | share of definite ai/human verdicts |
| `ai_recall` | code | AI rows only: caught it? |
| `human_specificity` | code | human rows only: avoided a false accusation? |
| `calibration` | code | 1 − Brier score of `ai_probability` |
| `concise` | code | 1 at ≤60 words, linear to 0 at 150 |
| `grounded_reasons` | Claude judge | do the reasons cite things actually in the content? |
| `tone` | Claude judge | slightly sarcastic and funny, aimed at the writing and not the person? |

Rough cost: about $4–6 per full run at Opus 5.5 rates (80 detections + 160 judge calls), and about $1 for the one-time dataset generation. Use `SLOPBOT_TRIALS=3` for error bars.

Caveat: Claude both generates some of the AI rows and judges reasons and tone. Expect it to be somewhat better at spotting its own style than other models'. Add other models' output to `custom.jsonl` to keep it honest.

## Configuration

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | or use an `ant auth login` profile |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | none | needed for `slopbot serve` |
| `BRAINTRUST_API_KEY` | none | enables tracing, feedback, datasets, evals |
| `BRAINTRUST_PROJECT` | `Slopbot` | |
| `SLOPBOT_MODEL` | `claude-opus-5-5` | |
| `SLOPBOT_EFFORT` | `medium` | `low`/`medium`/`high`/`xhigh`/`max` |
| `SLOPBOT_FALLBACKS` | on | `off` disables the server-side refusal fallback |
| `SLOPBOT_BRAINTRUST_LOGGING` | on | `off` stops production tracing |

## Privacy

Judged content goes to the Claude API. If Braintrust logging is on, the content and verdicts are also stored in your Braintrust project. Slopbot stores no message data itself and doesn't log Slack user IDs. Think about which channels you invite it to.

## Deploying

```bash
docker build -t slopbot .
docker run --env-file .env slopbot
```

## Development

```bash
uv run pytest            # unit tests, no API calls
uv run ruff check . && uv run ruff format --check .
```

## Credits

The eval data uses HC3: Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023), CC BY-SA 4.0. It's downloaded at build time and isn't redistributed here.

## License

MIT
