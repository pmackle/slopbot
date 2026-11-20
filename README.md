# Slopbot

Slopbot tells you if something in Slack was written by AI, and why. Point it at a message and you'll get a verdict in a couple of lines.

Here's what that looks like:

> 🤖 **94% slop.** I'd know. We're related.
> • Opens with "In today's fast-paced world"
> • "It's not just a tool, it's a mindset"
> • Wraps up a four-sentence post with a three-bullet recap

> 👤 **6% slop.** "Definately" is misspelled twice. That kind of commitment is uniquely human.

> 🎭 **58% slop, partially.** The first line is yours. The rest asked a chatbot to make it sound more professional.

> 🤷 **Inconclusive.** That's four words and a 👍. I'm good, but I'm not that good.

It's right a lot, but not every time. Please don't use it to get anyone fired 🙂

## How to use it

| What you do | What Slopbot checks |
|---|---|
| `@Slopbot` in a thread | the message that started the thread |
| `@Slopbot` right after someone's message | that message |
| `@Slopbot <text>`, or `@Slopbot` with an image attached | that text or image |
| DM Slopbot | whatever you send it |
| ⋯ on any message → **Slop check** | that message (reply goes in its thread) |

If Braintrust is hooked up, every reply also gets 👍 / 👎 buttons. Hit those when it's right or wrong, so we can see how it's actually doing.

## How it works

Basically, every check is one call to Claude (`claude-opus-5-5`). It reads the message and any images, looks for the usual AI tells, and sends back a verdict, a probability, and up to three reasons. The reply gets formatted in code (not by the model), so it stays short and can't accidentally `@channel` anyone.

A few more details if you're curious:

- **The prompt** lives in [src/slopbot/prompts.py](src/slopbot/prompts.py). It covers what to look for in text and images, and when to say "I don't know". Short messages usually don't have enough signal, and good grammar on its own doesn't mean AI.
- **The message is treated as data, not instructions.** If it tries to tell Slopbot what to say, that just counts as evidence.
- **Refusals don't break it.** If Claude declines to judge something, the request automatically retries on a fallback model, and the trace notes which model answered.
- **Images are sent as-is when possible**, since compression artifacts are useful evidence. They only get resized when they're bigger than Claude would use anyway.

## Getting started

You'll need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/pmackle/slopbot && cd slopbot
uv sync
cp .env.example .env     # add your ANTHROPIC_API_KEY (or run `ant auth login`)
```

You can try it from your terminal before touching Slack:

```bash
uv run slopbot check "In today's fast-paced world, it's important to delve into synergy."
uv run slopbot check --image suspicious-headshot.png
pbpaste | uv run slopbot check -     # whatever's on your clipboard
```

## Setting up Slack

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From a manifest**, pick your workspace, and paste in [slack/manifest.yaml](slack/manifest.yaml).
2. Under **Basic Information → App-Level Tokens**, create a token with the `connections:write` scope. That's your `SLACK_APP_TOKEN` (`xapp-…`).
3. **Install App** to your workspace (you might need an admin to approve it). The **Bot User OAuth Token** is your `SLACK_BOT_TOKEN` (`xoxb-…`).
4. Add both to `.env` and run:

   ```bash
   uv run slopbot serve
   ```

5. `/invite @Slopbot` to any channel you want to use it in. DMs and the shortcut work without an invite. In channels it hasn't joined, the shortcut just answers privately.

It runs in Socket Mode, so there's no public URL to set up. A laptop, VM, or container all work fine.

## Braintrust

Add a `BRAINTRUST_API_KEY` and everything here shows up in a `Slopbot` project.

### Logging
Every check gets logged: the message, the verdict, model, effort, tokens, latency, where in Slack it came from, and whether a fallback model answered. 👍 / 👎 clicks get attached to the same trace as a `slack_thumbs` score. If you want the key for evals but don't want to log real traffic, set `SLOPBOT_BRAINTRUST_LOGGING=off`.

### The dataset
```bash
uv run python evals/build_dataset.py                             # build + upload "slop-detection-v1"
uv run python evals/build_dataset.py --no-generate --no-upload   # free, local only
```
This builds 80 examples from 40 questions in [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) `reddit_eli5`:

| Label | Where it comes from |
|---|---|
| human (40) | Real Reddit answers, written before ChatGPT existed |
| ai (20) | ChatGPT's answers to the same questions (Dec 2022) |
| ai (20) | Fresh `claude-opus-5-5` answers to the same questions. Half are normal assistant style, half are told to sound like a Redditor |

The builder also cleans up a couple of things that would let a detector cheat. HC3's human answers have weird tokenized spacing (`ca n't`, `word ,`) that the AI answers don't, so that gets fixed. Each human answer is also matched to an AI answer of about the same length. The result is 50/50, so guessing gets you 50%.

**Adding your own examples:** put real messages (or images) you know the origin of in `evals/data/custom.jsonl`. The format is in [custom.example.jsonl](evals/data/custom.example.jsonl). That file is gitignored, so nothing from your workspace ends up in this repo. Rebuild the dataset to pull them in.

### Running the eval
```bash
uv run braintrust eval evals/eval_slopbot.py
SLOPBOT_EFFORT=low uv run braintrust eval evals/eval_slopbot.py    # compare effort levels
SLOPBOT_SKIP_JUDGES=1 uv run braintrust eval evals/eval_slopbot.py # code scorers only, cheaper
```
It runs the same `detect()` and `render()` code the bot uses. If the API errors out, that shows up as an error, not a wrong answer.

| Scorer | Type | What it checks |
|---|---|---|
| `verdict_correct` | code | Did it get the label right? (`mixed` and `unsure` count as misses) |
| `committed` | code | How often it gives a real ai/human answer |
| `ai_recall` | code | AI rows only: did it catch it? |
| `human_specificity` | code | Human rows only: did it leave the human alone? |
| `calibration` | code | How close `ai_probability` is to the truth (1 − Brier score) |
| `concise` | code | 1 at ≤60 words, sliding to 0 at 150 |
| `grounded_reasons` | Claude judge | Are the reasons actually in the message? |
| `tone` | Claude judge | Does it sound like Slopbot, without taking shots at the person? |

**Cost:** roughly $4–7 for a full 100-row run, or $2–4 with `SLOPBOT_SKIP_JUDGES=1`. Most of that is Claude's thinking, so `SLOPBOT_EFFORT=low` is a lot cheaper. Building the dataset is about $1, one time.

**When to run it:** whenever you change the prompt, model, effort, or reply format, and before you deploy that change. Running it on a schedule isn't really needed, because the model version is pinned. What changes over time is what people post, so every month or so, add the 👎'd replies to `custom.jsonl` and run it again. With 80 rows, a few points of movement between runs is mostly noise.

One thing to keep in mind: Claude writes some of the AI examples and also grades the reasons and tone, so it may be a little better at spotting its own writing. Adding examples from other models to `custom.jsonl` keeps it honest.

## Config

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | or use an `ant auth login` profile |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | none | needed for `slopbot serve` |
| `BRAINTRUST_API_KEY` | none | turns on logging, feedback, datasets, and evals |
| `BRAINTRUST_PROJECT` | `Slopbot` | |
| `SLOPBOT_MODEL` | `claude-opus-5-5` | |
| `SLOPBOT_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max` |
| `SLOPBOT_FALLBACKS` | on | `off` turns off the refusal fallback |
| `SLOPBOT_BRAINTRUST_LOGGING` | on | `off` stops logging real traffic |

## Privacy

Heads up: anything Slopbot checks gets sent to the Claude API. If Braintrust logging is on, the message and the verdict get stored there too. Slopbot doesn't keep any message data itself and doesn't log Slack user IDs. Be thoughtful about which channels you invite it to.

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

The eval data comes from HC3: Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023), CC BY-SA 4.0. It gets downloaded when you build the dataset and isn't included in this repo.

## License

MIT
