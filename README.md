# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that hosts an OpenAI-compatible and an Anthropic-compatible API over
ten simulated free-tier model providers, and answers every request with the best
free model that can serve it.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI Chat Completions, JSON and SSE) and
  `POST /v1/messages` (Anthropic Messages, JSON and SSE), plus `GET /v1/models`
  in the shape each auth header names.
- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
  sealed quality table and the workload manifest. Probes every provider's catalog
  route in its own wire and registers the one catalog it built.
- Routes each request to the highest-rated free model that meets the request's
  requirements (tools, vision, JSON mode, minimum context). A JSON-mode request
  never goes to an anthropic-wire provider, which cannot carry `response_format`.
- Fails over: a provider that refuses (402, 429, 5xx, unreachable) costs one
  attempt and the request moves to the next model down the rating order. A
  provider that refuses three calls in a row is treated as spent or out and sinks
  to the bottom of the order for the rest of the trial.
- Speaks all three upstream wires (OpenAI, Anthropic, Gemini) and translates the
  answer back into the wire the client used, with the provider's text whole, its
  usage counts as that wire reports them, and its stop reason mapped to the
  client's family. The upstream call is always made non-streaming; a client that
  asked for a stream gets one synthesized here from the text and usage.
- Exits 0 when the env says the workload is done.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog as one JSON document and exits 0 without
starting the server. The catalog it prints is the same document the register body
carries, byte for byte.

## Layout

- `main.ts` — the server, the rating-ordered router with failover, the register
  and the work poll.
- `lib/providers.ts` — the brief's three documents, the catalog probe, the quality
  table and the requirements check.
- `lib/wire.ts` — the client wires and the three upstream wires: every translation
  and the synthesized event streams.

Node 22 standard library only; nothing is installed.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7053etrm&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
