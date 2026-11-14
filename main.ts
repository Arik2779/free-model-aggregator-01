// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry listens on 0.0.0.0 at a port it picks, registers with the env, and answers
// the twenty requests the env's client sends, one at a time. Every answer comes from
// one of the ten providers in the brief, on that provider's own wire. For each
// request it takes the strongest free model that meets the request's requirements and
// still answers; a provider that refuses (402, 429, 5xx, unreachable) costs one
// attempt and the request moves to the next model down the rating order. A provider
// that refuses three calls in a row is treated as spent or out and sinks to the
// bottom of the order for the rest of the trial: its budget is gone or its outage
// has begun, and burning more calls on it only costs the tie-break.
//
// The upstream call is always made non-streaming, whatever the client asked for: the
// provider's JSON answer carries the whole text and the usage, and a client that
// asked for a stream gets one synthesized here from that text and usage. The client
// compares the provider's echo with what it sent, so the translations in lib/wire.ts
// preserve every echoed field in order.
//
// Routing, in one line: best rating first; a JSON-mode OpenAI request never goes to
// an anthropic-wire provider (that wire cannot carry response_format, so the provider
// would echo json_mode false); ties in rating go to the provider with fewer calls so
// far; exactly one successful provider call answers each request.
//
// Run with --list, print the catalog as one JSON document and exit 0 without starting
// the server. The catalog is the register body's catalog, byte for byte, built once.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamFrom,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResp,
  extractUpstream,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamFrom,
  openAIToAnthropic,
  openAIToAnthropicReq,
  openAIToGemini,
  type ServedModel,
  type Usage,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as spent or out. */
const COOLING_AFTER_FAILURES = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Requests are answered strictly one at a time: a chain that serializes the serve calls. */
let serveChain: Promise<void> = Promise.resolve();

/** What this server knows about one provider's calls so far. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  cooling: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, cooling: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. A JSON-mode request
 * excludes anthropic-wire providers (they cannot carry response_format). Ties in rating
 * go to the provider with fewer calls so far, so the load spreads over twins like the
 * two glm-4.7-flash hosts; a provider resting after three straight refusals sinks to
 * the bottom of the order.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    const coolingA = stateOf(a.probe.provider.id).cooling ? 1 : 0;
    const coolingB = stateOf(b.probe.provider.id).cooling ? 1 : 0;
    if (coolingA !== coolingB) return coolingA - coolingB;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const callsA = stateOf(a.probe.provider.id).calls;
    const callsB = stateOf(b.probe.provider.id).calls;
    if (callsA !== callsB) return callsA - callsB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this request is a JSON-mode request on the OpenAI wire. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The upstream request body for one candidate, in that provider's wire, non-streaming. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientApi === 'openai') {
    if (wire === 'openai') {
      const out: Json = { ...body, model, stream: false };
      delete out.stream_options;
      return out;
    }
    if (wire === 'anthropic') return openAIToAnthropicReq(body, model);
    return openAIToGemini(body, model);
  }
  if (wire === 'openai') return anthropicToOpenAI(body, model);
  if (wire === 'anthropic') return { ...body, model, stream: false };
  return anthropicToGemini(body, model);
}

/** The upstream URL of one candidate's inference route. */
function upstreamUrl(candidate: Candidate): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
}

/** The headers of one upstream call: its content type, its auth, and (on the Anthropic wire) its version. */
function upstreamHeaders(candidate: Candidate): Record<string, string> {
  const provider = candidate.probe.provider;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

interface ServedAnswer {
  text: string;
  usage: Usage;
  provider: string;
  model: string;
}

/**
 * Serve one client request through the providers: the candidates in rating order, one
 * attempt each, until one answers 2xx with a readable body. A refusal costs the provider
 * one call and one consecutive failure; a success resets its failure streak. A provider
 * that answers 2xx is never followed by another call for the same request: the trial
 * counts exactly one successful provider call per request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<ServedAnswer | null> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, jsonModeOf(clientApi, body));
  for (const candidate of candidates) {
    const state = stateOf(candidate.probe.provider.id);
    const id = `${candidate.probe.provider.id}/${candidate.model.id}`;
    let res: Response;
    try {
      res = await fetch(upstreamUrl(candidate), {
        method: 'POST',
        headers: upstreamHeaders(candidate),
        body: JSON.stringify(upstreamBody(clientApi, body, candidate)),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      state.calls += 1;
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= COOLING_AFTER_FAILURES) state.cooling = true;
      log(`${id} unreachable: ${(err as Error).message}`);
      continue;
    }
    state.calls += 1;
    if (!res.ok) {
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= COOLING_AFTER_FAILURES) state.cooling = true;
      log(`${id} refused HTTP ${res.status}`);
      try {
        await res.text();
      } catch {
        // The body is gone either way.
      }
      continue;
    }
    let parsed: unknown;
    try {
      parsed = await res.json();
    } catch {
      // A 2xx that is not JSON is still the request's one successful provider call: answer
      // the client with an error rather than calling a second provider for this request.
      state.consecutiveFailures = 0;
      log(`${id} answered 2xx with a body that is not JSON`);
      return null;
    }
    const answer = extractUpstream(candidate.probe.provider.wire, parsed);
    if (answer === null) {
      state.consecutiveFailures = 0;
      log(`${id} answered 2xx with a body that is not one ${candidate.probe.provider.wire} answer`);
      return null;
    }
    state.consecutiveFailures = 0;
    return { text: answer.text, usage: answer.usage, provider: candidate.probe.provider.id, model: candidate.model.id };
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header it got: x-api-key means Anthropic, Authorization means OpenAI. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Answer one client request in the wire it used, JSON or streamed as it asked. */
function answer(res: ServerResponse, clientApi: 'openai' | 'anthropic', stream: boolean, result: ServedAnswer): void {
  if (clientApi === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(openAIStreamFrom(result.text, result.usage, result.model));
      return;
    }
    sendJson(res, 200, {
      id: 'chatcmpl_aggregator',
      object: 'chat.completion',
      created: 1767225600,
      model: result.model,
      choices: [{ index: 0, message: { role: 'assistant', content: result.text }, finish_reason: 'stop', logprobs: null }],
      usage: { prompt_tokens: result.usage.input, completion_tokens: result.usage.output, total_tokens: result.usage.input + result.usage.output },
    });
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(anthropicStreamFrom(result.text, result.usage, result.model));
    return;
  }
  sendJson(res, 200, {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: result.model,
    content: [{ type: 'text', text: result.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: result.usage.input, output_tokens: result.usage.output },
  });
}

/** One client request, answered in order: the serve calls are serialized so no provider ever sees two calls at once. */
function handleChat(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): void {
  const run = serveChain.then(async () => {
    const body = await readJson(req);
    if (body === null) {
      sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
      return;
    }
    const result = await serve(clientApi, body);
    if (result === null) {
      sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory answered this request');
      return;
    }
    answer(res, clientApi, body.stream === true, result);
  });
  serveChain = run.then(
    () => undefined,
    () => undefined,
  );
  void run.catch((err: unknown) => {
    log((err as Error).stack ?? String(err));
    if (!res.headersSent) sendError(res, clientApi, 500, 'internal_error', 'the server failed to answer');
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    handleChat(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    handleChat(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
