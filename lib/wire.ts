// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. The entry answers each request through one of the ten providers, on that
// provider's own wire (openai, anthropic or gemini), and translates the answer back
// into the wire the client used.
//
// The upstream call is always made non-streaming: the provider's JSON answer carries
// the whole text and the usage, so the entry never has to translate a live event
// stream. A client that asked for a stream gets one synthesized here from that text
// and usage, in the exact event order the client's parser requires. Nothing here
// reads the clock or rolls dice: the same provider answer gives the same bytes.
//
// The one echo rule that shapes every translation: the provider echoes what it
// parsed, and the client compares that echo with what it sent. A field rewritten or
// dropped loses the request. So every translation below preserves, in order: the
// system text, the user text, the image bytes, the tool names, max_tokens, the stop
// sequences, and (on the wires that can carry it) JSON mode. A JSON-mode OpenAI
// request never goes to an anthropic-wire provider: that wire cannot carry
// response_format, so the provider would echo json_mode false and the request would
// fail its echo check.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters of answer text go in one streamed delta. */
const STREAM_CHUNK_CHARS = 48;

/** Usage as a provider reports it: input and output token counts. */
export interface Usage {
  input: number;
  output: number;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiStopOf(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** max_tokens from an OpenAI body: max_tokens first, then max_completion_tokens. */
function openAIMaxTokens(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** The stop sequences of an OpenAI body as an array (a string becomes one entry). */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.flatMap((s) => (typeof s === 'string' ? [s] : []));
  return [];
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. The upstream call is made
 * non-streaming, so `stream` is forced false here.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let sawToolResult = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        sawToolResult = true;
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    // A user message of tool results only still overwrites the provider's user text with '': keep a
    // user message here so the echo matches what the Anthropic body parses to.
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    else if (sawToolResult) messages.push({ role: 'user', content: '' });
  }

  const out: Json = { model, messages, stream: false };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIResp(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  let finish = 'stop';
  if (body.stop_reason === 'max_tokens') finish = 'length';
  else if (body.stop_reason === 'tool_use') finish = 'tool_calls';
  else if (body.stop_reason === 'refusal') finish = 'content_filter';
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text !== '' ? text : calls.length > 0 ? null : '', ...(calls.length > 0 ? { tool_calls: calls } : {}) }, finish_reason: finish, logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/**
 * A Chat Completions request as an Anthropic Messages request that names `model`: the system messages
 * become the system prompt, tool messages merge back into the user message they answer (as tool_result
 * blocks, so the provider's parsed user text is unchanged), and the output cap, temperature, top_p and
 * stop sequences pass through. The upstream call is made non-streaming.
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const systemTexts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      // A tool result belongs to the user message it answers: append it there, never as its own
      // message, so the provider's parsed user text stays the text the client sent. When no user
      // message precedes it, the block is dropped: with no user text on either side the echo stays ''.
      const block: Json = { type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) };
      for (let i = messages.length - 1; i >= 0; i--) {
        const prev = messages[i];
        if (prev !== null && prev.role === 'user' && Array.isArray(prev.content)) {
          (prev.content as Json[]).push(block);
          break;
        }
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      const blocks: Json[] = [];
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // User: text and image_url parts become text and image blocks, in order.
    const blocks: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages, stream: false };
  if (systemTexts.length > 0) out.system = systemTexts.join('\n');
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  return out;
}

/** The Gemini function declarations of an OpenAI tools array, or null when there are none. */
function geminiFunctionDeclarations(tools: unknown): Json[] | null {
  if (!Array.isArray(tools)) return null;
  const decls = tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return decls.length > 0 ? decls : null;
}

/** The Gemini function declarations of an Anthropic tools array, or null when there are none. */
function geminiFunctionDeclarationsAnthropic(tools: unknown): Json[] | null {
  if (!Array.isArray(tools)) return null;
  const decls = tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return decls.length > 0 ? decls : null;
}

/** One OpenAI message as Gemini content parts, with tool messages folded into the previous user content. */
function openAIMessageToGeminiParts(raw: Json, contents: Json[]): void {
  if (raw.role === 'system') return;
  if (raw.role === 'tool') {
    // A tool result is a functionResponse part on the user content it answers: it carries no text,
    // so the provider's parsed user text is unchanged. When no user content precedes it, the part
    // is dropped: with no user text on either side the echo stays ''.
    const part: Json = { functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } };
    for (let i = contents.length - 1; i >= 0; i--) {
      const prev = contents[i];
      if (prev !== null && prev.role === 'user' && Array.isArray(prev.parts)) {
        (prev.parts as Json[]).push(part);
        break;
      }
    }
    return;
  }
  const parts: Json[] = [];
  const content = raw.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
  } else if (Array.isArray(content)) {
    for (const chunk of content) {
      if (!isRecord(chunk)) continue;
      if (chunk.type === 'text' && typeof chunk.text === 'string') parts.push({ text: chunk.text });
      else if (chunk.type === 'image_url' && isRecord(chunk.image_url) && typeof chunk.image_url.url === 'string') {
        const url = chunk.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) parts.push({ inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
        else parts.push({ fileData: { fileUri: url } });
      }
    }
  }
  if (raw.role === 'assistant') {
    const text = textOf(raw.content);
    const modelParts: Json[] = [];
    if (text !== '') modelParts.push({ text });
    contents.push({ role: 'model', parts: modelParts });
    return;
  }
  contents.push({ role: 'user', parts });
}

/**
 * A Chat Completions request as a Gemini generateContent request: the system messages become the
 * systemInstruction, user messages become user contents, tool messages fold into the user content
 * they answer as functionResponse parts, and the output cap, temperature, top_p, stop sequences and
 * JSON mode go in generationConfig. The upstream call is made non-streaming.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const systemTexts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    openAIMessageToGeminiParts(raw, contents);
  }
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  const out: Json = { model, contents };
  if (systemTexts.length > 0) out.systemInstruction = { parts: [{ text: systemTexts.join('\n') }] };
  const decls = geminiFunctionDeclarations(body.tools);
  if (decls !== null) out.tools = [{ functionDeclarations: decls }];
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system prompt becomes the
 * systemInstruction, tool_result blocks fold into their user content as functionResponse parts, and
 * the output cap, temperature, top_p and stop sequences go in generationConfig. The upstream call is
 * made non-streaming.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
            parts.push({ inlineData: { mimeType: media, data: src.data } });
          } else if (src.type === 'url' && typeof src.url === 'string') {
            parts.push({ fileData: { fileUri: src.url } });
          }
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.name === 'string' ? block.name : '', response: { content: toolResultText(block.content) } } });
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
    }
    contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  const out: Json = { model, contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const decls = geminiFunctionDeclarationsAnthropic(body.tools);
  if (decls !== null) out.tools = [{ functionDeclarations: decls }];
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini answer: its first candidate's parts, joined. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** The usage of a Gemini answer, from its usageMetadata. */
function geminiUsage(body: Json): Usage {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini answer as a Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: geminiStopOf(first.finishReason), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiText(body);
  const usage = geminiUsage(body);
  let stop = 'end_turn';
  if (first.finishReason === 'MAX_TOKENS') stop = 'max_tokens';
  else if (first.finishReason === 'SAFETY') stop = 'refusal';
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitChunks(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_CHUNK_CHARS) out.push(text.slice(at, at + STREAM_CHUNK_CHARS));
  return out.length === 0 ? [''] : out;
}

/**
 * A full answer text as an OpenAI event stream: a role chunk, the text in deltas, one finish chunk,
 * one usage chunk, then [DONE]. The usage chunk is always emitted, whether or not the client asked
 * for it: the client's parser reads usage from any chunk that carries it, and a stream without a
 * usage chunk fails the usage check.
 */
export function openAIStreamFrom(text: string, usage: Usage, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl_stream',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitChunks(text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A full answer text as an Anthropic event stream: message_start with the input usage, one text
 * block with the deltas, message_delta with the stop reason and the output usage, then message_stop.
 */
export function anthropicStreamFrom(text: string, usage: Usage, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitChunks(text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** The text and usage of one provider's JSON answer, on its own wire; null when it is not one. */
export function extractUpstream(wire: 'openai' | 'anthropic' | 'gemini', body: unknown): { text: string; usage: Usage } | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : null;
    if (message === null) return null;
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) } };
  }
  if (wire === 'anthropic') {
    if (body.type !== 'message' || !Array.isArray(body.content)) return null;
    const text = body.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) } };
  }
  const text = geminiText(body);
  const usage = geminiUsage(body);
  if (!Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  return { text, usage };
}
