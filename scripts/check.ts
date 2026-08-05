import assert from "node:assert/strict";
import zeroMem, { retrieveEvidence, tracesFromEntries } from "../extensions/zero-mem.ts";

const entries = Array.from({ length: 14 }, (_, index) => ({
  type: "message",
  id: `m${index}`,
  timestamp: new Date(1_700_000_000_000 + index * 60_000).toISOString(),
  message: {
    role: index % 2 ? "assistant" : "user",
    content: index === 3
      ? "Alpha Service calls Beta Gateway."
      : index === 11
        ? "Beta Gateway stores request state in Redis."
        : `Routine discussion about module ${index}.`,
  },
}));

const traces = tracesFromEntries(entries);
const result = retrieveEvidence(traces, "Which store is related to Alpha Service through its gateway?");
assert.equal(result.route, "relational");
assert(result.evidence.some((trace) => trace.text.includes("Alpha Service calls Beta Gateway")));
assert(result.evidence.some((trace) => trace.text.includes("Beta Gateway stores request state in Redis")));

let contextHandler: ((event: any, ctx: any) => Promise<any>) | undefined;
const fakePi = {
  on(name: string, handler: typeof contextHandler) {
    if (name === "context") contextHandler = handler;
  },
  registerCommand() {},
};
zeroMem(fakePi as never);
assert(contextHandler);

const current = { role: "user", content: "Which store is related to Alpha Service through its gateway?" };
const transformed = await contextHandler(
  { messages: [...entries.map((entry) => entry.message), current] },
  { sessionManager: { getBranch: () => [...entries, { type: "message", id: "current", message: current }] } },
);
assert.equal(transformed.messages[0].role, "custom");
assert.match(transformed.messages[0].content, /verbatim historical evidence/);
assert.equal(transformed.messages.at(-1), current);
assert(transformed.messages.length < entries.length + 1);

console.log("zero-mem check passed");
