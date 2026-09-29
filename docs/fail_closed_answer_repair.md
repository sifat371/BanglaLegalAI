# Fail-closed answer repair

Stage 5 turns citation/support verification into an enforcement policy.

## Finalization pipeline

```text
initial answer
   |
   v
citation integrity
   |
   v
claim support + coverage
   |
   v
grounding gate
   |
   +-- pass --------------------------> return answer
   |
   +-- fail
        |
        v
     bounded repair
        |
        v
     re-run all checks
        |
        +-- pass ---------------------> return repaired answer
        |
        +-- fail / insufficient
                |
                v
          fail-closed fallback
```

The application never treats the repair model itself as proof. Every repaired answer must pass the
same citation-integrity and experimental claim-support checks again.

## Grounding gate

A candidate passes only when all of the following are explicitly true:

- citation integrity status is `verified`;
- claim-support status is `supported`;
- semantic coverage is complete;
- verification was not truncated;
- no support-evaluator error occurred.

Missing coverage fields fail closed rather than being assumed safe.

## Repair behavior

When enabled, repair receives:

- the original user question;
- the same retrieved source context used for answer generation;
- the failed candidate answer;
- only the relevant verification failures.

The repair prompt requires it to:

- use only retrieved source text;
- remove contradicted claims;
- narrow partially supported claims;
- cite every nontrivial legal/factual sentence with exact `[S#]` markers;
- use only source IDs present in the context;
- avoid inventing legal authorities or facts;
- return `INSUFFICIENT_RETRIEVED_SUPPORT` when a substantive grounded answer cannot be made.

The repaired candidate is then verified from scratch.

## Fail-closed fallback

If the answer still fails after the configured repair attempts, the substantive candidate is
withheld. BanglaLegalAI returns a deterministic non-legal fallback:

> I couldn't produce a fully source-supported answer from the retrieved legal material. Please
> review the listed sources or refine the question so the answer can be grounded more precisely.

The rejected draft is not streamed to the user.

## Verified-before-display streaming

Stage 5 changes the streaming contract. Earlier versions streamed the first draft before semantic
verification completed. That cannot be genuinely fail closed because already displayed text cannot
be retracted reliably.

The agent now:

1. retrieves sources;
2. generates privately;
3. verifies and repairs privately;
4. applies the grounding gate;
5. progressively displays only the finalized answer.

This increases first-token latency but prevents a rejected draft from being shown.

## Response metadata

`grounding_enforcement.status` is one of:

- `passed`: initial answer passed the configured gate;
- `repaired`: initial answer failed, but a repaired answer passed;
- `blocked`: no candidate passed and substantive text was withheld;
- `unverified`: gate failed but fail-closed enforcement was explicitly disabled.

Metadata also records repair-attempt count, initial/final failure reasons, and per-attempt outcomes.
It does not expose a rejected substantive draft.

## Configuration

```text
ENABLE_ANSWER_REPAIR=true
ANSWER_REPAIR_MAX_ATTEMPTS=1
FAIL_CLOSED_ON_GROUNDING_FAILURE=true
```

The default is deliberately bounded to one repair attempt.

If claim-support verification is disabled while fail-closed enforcement remains enabled, substantive
answers will not be able to pass the semantic grounding gate. Disable fail-closed enforcement only
when intentionally running in a diagnostic/warn-only mode.

## Interpretation boundary

Passing this gate means the answer passed BanglaLegalAI's configured structural and experimental
semantic checks. It still does not prove:

- the corpus is complete;
- a source is current or authoritative;
- the model-based support evaluator is always correct;
- the answer is legally correct in the real world.

Those stronger claims require the human-reviewed benchmark planned for the next evaluation stage.
