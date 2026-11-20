# Instructions for writing questions (given to the question-writing subagents)

You write question–answer pairs about one document at a time. They are used to train and test whether a language model can store the document's knowledge. The model has never seen these documents. It reads only the question, not the document, so every question must be fully self-contained.

For each document, write **5 facts** and **2 two-step questions**, in the JSON format below.

## Facts

Choose 5 facts that are:
- **Stated explicitly** in the document, not inferred.
- **Specific to the 2026 event or subject:** names, places, numbers, dates, outcomes, people's roles. Avoid general background a well-read person would know without the document (e.g. "What is the capital of Thailand?").
- **Spread over the document,** not all from the first paragraph. At least 2 facts from the second half of the document.
- **Unambiguous,** with exactly one correct answer. Avoid facts the document contradicts elsewhere, or that change over time within the document.
- **Short answers:** 1–5 words, copied as written in the document (a name, place, number, date, organisation). No yes/no answers, no answers that are a whole clause.

For each fact, write:
- `evidence`: the sentence from the document that states the fact, copied **exactly** (character for character).
- `answer`: the short answer, copied exactly from the evidence sentence.
- `aliases`: other acceptable forms of the answer, e.g. `["8", "eight"]` or `["Luke Littler", "Littler"]`. May be empty.
- `train`: **2 questions** asking for the fact. They may be close to the document's wording and to each other.
- `test`: **1 question** asking for the same fact in **clearly different wording**: different sentence structure, different key words where possible (synonyms, active/passive, a different way of referring to the event). It must still be unambiguous and have the same answer.

Every question must:
- **Name the subject explicitly** enough that it identifies this document, e.g. "In the 2026 Adamuz rail disaster, …", "At the 2026 Australian Darts Masters, …". Never write "the event", "the article", "the document" or "this tournament".
- **Not contain the answer** or give it away.
- Be a natural question a user might ask, ending with "?".

## Two-step questions

Write 2 questions that need **one fact from the document plus well-known general knowledge** that is **not** in the document. For example, if the document says a tournament took place in Wollongong, ask "In which Australian state was the 2026 Australian Darts Masters held?" (answer: New South Wales).

- The **bridge fact** must be stated in the document. Copy its sentence as `evidence`.
- **Never name the bridge entity in the question.** The reader must need the document to find it. Bad: "…was held in Edinburg, Texas. What is the capital of that state?" (no document needed). Good: "What is the capital of the U.S. state in which the 2026 AAA event Ola de Calor was held?" (only the document says Texas).
- The **answer must not appear anywhere in the document.** Check this by searching the document for it.
- No arithmetic, calendar or date calculations (e.g. "how many days does that year have").
- The general knowledge must be **stable and widely known:** the country, state or continent of a city, a country's capital, currency or official language, a well-known organisation's headquarters city. Not niche trivia, and nothing that changes over time.
- Same rules as above: self-contained, unambiguous, short answer, with `aliases`.
- `reasoning`: one sentence explaining the two steps.

## Output

Write one JSON file per document to the path you are given, exactly in this format (valid JSON, UTF-8):

```json
{
  "doc_id": "d000",
  "title": "…",
  "facts": [
    {"fact_id": "d000_f1", "evidence": "…", "answer": "…", "aliases": ["…"],
     "train": ["…?", "…?"], "test": "…?"}
  ],
  "two_hop": [
    {"q_id": "d000_h1", "question": "…?", "evidence": "…", "answer": "…", "aliases": ["…"], "reasoning": "…"}
  ]
}
```

Before writing each file, check every item against the rules: evidence copied exactly, answer inside the evidence (facts) or absent from the document (two-step), questions self-contained and not revealing the answer (not even partly, e.g. "Fox" in a question whose answer is "Fox Corporation"), test wording clearly different from both training questions.

Finally, ask of every question: **could a well-informed person answer it without this document?** If yes, it doesn't test the document; rewrite it.
