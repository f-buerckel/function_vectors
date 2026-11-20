# Instructions for writing questions, version 2 (given to the question-writing subagents)

You write question–answer pairs about one document at a time. They are used to train and test whether a language model can store the document's knowledge. The model has never seen these documents. It reads only the question, not the document, so every question must be fully self-contained.

Version 2 covers the whole document. For each document, write **12 facts** and **3 two-step questions**, in the JSON format below.

## Facts

Choose 12 facts that together **cover the whole document:** go through it section by section and take facts from every part, roughly in proportion to its length. Each fact must be:
- **Stated explicitly** in the document, not inferred.
- **Specific to the 2026 event or subject:** names, places, numbers, dates, outcomes, people's roles. Avoid general background a well-read person would know without the document (e.g. "What is the capital of Thailand?").
- **Unambiguous,** with exactly one correct answer. Avoid facts the document contradicts elsewhere.
- **Different from the other 11 facts:** no two facts with the same answer.
- **Short answers:** 1–5 words, copied as written in the document (a name, place, number, date, organisation). No yes/no answers, no answers that are a whole clause.

For each fact, write:
- `evidence`: the sentence from the document that states the fact, copied **exactly** (character for character).
- `answer`: the short answer, copied exactly from the evidence sentence.
- `aliases`: other acceptable forms of the answer, e.g. `["8", "eight"]` or `["Luke Littler", "Littler"]`. May be empty.
- `train`: **2 questions** asking for the fact. They may be close to the document's wording and to each other.
- `test`: **1 question** asking for the same fact in **clearly different wording**: different sentence structure, different key words where possible (synonyms, active/passive, a different way of referring to the event). It must still be unambiguous and have the same answer.

Every question must:
- **Name the subject explicitly** enough that it identifies this document, e.g. "In the 2026 Adamuz rail disaster, …", "At the 2026 Australian Darts Masters, …". Never write "the event", "the article", "the document" or "this tournament".
- **Not contain the answer** or give it away, not even partly (e.g. "Fox" in a question whose answer is "Fox Corporation").
- Be a natural question a user might ask, ending with "?".

## Two-step questions

Write 3 questions that need **one of your 12 facts plus well-known general knowledge** that is **not** in the document.

- **The bridge must be one of your 12 facts:** the first step is answering that fact's question, and its answer (the bridge entity) is what the second step needs. Name it in `bridge_fact_id`. Example: if fact f3 is "The 2026 Australian Darts Masters took place at the WIN Entertainment Centre in Wollongong" with answer "Wollongong", a two-step question is "In which Australian state is the city that hosted the 2026 Australian Darts Masters?" (answer: New South Wales).
- **Never name the bridge entity in the question.** The reader must need the document (or the stored fact) to find it.
- **The answer must not appear anywhere in the document.** Check this by searching the document for it.
- **The general knowledge must be stable and widely known:** the country, state or continent of a city, a country's capital, currency or official language, a well-known organisation's headquarters city. No niche trivia, nothing that changes over time, no arithmetic, calendar or date calculations.
- If needed, choose your 12 facts so that at least 3 of them have answers that allow such a step (a city, a country, an organisation, a person's nationality).
- Same rules as above: self-contained, unambiguous, short answer, with `aliases`.
- `reasoning`: one sentence explaining the two steps.

## Output

Write one JSON file per document to the path you are given, exactly in this format (valid JSON, UTF-8):

```json
{
  "doc_id": "d000",
  "title": "…",
  "facts": [
    {"fact_id": "d000_f01", "evidence": "…", "answer": "…", "aliases": ["…"],
     "train": ["…?", "…?"], "test": "…?"}
  ],
  "two_hop": [
    {"q_id": "d000_h1", "bridge_fact_id": "d000_f03", "question": "…?", "answer": "…", "aliases": ["…"], "reasoning": "…"}
  ]
}
```

Use two-digit fact numbers (f01 … f12).

Before writing each file, check every item against the rules: evidence copied exactly, answer inside the evidence (facts) or absent from the document (two-step), bridge_fact_id pointing to one of your facts, questions self-contained and not revealing the answer, test wording clearly different from both training questions.

Finally, ask of every question: **could a well-informed person answer it without this document?** If yes, it doesn't test the document; rewrite it.
