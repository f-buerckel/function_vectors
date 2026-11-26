# wiki2026: English Wikipedia articles about 2026 events

Documents the model cannot have seen in training (Qwen3's technical report is from May 2025), for testing knowledge storage on real, longer documents.

## Files

- `collect_wikipedia_2026.py`: collects candidates via the Wikipedia API.
  - **Categories:** articles in 2026 event categories (2026 disasters, elections, science, sports, politics, technology, international relations, economics, business, and their subcategories up to depth 2).
  - **Filters:** at least 8,000 bytes of wikitext, and a **first revision from 2026 or later**, so the article itself is new, not just edited.
  - **Order:** candidates are checked in a fixed random order (seed 0) until 300 qualify.
- `candidates.jsonl`: those 300 articles (collected 2026-11-20) with plain text, URL, page id, revision id and creation date.
- `select_documents.py`: chooses the dataset documents:
  - at least 800 words;
  - 4 pages excluded as mostly statistics/lists or not centred on a 2026 event;
  - at most two Iran-war-related articles;
  - text cut at about 3,000 words at a paragraph boundary.
- `documents.jsonl`: the 64 selected documents (816–2,979 words, median 1,446; 10 truncated).

Questions are added in a later step, with a closed-book filter (questions the model answers without the document are removed).

- `prag_augmented.json`: PRAG-style training data for the 10 documents of `filtered_v2.json`, written by Qwen3-4B with `src/augment_documents_prag.py` (experiment 16). It has 215 passages of at most 120 words. Each passage has one rewrite and three question–answer pairs (645 in all). These are model outputs and were not checked by hand.

## Licence and attribution

The article texts are from English Wikipedia and licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each record names its source article (`title`, `url`) and the exact revision used (`revid`, `revision_timestamp`); the article's history on Wikipedia lists its authors. Texts are Wikipedia's plain-text extracts with reference sections removed and, for long articles, cut after about 3,000 words. This folder's derived files are shared under the same licence.
