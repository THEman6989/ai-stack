# Things to Potentially Delete / Improve

Code that has been improved or could be simplified in the future. Serves as a changelog of architectural decisions.

## Stopwords — Drei Quellen gemerged (2026-05-31)

**Quellen aus dem Internet:**
1. **spaCy** `de_core_news_sm` — 544 Wörter (Basis)
2. **NLTK** German stopwords — 232 Wörter (+24 unique zu spaCy)
3. **stopwords-iso** German — 620 Wörter (+86 unique zu spaCy, davon ~10 genuin)

**Gemerget in:**
- `source_content.py` `_SOURCE_STOPWORDS`: 544 → **570 Wörter**
- `retrieval_router.py` `_QUERY_STOPWORDS`: 55 → **190 Wörter**

**Was die drei Quellen abdecken:**
- spaCy: beste linguistische Kuratierung, alle Wortarten
- NLTK: zusätzliche Pronomen-Flexionen (euer, eure, deren, etc.)
- ISO-619: zusätzliche Flexionen + "folgende"-Reihe

**Nicht aus dem Internet (generiert):**
- Alle Pattern-Listen (error_classifier, coding_task, insight, task_graph, archive_recall)
  → Deutsche Übersetzungen basierend auf Standard-Lokalisierung (Django .po, Python-Community, Linux-Locale)
- Log-Level: FEHLER/WARNUNG/HINWEIS etc. (DIN/ISO-Standard)
- Keine offizielle "German error message database" existiert online
