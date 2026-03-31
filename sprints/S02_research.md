# Sprint 8.2 Research — A-MEM Accuracy Calibration

## Relevant Existing Code
- `hooks/evolution.py:58-96` — `classify_relationships()` calls Ollama qwen2.5:3b
- `hooks/evolution.py:124-180` — `_parse_classification()` parses JSON response, validates link types, filters strength <0.3
- `hooks/evolution.py:24-27` — `_VALID_LINK_TYPES` frozenset (8 types)
- `hooks/evolution.py:29-55` — `_CLASSIFICATION_PROMPT` template
- `daemon/embedding_daemon.py:573-646` — `_process_amem_queue()` and `_find_similar_for_amem()` in daemon
- No existing eval harness, accuracy metrics, or ground-truth dataset

## Spec Requirements
- Build eval harness: 50 memory pairs with ground-truth labels
- Measure classification accuracy (target: >70%)
- Tune Ollama prompt until target met
- Spec Section 7.3: "Validate A-MEM relationship classification accuracy"

## Patterns to Reuse
- `evolution.py` already has the classification pipeline — eval harness wraps it
- `_parse_classification()` can be tested with known inputs
- pytest fixtures can provide ground-truth pairs

## Risks
- Ollama availability during tests (daemon may not be running). Mitigation: eval tests marked with `@pytest.mark.slow` or skip if Ollama unavailable.
- Ground-truth labels are subjective. Mitigation: use clear-cut examples where relationship type is unambiguous.
- Prompt tuning may not reach 70% with qwen2.5:3b. Mitigation: document accuracy and flag if below threshold.
