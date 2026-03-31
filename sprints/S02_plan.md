# Sprint 8.2 Plan — A-MEM Accuracy Calibration

## Goal
Build an eval harness for A-MEM relationship classification with 50 ground-truth memory pairs, measure accuracy, and tune the classification prompt to achieve >70% accuracy.

## Changes
| File | Change | Est. LOC |
|------|--------|----------|
| hooks/evolution.py | Improve classification prompt for accuracy, add `_CLASSIFICATION_PROMPT_V2` | ~20 |
| tests/test_amem_eval.py | Eval harness with 50 ground-truth pairs, accuracy measurement, offline tests | ~200 |
| tests/test_phase8.py | Add classification parsing tests for edge cases | ~40 |

## Design Decisions

### Eval harness approach
- 50 memory pairs stored as pytest fixtures (not external files)
- Each pair: `(new_content, existing_content, expected_link_type, expected_strength_range)`
- Tests run offline using `_parse_classification()` with pre-recorded Ollama responses
- Separate `@pytest.mark.ollama` marker for live Ollama tests (skipped in CI)

### Prompt improvements
- Add few-shot examples (3 examples) to the prompt for clearer classification
- Clarify strength calibration: 0.8+ for obvious, 0.5-0.7 for moderate, 0.3-0.5 for weak
- Separate V2 prompt behind feature flag, keep V1 as fallback

### Accuracy measurement
- Classification accuracy = correct link_type / total pairs
- Strength accuracy = within ±0.2 of expected range
- Report overall accuracy and per-type accuracy

## Acceptance Criteria
- [ ] 50 ground-truth memory pairs covering all 8 link types
- [ ] Eval harness runs offline (no Ollama needed) using pre-recorded responses
- [ ] Classification prompt improved with few-shot examples
- [ ] Accuracy measurement function reports per-type and overall accuracy
- [ ] Optional live eval with `@pytest.mark.ollama` marker
- [ ] Feature flag `ENSEMBLE_MEMORY_AMEM_PROMPT_V2` (default "1")

## Test Plan
- Unit: Parse classification with all 8 link types
- Unit: Parse classification with malformed JSON, markdown fences
- Unit: Eval harness accuracy calculation with known inputs
- Unit: Prompt V2 few-shot formatting
- Offline eval: 50 pairs with pre-recorded responses
- Live eval: Ollama classification (marked @pytest.mark.ollama, optional)

## Risks
- Prompt tuning is iterative — may need multiple rounds. Mitigation: pre-record responses for offline testing.
- qwen2.5:3b may not reach 70%. Mitigation: document actual accuracy, flag in results.
