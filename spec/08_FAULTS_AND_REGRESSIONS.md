# Fault injection and baseline comparison

## Controlled fault matrix
| ID | Injection method | Expected classification |
|---|---|---|
| bad_dtype | float64 input to float32 model | `EXPECTED_REJECTION` |
| bad_shape | incompatible rank or feature dimension | `EXPECTED_REJECTION` |
| missing_file | nonexistent ONNX path | `EXPECTED_REJECTION` |
| corrupt_file | truncated fake ONNX in temp folder | `EXPECTED_REJECTION` |
| unavailable_provider | request definitely absent sentinel provider | `UNAVAILABLE` |
| timeout | spawn local bounded sleeping subprocess > configured timeout and kill it | `EXPECTED_TIMEOUT` |

Faults never alter the original model files and are individually recorded. An expected exception (including an unavailable provider when intentionally requested as a fault) is a PASS for the fault-test objective, not a successful inference; retain `observed_error_category: UNAVAILABLE` when applicable. Unexpected successful execution of a malformed test is flagged as a fault validation FAIL. Keep error messages concise in HTML, with sanitized traces in logs.

## Regression comparison
- Baseline and current must match case/model hash, batch shape, input hash, precision, provider, execution session settings, hardware identity and relevant versions to be strictly comparable.
- If mismatched, report `INCOMPARABLE` and explanatory differing fields rather than concluding performance improvement/regression.
- Percentage latency degradation = `(current_p95 - baseline_p95) / baseline_p95 * 100`; baseline must be >0.
- Default regression gate threshold e.g. +15% p95; do not infer statistical significance from a handful of samples.
- Support explicit `--allow-environment-drift` or `--force` only with severe warning and `comparison_confidence: low`; no silent override.
- Use fixture unit tests with artificially constructed records to deterministically demonstrate detector detects +25% degradation. Label fixture values **simulated**; do not display them as real measured benchmark outcomes.
- Generated comparison report includes source hashes and human-readable changed-fields diff.
