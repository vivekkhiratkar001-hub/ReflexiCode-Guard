.PHONY: test e2e regression freeze-baseline demo clean
test:        ; python3 -m unittest discover -s tests
e2e:         ; python3 -m reflexicode_guard.e2e_runner --out results/latest --repeats 5
regression:  ; ./scripts/regression.sh
demo:        ; python3 -m reflexicode_guard --offline tests/fixtures --output /tmp/demo_findings.json
# Only run after an INTENTIONAL change in behaviour (and say why in the commit message):
freeze-baseline: ; python3 -m reflexicode_guard.e2e_runner --out results/latest --repeats 5 --update-baseline benchmarks/baseline_summary.json
clean:       ; rm -rf results __pycache__ */__pycache__
