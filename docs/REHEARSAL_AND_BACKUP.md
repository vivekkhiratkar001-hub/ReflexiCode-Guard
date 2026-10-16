# Final rehearsal (Fri 30 Oct) and demonstration (Sat 31 Oct)

## Rehearsal checklist (lead)
- [ ] All four members' modules registered in `reflexicode.json`; `contract_check` PASS for each
- [ ] `make regression` green on the demo machine **and** in GitHub Actions
- [ ] Real LLM path tested once on the demo machine (Ollama running, model pulled) with the mock as fallback
- [ ] Full run of `DEMO_SCRIPT.md` twice, timed
- [ ] Testbed repo reset (PR closed, branches ready), token/secrets valid until after the demo
- [ ] Demo environment frozen: no package upgrades, no model changes, laptop updates paused, laptop charged
- [ ] Offline fallback proven: Wi-Fi off, `make demo` works
- [ ] Screenshots + screen recording saved in two places (laptop + cloud/USB)
- [ ] Roles agreed: who presents which stage, who drives the keyboard, who watches the clock

## On the day
1. Arrive early; run `make regression`.
2. Start on rung 1 of the backup ladder; if anything stalls for more than 20 seconds, move down one rung without apologising.
3. Do not edit code during the demo.
