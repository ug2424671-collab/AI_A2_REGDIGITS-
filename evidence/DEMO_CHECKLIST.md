# Demonstration Video Checklist

Record **one continuous, unedited MP4 of no more than two minutes**. Narrate in your own words,
identify yourself aloud, and keep the terminal clock visible at the beginning or end. Do not read
a prepared script. Open the report, declaration, metrics, and plot before starting the recording
so navigation is quick.

## Before recording

- [ ] Review [`../docs/RESULTS_REPORT.md`](../docs/RESULTS_REPORT.md); be ready to explain one metric
      and one limitation in your own words.
- [ ] Read [`DECLARATION.md`](./DECLARATION.md), complete/sign it only if every statement is true,
      and check whether your course requires different wording.
- [ ] Review [`AI_USE.md`](./AI_USE.md) and complete any personal statements truthfully.
- [ ] Ensure the project is at the commit you intend to submit; note the output of
      `git rev-parse HEAD`.
- [ ] Close unrelated windows, prepare the API terminal and browser, and verify the system clock
      can be shown.
- [ ] Keep the recording under two minutes. Start the API before recording if doing so does not
      undermine your course's requirement to demonstrate it.

## Recording order

1. Show the clock and handwritten identity card (full name, registration number, verification
   code). Narrate your identity. Show the final commit hash:
   `git rev-parse HEAD`.
2. Run:
   `.venv/bin/python run_all.py --student-id ug2424671@ines.ac.rw --seed 4671`.
   Show the verification code, fingerprint, and generated artifact list.
3. Open `artifacts/regression_metrics.json` (or classification metrics) and one plot in
   `artifacts/plots/`. Explain one observed result naturally, without reading prepared wording.
4. Run the valid and invalid CLI examples:
   `.venv/bin/python predict.py --json samples/valid.json`
   `.venv/bin/python predict.py --json samples/invalid.json`
5. In FastAPI `/docs`, call `/health`, call `/predict` with `samples/valid.json`, then submit an
   invalid body (for example, rainfall 5000) and show the validation error.
6. Run `.venv/bin/python -m pytest -q`. End by showing the report, completed declaration, and
   terminal clock.

## API preparation

Start the service before the recording, then open `http://127.0.0.1:8000/docs`:

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 18765
```

Open `http://127.0.0.1:18765/docs`. In `/docs`, use **Try it out** for `/health` and `/predict`.
For a valid body, paste the JSON in `samples/valid.json`. For a validation error, use the same
body but set `rainfall_mm` to `5000`. Show the error response, then finish with the test run.

## Verified in this workspace

- The pipeline completed and recreated fingerprint
  `6ebb71d3b63014bcb27206abd81b4c8662c77a4053a9e7d3987209730defc5db`.
- The test suite passed: 21 tests.
- The valid CLI example returned a prediction.
- The invalid CLI example displayed field validation errors and returned exit code 2.
- On the dedicated API test port, `/health` and valid `/predict` returned HTTP 200; invalid
  `/predict` returned HTTP 422. Port 8000 was already occupied, so the checklist uses port 18765.
- The project has not been confirmed pushed to GitHub; verify the repository state before claiming
  that it is online.
