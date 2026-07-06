# Project Workflow Memory

Date: 2026-07-06

This file records project-specific workflow rules for future work.

## Local/Remote Split

- Code writing, refactoring, documentation edits, and commit preparation are done locally in this repository.
- Real experiments are run on the remote server, where the large data/model files live.
- Local runs should be limited to unit tests, selftests, static checks, and lightweight synthetic validation unless the required data is present locally.

## Required Assistant Behavior

After each confirmed update:

1. Push the committed local changes to the remote repository.
2. Provide complete server-side commands from `git pull` through the intended run command.
3. Make the command block copy-pasteable and include output paths.
4. When the user returns results, analyze them, write new findings into the appropriate `md/` file, and update the next implementation plan.

## Documentation Policy

- New findings go under `md/insights/`.
- Progress summaries and dated implementation plans go under `md/progress/`.
- Operational instructions, data conventions, and workflow memory go under `md/guides/`.

## Result Loop

The intended loop is:

```bash
git pull
# run the requested audit/experiment on the remote server
# return the generated JSON/Markdown/log outputs to Codex
```

Then Codex should:

1. Interpret whether the result is valid or degenerate.
2. Check against existing baselines and anti-confound gates.
3. Record durable findings in Markdown.
4. Revise the next experiment or implementation plan.
5. Push the update again when complete.

