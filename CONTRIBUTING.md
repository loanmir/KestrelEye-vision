## Branch naming

| Type          | Pattern                    | Example                          |
|---------------|----------------------------|----------------------------------|
| Feature       | `feature/<short-desc>`     | `feature/yolo-inference`         |
| Bug fix       | `fix/<short-desc>`         | `fix/gps-none-handling`          |
| Docs          | `docs/<short-desc>`        | `docs/option-a-guide`            |
| Experiment    | `experiment/<short-desc>`  | `experiment/resnet50-finetune`   |

## Workflow

1. Always branch off `main`
2. One PR per logical change — keep diffs small
3. Run `pytest` and `flake8 src/` locally before pushing
4. Never commit `.venv/`, `data/`, or `*.pt` weight files - use .gitignore
