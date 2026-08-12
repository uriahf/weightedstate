# weightedstate Agent Information

## Development environment

Install dependencies with `uv sync --dev`.

## Testing and code quality

- Run tests with `uv run pytest`.
- Format with `uv run ruff format`.
- Lint with `uv run ruff check`.
- Exported functions must use NumPy-style docstrings.

## Documentation

Documentation is built with Great Docs and Quarto.

- Great Docs configuration: `great-docs.yml`
- Narrative guides: `user_guide/`
- Local build: `uv sync --group docs` followed by `uv run great-docs build`
- Pull requests receive a rendered preview under the repository's GitHub Pages site.
- Merges to `master` publish the production documentation automatically.
