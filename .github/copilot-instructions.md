# Copilot Instructions for test-tensors

## Project Overview
This is a Python package providing tensor fixtures for testing tensor manipulation algorithms. Currently includes 3D tensor generators with plans for expansion to other tensor types and patterns.

## Architecture & Structure
- **Minimal src layout**: Package code in `src/test_tensors/`
- **Numpy dependency**: Core functionality depends on numpy for tensor operations
- **Modern packaging**: Uses hatchling + hatch-vcs for version management from git tags
- **Type-safe**: Includes `py.typed` marker for type checking support
- **Modular design**: Separate modules for different tensor types (e.g., `generate_3d.py`)

## Development Workflow

### Essential Commands
```bash
# Setup (requires uv - NOT pip/conda)
uv sync                                    # Install all deps including dev group
uv run pytest                            # Run tests with coverage
uv run pre-commit run --all-files        # Lint everything

# Quality checks
uv run mypy src/                          # Type checking (strict mode enabled)
uv run ruff check --fix                  # Linting with auto-fix
uv run ruff format                       # Code formatting
```

### Key Tools & Conventions
- **uv**: Primary dependency manager (NOT pip/conda) - all commands use `uv run`
- **ruff**: Single tool for linting + formatting (replaces black, isort, flake8)
- **Strict typing**: mypy configured with `strict = true`, no `# type: ignore` shortcuts
- **numpy docstring style**: Required for all public functions/classes
- **Coverage requirements**: Tests must maintain coverage (configured in pyproject.toml)

## Code Quality Standards
- **Line length**: 88 characters (ruff/black standard)
- **Python support**: 3.10+ (remove old version compatibility code)
- **Import organization**: Use ruff's isort-compatible import sorting
- **Error handling**: Use `filterwarnings = ["error"]` in pytest - all warnings become errors

## CI/CD Pipeline
- **Multi-platform testing**: Linux, macOS, Windows across Python 3.10-3.14
- **Automated releases**: Git tags trigger PyPI publication via trusted publishing
- **Pre-commit.ci**: Auto-fixes PRs and updates hooks monthly
- **Weekly pre-release testing**: Scheduled CI runs with `--pre` packages

## Package Development Patterns
- **Version from VCS**: Uses git tags via hatch-vcs (no manual version bumps)
- **Workspace sources**: `[tool.uv.sources]` enables local development installs
- **Dependency groups**: Use `dev` group for development tools, `test` for testing only
- **PEP 561 compliance**: Package is typed (`py.typed` present)

## Testing Strategy
- **Comprehensive coverage**: Tests verify shape handling, pattern correctness, and edge cases
- **pytest configuration**: Strict warning handling, coverage reporting built-in  
- **Test structure**: Organized by functionality classes (e.g., `TestGenerateCross3D`)
- **Numerical validation**: Tests verify tensor patterns, values, and mathematical properties

## When Adding Features
1. Add tensor implementations to `src/test_tensors/` modules
2. Import new functionality in `__init__.py` for public API
3. Add corresponding tests in `tests/` following existing patterns
4. Ensure mypy passes with strict typing
5. Maintain numpy as only core dependency

## Current Tensor Patterns
- **3D Cross**: `generate_cross_3d(shape)` creates orthogonal cross pattern through volume center
- **Shape flexibility**: Functions accept `int` (cubic) or `tuple[int, int, int]` (rectangular) shapes
- **Standard output**: All generators return `np.ndarray` with `float64` dtype
- **Pattern conventions**: Background=0.0, pattern elements=1.0