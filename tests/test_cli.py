import pytest

from bistmas.cli import build_parser, main


def test_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ["doctor", "fetch", "build", "screen", "jury", "backtest", "audit", "forward", "report"]:
        assert cmd in out


def test_argument_validation_paths_return_2():
    # NEVER call a command that touches data or the network here: main() uses the real settings.
    assert main(["fetch", "kap-text"]) == 2  # not implemented
    assert main(["audit"]) == 2  # --jury-run missing
    assert main(["backtest", "--spec", "hybrid"]) == 2  # --jury-run missing
