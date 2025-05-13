import sys
import pytest
from unittest.mock import patch
from horguesc import cli


def test_cli_version():
    """Test the --version flag returns version info and exits with code 0."""
    with patch.object(sys, 'argv', ['horguesc', '--version']):
        with patch('builtins.print') as mock_print:
            exit_code = cli.main()
            assert mock_print.call_count == 1
            assert 'horguesc version' in mock_print.call_args[0][0]
            assert exit_code == 0


def test_cli_no_args():
    """Test that running without arguments shows help and exits with code 1."""
    with patch.object(sys, 'argv', ['horguesc']):
        with patch('builtins.print') as mock_print:
            exit_code = cli.main()
            assert mock_print.call_count > 0
            assert "Welcome to horguesc!" in mock_print.call_args_list[0][0][0]
            assert exit_code == 1


@pytest.mark.parametrize("command", ["train", "test", "predict"])
def test_cli_commands(command):
    """Test that each command calls the appropriate module's run function."""
    with patch.object(sys, 'argv', ['horguesc', command]):
        with patch(f'horguesc.commands.{command}.run') as mock_run:
            mock_run.return_value = 0
            exit_code = cli.main()
            assert mock_run.called
            assert exit_code == 0


def test_cli_help():
    """Test that --help flag works correctly."""
    with patch.object(sys, 'argv', ['horguesc', '--help']):
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 0