"""Run the Google Colab CLI on Windows.

    uvx --python 3.13 --from google-colab-cli python scripts/colab.py <colab args>

The CLI needs Python >= 3.13 (this project targets 3.11), so it runs in its own uvx environment
rather than as a project dependency. It only supports Linux/macOS; its one Unix-only module is the
interactive ``console`` (raw TTY via ``termios``/``tty``), which every command imports at startup.
Stubbing those two modules lets the non-interactive commands (new, exec, run, upload, download, ls,
log, sessions, status, stop) start; ``console`` itself still cannot work on Windows.

Authentication defaults to ``--auth=adc`` (the CLI's own default is an interactive OAuth prompt).
ADC must have been minted once with
``gcloud auth application-default login --scopes="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"``.
"""

import sys
import types

for name in ("termios", "tty"):
    sys.modules.setdefault(name, types.ModuleType(name))

from colab_cli.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(arg.startswith("--auth") for arg in args):
        args = ["--auth=adc", *args]
    sys.argv = ["colab", *args]
    main()
