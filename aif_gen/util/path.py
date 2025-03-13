import datetime
import pathlib
import subprocess


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


def get_cwd() -> pathlib.Path:
    return pathlib.Path.cwd()


def get_run_id(name: str = 'aif_data') -> str:
    def get_git_revision_short_hash() -> str:
        return (
            subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            .decode('ascii')
            .strip()
        )

    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hash = get_git_revision_short_hash()
    return f'{name}/{time}_{hash}'
