import subprocess


def subprocess_runner(command: list[str]) -> int:
    completed = subprocess.run(command, check = False)
    
    return completed.returncode