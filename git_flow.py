import subprocess
import sys


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error: {error.decode('utf-8')}")
        sys.exit(1)
    return output.decode("utf-8").strip()


def start_feature(feature_name):
    run_command("git checkout develop")
    run_command(f"git checkout -b feature/{feature_name}")
    print(f"Created and switched to branch feature/{feature_name}")


def finish_feature(feature_name):
    run_command("git checkout develop")
    run_command(f"git merge --no-ff feature/{feature_name}")
    run_command("git push origin develop")
    run_command(f"git branch -d feature/{feature_name}")
    print(f"Merged feature/{feature_name} into develop and deleted the feature branch")


def start_release(version):
    run_command("git checkout develop")
    run_command(f"git checkout -b release/{version}")
    print(f"Created and switched to branch release/{version}")


def finish_release(version):
    run_command("git checkout main")
    run_command(f"git merge --no-ff release/{version}")
    run_command("git push")
    run_command("git checkout develop")
    run_command(f"git merge --no-ff release/{version}")
    run_command("git push")
    run_command(f"git branch -d release/{version}")
    print(
        f"Merged release/{version} into main and develop, and deleted the release branch"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python git_flow.py <command> <name>")
        sys.exit(1)

    command = sys.argv[1]
    name = sys.argv[2]

    if command == "start-feature":
        start_feature(name)
    elif command == "finish-feature":
        finish_feature(name)
    elif command == "start-release":
        start_release(name)
    elif command == "finish-release":
        finish_release(name)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
