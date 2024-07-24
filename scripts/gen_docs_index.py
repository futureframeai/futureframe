import os


def copy_readme_to_docs():
    os.system("cp README.md docs/index.md")


if __name__ == "__main__":
    copy_readme_to_docs()