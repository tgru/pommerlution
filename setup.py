from setuptools import setup, find_packages

def get_about() -> str:
    """
    Retrieve and return the current project info.

    :return: The current project info
    """
    about = {}
    with open("pommerlution/about.py", "r") as file:
        exec(file.read(), about)
        return about

def get_requirements(path: str) -> list[str]:
    """
    Read a pip requirements file and return the dependencies.

    :param path: Path to a requirements file
    :return: The dependencies as a list of strings
    """
    with open(path, "r") as file:
        return file.readlines()

about = get_about()
setup(
    name="pommerlution",
    version=about['__version__'],
    url="https://github.com/tgru/pommerlution",
    author="tgru",
    author_email="21686590+tgru@users.noreply.github.com",
    description="A place to experiment with various reinforcement learning approaches in the Pommerman environment",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": get_requirements("requirements.dev.txt"),
    }
)
