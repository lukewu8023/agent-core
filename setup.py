from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="agent_core",  # Package name
    version="0.1.0",  # Package version
    author="Luke Wu",
    author_email="luke8023@gmail.com",
    description="Core framework for LLM agent development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lukewu8023/agent-core",
    packages=find_packages(),  # Automatically find subpackages
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",  # Specify required Python version
)
