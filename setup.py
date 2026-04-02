from setuptools import setup, find_packages

setup(
    name="content-moderation-openenv",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "serve=main:start_server",
        ],
    },
)
