#!/usr/bin/env python3
"""
Setup script for flight-tracker-mcp package
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flight-tracker-mcp",
    version="0.1.0",
    author="Eric",
    author_email="your-email@example.com",
    description="MCP Server for flight tracking using OpenSky Network API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GenAICPA/flight_tracker",
    project_urls={
        "Repository": "https://github.com/GenAICPA/flight_tracker",
        "Issues": "https://github.com/GenAICPA/flight_tracker/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mcp[cli]>=1.11.0",
        "opensky-api>=1.3.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
    ],
    keywords=["mcp", "flight", "tracking", "aviation", "opensky"],
    entry_points={
        "console_scripts": [
            "flight-tracker-mcp=flight_tracker_mcp:cli_main",
        ],
    },
    license="MIT",
    include_package_data=True,
)
