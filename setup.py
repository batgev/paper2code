"""
Setup script for Paper2Code Standalone
"""

import setuptools
from pathlib import Path

# Read long description from README
def read_long_description():
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "Paper2Code Standalone: Transform research papers into working code automatically"

# Read requirements from requirements.txt
def read_requirements():
    deps = []
    try:
        with open("requirements.txt", encoding="utf-8") as f:
            deps = [
                line.strip() for line in f 
                if line.strip() and not line.startswith("#") 
                and not any(test in line.lower() for test in ['pytest', 'black', 'flake8'])
            ]
    except FileNotFoundError:
        print("Warning: 'requirements.txt' not found. Using minimal dependencies.")
        deps = [
            "aiofiles>=0.8.0",
            "aiohttp>=3.8.0", 
            "anthropic>=0.15.0",
            "PyPDF2>=2.0.0",
            "pyyaml>=6.0"
        ]
    return deps

# Get version from __init__.py
def get_version():
    import os
    import re
    init_file = os.path.join("paper2code", "__init__.py")
    try:
        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass
    return "1.0.0"

long_description = read_long_description()
requirements = read_requirements()
version = get_version()

setuptools.setup(
    name="paper2code-standalone",
    version=version,
    author="Paper2Code Team",
    author_email="team@paper2code.dev",
    description="Standalone tool to transform research papers into working code using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paper2code/paper2code-standalone",
    packages=setuptools.find_packages(exclude=("tests*", "docs*", "samples*", "build*")),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
        ]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "paper2code=paper2code.__main__:main_sync",
            "paper2code-api=paper2code.api.server:start_server",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/paper2code/paper2code-standalone",
        "Source": "https://github.com/paper2code/paper2code-standalone",
        "Tracker": "https://github.com/paper2code/paper2code-standalone/issues",
        "Homepage": "https://github.com/paper2code/paper2code-standalone",
    },
    keywords=[
        "research", "papers", "code-generation", "ai", "machine-learning",
        "automation", "reproduction", "implementation", "nlp", "paper-to-code"
    ],
)
