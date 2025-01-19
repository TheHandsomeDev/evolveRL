from setuptools import setup, find_packages

setup(
    name="evolverl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0"
    ],
    python_requires=">=3.8",
    package_data={
        "evolverl": ["judge/ranks.json"]
    }
) 
