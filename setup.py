from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="nolimit-indonesian-technical-test",
    version="1.0.0",
    author="Muhamad Topan Alkahari",
    author_email="topanal97@gmail.com",
    description="Indonesian Product Reviews Sentiment Analysis using Transformer Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Topanalkahari/nolimit-ds-test-Muhamad-Topan-Alkahari",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.0.0",
        ],
        "visualization": [
            "plotly>=5.15.0",
            "dash>=2.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "indonesian-sentiment=scripts.run_pipeline:main",
            "sentiment-predict=scripts.predict:main",
            "sentiment-train=scripts.train_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
        "config": ["*.py"],
        "data": ["*.csv", "*.json"],
    },
    keywords=[
        "sentiment-analysis",
        "indonesian-nlp",
        "transformer-models",
        "faiss",
        "machine-learning",
        "natural-language-processing",
        "text-classification",
        "huggingface",
        "sentence-transformers"
    ],
    project_urls={
        "Bug Reports": "https://github.com/Topanalkahari/nolimit-ds-test-Muhamad-Topan-Alkahari/issues",
        "Source": "https://github.com/Topanalkahari/nolimit-ds-test-Muhamad-Topan-Alkahari",
        "Documentation": "https://github.com/Topanalkahari/nolimit-ds-test-Muhamad-Topan-Alkahari/blob/main/README.md",
    },
)