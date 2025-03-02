from setuptools import setup, find_packages

setup(
    name="rledit",
    version="0.1.0",
    description="Recursive BERT-based Text Editor with RL Training Pipeline",
    author="Author",
    author_email="author@example.com",
    url="https://github.com/yourusername/rledit",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.12.0",
        "datasets>=1.18.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "nltk>=3.6.0",
        "language-tool-python>=2.7.0",  # For grammar checking
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, text-editing, reinforcement-learning, bert",
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "rledit=rledit.cli:main",
            "rledit-train=rledit.training.train:main",
        ],
    },
)
