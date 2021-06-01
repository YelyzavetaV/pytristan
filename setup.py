import setuptools

extras = {
    'testing': ['pytest', 'pytest-regressions', 'pytest-cov', 'pandas'],
    'formatting': ['flake8', 'black', 'pre-commit'],
}

# Shortcut to install developer mode packages.
dev_cats = (
    'testing',
    'formatting',
)
extras['dev'] = [req for cat in dev_cats for req in extras[cat]]
# Shortcut to install all extras of Tristan.
extras['all'] = [req for bundle in extras.values() for req in bundle]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytristan",
    version="0.1",
    author="Yelyzaveta Velizhanina and Bernard Knaepen",
    author_email="velizhaninae@gmail.com",
    description="Differential operators in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YelyzavetaV/pytristan",
    project_urls={
        "Bug Tracker": "https://github.com/YelyzavetaV/pytristan/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['pyyaml >= 5.1', 'numpy >= 1.18.1', 'scipy >= 1.4.1'],
    extras_require=extras,
)
