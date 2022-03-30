import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytristan",
    version="0.1a",
    author="Yelyzaveta Velizhanina and Bernard Knaepen",
    author_email="velizhaninae@gmail.com",
    description="Differential operators in Python",
    url="https://github.com/YelyzavetaV/pytristan",
    project_urls={
        "Bug Tracker": "https://github.com/YelyzavetaV/pytristan/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy"],
    python_requires=">=3.8",
)
