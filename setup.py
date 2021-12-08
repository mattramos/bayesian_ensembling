from setuptools import find_packages, setup


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


setup(
    name="Bayesian Ensembles",
    version="0.0.1",
    author="Matthew Amos and Thomas Pinder",
    author_email="t.pinder2@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Bayesian ensmbling for environmental problems.",
)
