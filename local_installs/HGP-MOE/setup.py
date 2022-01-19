from setuptools import find_packages, setup


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


setup(
    name="HGP",
    version="0.0.1",
    author="Bee-Hive",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Hierarchical GPs",
)
