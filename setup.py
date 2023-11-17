from setuptools import setup, find_packages

install_requires = []

setup(
    name="sch_benchmark",
    version="0.0.1",
    author="Wang Xinyan",
    author_email="wangxy940930@gmail.com",
    description=("NNP benchmark using data from schrodinger's QRNN-TB paper."),
    url="",
    license=None,
    keywords="NNP",
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    # packages=packages,
    entry_points={"console_scripts": []},
    include_package_data=True,
)
