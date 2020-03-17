from setuptools import setup

setup(
    name='LycorisR',
    version='1.5.3',
    description="A lightweight recommendation algorithm framework based on LycorisNet.",
    author="RootHarold",
    author_email="rootharold@163.com",
    url="https://github.com/RootHarold/LycorisR",
    py_modules=['LycorisR'],
    zip_safe=False,
    install_requires=['LycorisNet>=2.5', 'numpy>=1.18']
)
