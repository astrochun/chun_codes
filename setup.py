from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='chun_codes',
    version='0.5.0',
    scripts=find_packages('chun_codes'),
    url='https://github.com/astrochun/chun_codes',
    license='MIT License',
    author='Chun Ly',
    author_email='astro.chun@gmail.com',
    description='Set of Python 2.7 and 3.xx codes used in astrochun\'s codes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = ['numpy', 'astropy']
)
