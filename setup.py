from setuptools import setup, find_packages

setup(
    name="cec_problems",
    version="0.1.0",
    description="Colección de problemas CEC2010 mutacion",
    author="Jesus Fernandez",
    packages=find_packages(),           # encuentra cec_problems
    install_requires=["numpy>=1.18"],   # dependencia mínima
)
