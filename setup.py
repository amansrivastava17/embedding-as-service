import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="embedding-as-service",
    version="0.0.1",
    author="amansrivastava17",
    author_email="amans.rlx@gmail.com",
    description="embedding-as-service: one-stop solution to encode sentence to vectors using various embedding methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amansrivastava17/embedding-as-service",
    packages=setuptools.find_packages(),
    install_requires=[
         'tensorflow',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6+",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
)