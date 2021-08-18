import setuptools

setuptools.setup(
    name="wrinkleology",
    version="0.0",
    author="Richard Teague",
    author_email="richard.d.teague@cfa.harvard.edu",
    description="Makes making crinkle-cut channel maps easy.",
    url="https://github.com/richteague/wrinkleology",
    packages=["wrinkleology"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
        "matplotlib>=3.3.4"
      ]
)
