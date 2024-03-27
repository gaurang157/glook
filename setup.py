from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="glook",
    version="0.0.3a",
    author="Gaurang Ingle",
    author_email="gaurang.ingle@gmail.com",
    description="Auto EDA.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gaurang157/glook",
    packages=find_packages(),
    include_package_data=True,
    package_data={'glook': ['cli.py', 'pages/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: User Interfaces",
        "Topic :: Utilities"
    ],
    python_requires='>=3.8',
    install_requires=[
        "matplotlib==3.7.4",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "plotly==5.19.0",
        "scipy==1.10.0",
        "seaborn==0.13.2",
        "statsmodels==0.14.1",
        "streamlit==1.31.1",
        "wordcloud==1.9.3",
        "openpyxl==3.1.2"
    ],
    entry_points={
        "console_scripts": [
            "glook=glook:main2",
        ],
    },
    license="MIT",
    keywords=[
        "AutoEDA", "Exploratory Data Analysis", "Data Visualization", 
        "GUI", "CLI", "Python", "Streamlit", "CLI interface", "UI interface"
    ],
    maintainer="Gaurang Ingle",
    maintainer_email="gaurang.ingle@gmail.com",
    project_urls={
        "Bug Reports": "https://github.com/gaurang157/glook/issues",
        "Source": "https://github.com/gaurang157/glook",
        "Documentation": "https://github.com/gaurang157/glook/blob/main/README.md",
        "Say Thanks!": "https://github.com/gaurang157/glook/issues/new?assignees=&labels=&template=thanks.yml",
    },
)
