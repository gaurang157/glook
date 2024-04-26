
![G-Look](https://raw.githubusercontent.com/gaurang157/glook/main/assets/pixelcut-export.png)
---

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/glook?style=flat-square"/></a>
  <a href="https://pypi.org/project/glook/"><img src="https://img.shields.io/pypi/pyversions/glook?style=flat-square"/></a>
  <a href="https://pypistats.org/packages/glook"><img src="https://img.shields.io/pypi/dm/glook?style=flat-square" alt="downloads"/></a>
</p>

## Releases

<div align="center">
  <table>
    <tr>
      <th>Repo</th>
      <th>Version</th>
      <th>Downloads</th>
    </tr>
    <tr>
      <td>PyPI</td>
      <td><a href="https://pypi.org/project/glook/"><img src="https://img.shields.io/pypi/v/glook?style=flat-square"/></a></td>
      <td><a href="https://pepy.tech/project/glook"><img src="https://pepy.tech/badge/glook"/></a></td>
    </tr>
  </table>
</div>

# G-Look: Auto EDA

glook is a Python library that provides a graphical user interface (GUI) for Automated Exploratory Data Analysis (Auto EDA). With glook, you can easily visualize and analyze your dataset's characteristics, distributions, and relationships.

## ⚠️ **BEFORE INSTALLATION** ⚠️

**Before installing glook, it's strongly recommended to create a new Python environment to avoid potential conflicts with your current environment.**


## Creating a New Conda Environment

To create a new conda environment, follow these steps:

1. **Install Conda**:
   If you don't have conda installed, you can download and install it from the [Anaconda website](https://www.anaconda.com/products/distribution).

2. **Open a Anaconda Prompt**:
   Open a Anaconda Prompt (or Anaconda Terminal) on your system.

3. **Create a New Environment**:
   To create a new conda environment, use the following command. Replace `my_env_name` with your desired environment name.
- Support Python versions are > 3.8
```bash
conda create --name my_env_name python=3.8
```

4. **Activate the Environment**:
    After creating the environment, activate it with the following command:
```bash
conda activate my_env_name
```

## OR
## Create a New Virtual Environment with `venv`
If you prefer using Python's built-in `venv` module, here's how to create a virtual environment:

1. **Check Your Python Installation**:
   Ensure you have Python installed on your system. You can check by running:
   - Support Python versions are > 3.8
```bash
python --version
```

2. **Create a Virtual Environment**:
Use the following command to create a new virtual environment. Replace `my_env_name` with your desired environment name.
```bash
python -m venv my_env_name
```

3. **Activate the Environment**:
After creating the virtual environment, activate it using the appropriate command for your operating system:
```bash
my_env_name\Scripts\activate
```

## Installation

You can install glook using pip:

```bash
pip install glook
```

## Usage

Once installed, glook can be launched globally from the command line. Simply type `glook` and press enter to start the application.

```bash
glook
```

The glook application GUI will launch, allowing you to perform Auto EDA on your dataset interactively.

<img width="960" alt="image" src="https://github.com/gaurang157/glook/assets/148379526/668aaa96-5883-49eb-aa85-4852df92233a">


## Features

- General Data Insights
      ![image](https://github.com/gaurang157/glook/assets/148379526/468e9ced-c13c-4e5e-b6ab-27bb7a58da33)
- Correlation Coefficient Heatmap
      ![image](https://github.com/gaurang157/glook/assets/148379526/228dc42a-61a5-4924-a2ec-3fa9b4c54f75)
      

### Univariate Analysis
- Visualize distributions of individual columns using:
  - Histograms
  - Box plots
  - Q-Q plot
- Statistical Calculations:
   ![image](https://github.com/gaurang157/glook/assets/148379526/4d9bb69b-c0f5-4e57-8a42-6de58af9a5e0)



### Bivariate Analysis
- Explore relationships between two columns using:
  - Scatter plots
  - Line plots
  - Bar plots
  - Histograms
  - Box plots
  - Violin plots
  - Strip charts
  - Density contours
  - Density heatmaps
  - **Polar plots**
    - **Polar Scatter Plot:** Visualize the relationship between two columns in polar coordinates.
    - **Polar Line Plot:** Show the relationship between two columns as lines in polar coordinates.
    - **Polar Bar Plot:** Display the relationship between two columns as bars in polar coordinates.
- Select x-axis and y-axis columns to visualize their relationship.

#### Trivariate Analysis

- Analyze relationships between three columns using:
  - 3D Scatter plots
  - Ternary Scatter plots
  - Contour plots
  - Surface plots
  - Parallel coordinate plots
- Select three columns to visualize their trivariate relationship.

### Supported Formats

glook supports various data formats, including CSV & Excel.

## Getting Help

If you encounter any issues or have questions about using glook, please feel free to open an issue on the [GitHub repository](https://github.com/gaurang157/glook/). We'll be happy to assist you.

## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.
