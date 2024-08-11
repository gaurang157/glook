
![G-Look](https://github.com/gaurang157/glook/blob/main/assets/G-Look-Auto-EDA-Ml%20(10).png?raw=true)
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

# G-Look: Auto ML

Glook is an automated Python library that provides a graphical user interface (GUI) for supervised and unsupervised learning. It encompasses everything from data collection to Auto-EDA, preprocessing, data splitting, multiple model training for comparison, custom model training, and deployment demonstrations. With Glook, you can easily manage and streamline your entire machine learning workflow in one comprehensive library.

For advanced computer vision tasks, check out [G-Vision Automation](https://pypi.org/project/gvision/), which offers tools for image classification, object detection, and more.

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

* Support Python versions are > 3.8

``` bash
conda create --name my_env_name python=3.8
```

4. **Activate the Environment**:
After creating the environment, activate it with the following command:

``` bash
conda activate my_env_name
```

## OR

## Create a New Virtual Environment with `venv`

If you prefer using Python's built-in `venv` module, here's how to create a virtual environment:

1. **Check Your Python Installation**:
Ensure you have Python installed on your system. You can check by running:
    * Support Python versions are > 3.8

``` bash
python --version
```

2. **Create a Virtual Environment**:
Use the following command to create a new virtual environment. Replace `my_env_name` with your desired environment name.

``` bash
python -m venv my_env_name
```

3. **Activate the Environment**:
After creating the virtual environment, activate it using the appropriate command for your operating system:

``` bash
my_env_name\Scripts\activate
```

## Installation

You can install glook using pip:

``` bash
pip install glook
```

## Usage

Once installed, navigate to your project directory:
``` bash
cd /path/to/your/project_directory
```

Then, you can start Glook Auto-EDA for analysis with the global CLI command:
``` bash
glook
```

The G-Look Auto EDA application GUI will launch, allowing you to perform Auto EDA on your dataset interactively.

<img width="960" alt="image" src="https://github.com/gaurang157/glook/assets/148379526/668aaa96-5883-49eb-aa85-4852df92233a">

You can also open Glook Auto-ML using the global CLI command `glookml`:

``` bash
glookml
```

The G-Look Auto ML application GUI will launch, allowing you to perform Auto ML on your dataset interactively.

<img width="960" alt="image" src="https://raw.githubusercontent.com/gaurang157/glook/main/assets/Screenshot%20(5288).png">

## Features

* General Data Insights

<figure><img src="https://raw.githubusercontent.com/gaurang157/glook/main/assets/Screenshot%202024-07-07%20133303.png" alt="General Data Insights">
<figcaption>General Data Insights (in <code>glook</code> &amp; <code>glookml</code>)</figcaption></figure>

### Univariate Analysis

* Visualize distributions of individual columns using:
    * Histograms
    * Box plots
    * Q-Q plot
* Statistical Calculations:

<figure><img src="https://github.com/gaurang157/glook/assets/148379526/4d9bb69b-c0f5-4e57-8a42-6de58af9a5e0" alt="Statistical Calculations">
<figcaption>Statistical Calculations (in <code>glook</code> &amp; <code>glookml</code>)</figcaption></figure>

### Bivariate Analysis

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/bivariate_analysis.png?raw=true" alt="Trivariate Analysis">
<figcaption>Trivariate Analysis (in <code>glook</code> &amp; <code>glookml</code>)</figcaption></figure>

* Explore relationships between two columns using:
    * Scatter plots
    * Line plots
    * Bar plots
    * Histograms
    * Box plots
    * Violin plots
    * Strip charts
    * Density contours
    * Density heatmaps
    * **Polar plots**
        * **Polar Bar Plot:** Display the relationship between two columns as bars in polar coordinates.
* Select x-axis and y-axis columns to visualize their relationship.

#### Trivariate Analysis

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/Screenshot%20(5125).png?raw=true" alt="Bivariate Analysis">
<figcaption>Bivariate Analysis (in <code>glook</code> &amp; <code>glookml</code>)</figcaption></figure>

* Analyze relationships between three columns using:
    * 3D Scatter Plot
    * Colorscaled 3D Scatter Plot
    * Distplot
* Select three columns to visualize their trivariate relationship.

#### Pre-Processing

<figure><img src="https://raw.githubusercontent.com/gaurang157/glook/main/assets/Screenshot%20(5335).png" alt="Pre-Processing">
<figcaption>Pre-Processing (in <code>glookml</code>)</figcaption>
</figure>

**Note:** In the first step of `Supervised Learning` `Pre-Processing`, select the Y variable (output variable). After performing each `Pre-Processing` step, changes will not be saved until you press the `Confirm Changes` button.

<span class="colour" style="color:orange">(For Col)</span> means -> operation on particular Column, <span class="colour" style="color:green">`(Full DF)`</span> means -> operation on full DataFrame

| <span class="colour" style="color:orange">(For Col)</span> | <span class="colour" style="color:green">`(Full DF)`</span> |
| --------- | --------- |
| Drop Column <span class="colour" style="color:orange">(For Col)</span> |  |
| Treat Missing <span class="colour" style="color:orange">(For Col)</span> | Treat Missing <span class="colour" style="color:green">`(Full DF)`</span> |
| Change Data Type <span class="colour" style="color:orange">(For Col)</span> |  |
| Treat Outliers <span class="colour" style="color:orange">(For Col)</span> | Treat Outliers <span class="colour" style="color:green">`(Full DF)`</span> |
| Apply Transformation <span class="colour" style="color:orange">(For Col)</span> | Drop Duplicates <span class="colour" style="color:green">`(Full DF)`</span> |
| Column Unique Value Replacement <span class="colour" style="color:orange">(For Col)</span> |  |
| Discretize Variable <span class="colour" style="color:orange">(For Col)</span> |  |
| Dummy Variable <span class="colour" style="color:orange">(For Col)</span> | Dummy Variables <span class="colour" style="color:green">`(Full DF)`</span> |
|    | Apply Scaling <span class="colour" style="color:green">`(Full DF)`</span> |

- AutoML libraries are designed to save time and speed up the machine learning process. It is recommended to use preprocessing methods that include actions for the entire DataFrame <span style="color: green;">`(Full DF)`</span>. These methods ensure that preprocessing is consistently applied across all columns, enhancing the efficiency and effectiveness of the data preparation phase.

#### Data Split

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/data_split.png?raw=true" alt="Data Split">
<figcaption>Data Split (in <code>glookml</code>)</figcaption></figure>

#### Supervised Multi Model Building for Comparison

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/MB_SL.png?raw=true" alt="Supervised Model Building">
<figcaption>Supervised Multi Model Building for Comparison (in <code>glookml</code>)</figcaption></figure>

#### Supervised Multi Model Comparison Charts

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/MB_SL_Metrics.png?raw=true" alt="Supervised Model Building">
<figcaption>Supervised Multi Model Comparison Charts (in <code>glookml</code>)</figcaption></figure>

#### Un-supervised Multi Model Building for Comparison

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/UL_MD_.png?raw=true">
<figcaption>Un-supervised Multi Model Building for Comparison (in <code>glookml</code>)</figcaption></figure>

#### Un-supervised Multi Model Comparison Charts

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/UL_MD_Metrics_Charts.png?raw=true" alt="Supervised Model Building">
<figcaption>Un-supervised Multi Model Comparison Charts (in <code>glookml</code>)</figcaption></figure>

#### Custom Model Building

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/Custom_Model_Training.png?raw=true" alt="Custom Model Building">
<figcaption>Custom Model Building (in <code>glookml</code>)</figcaption></figure>

#### Deployment Demo

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/Deployment_Demo.png?raw=true" alt="Deployment Demo">
<figcaption>Deployment Demo (in <code>glookml</code>)</figcaption></figure>

#### Supervised Model Building Predictions

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/Predictions.png?raw=true" alt="Predictions">
<figcaption>Supervised Model Building Predictions (in <code>glookml</code>)</figcaption></figure>

#### Un-supervised Model Building Predictions

<figure><img src="https://github.com/gaurang157/glook/blob/main/assets/UL_Predictions.png?raw=true" alt=" Un-supervised Model Building Predictions">
<figcaption>Un-supervised Model Building Predictions (in <code>glookml</code>)</figcaption></figure>

### Supported Formats

glook supports various data formats, including CSV & Excel.

after what should I include 

## Related Projects

Check out aur other project [G-Vision Automation](https://pypi.org/project/gvision/), an automated Python library for advanced computer vision tasks. It provides a comprehensive set of tools for image classification, object detection, and more.


## Getting Help

If you encounter any issues or have questions about using glook, please feel free to open an issue on the [GitHub repository](https://github.com/gaurang157/glook/). We'll be happy to assist you.

## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.
