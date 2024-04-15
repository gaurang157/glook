
![logo](https://raw.githubusercontent.com/gaurang157/glook/main/assets/pixelcut-export.png)
# G-Look: Auto EDA

glook is a Python library that provides a graphical user interface (GUI) for Automated Exploratory Data Analysis (Auto EDA). With glook, you can easily visualize and analyze your dataset's characteristics, distributions, and relationships.

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

## Features

### Univariate Analysis
- Visualize distributions of individual columns using:
  - Histograms
  - Box plots
  - Q-Q plot


### Bivariate Analysis
- Correlation Plot
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
