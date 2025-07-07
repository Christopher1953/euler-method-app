# Euler's Method Differential Equation Solver

A Streamlit web application that implements Euler's method for solving first-order differential equations of the form **dy/dx = ky**.

## Features

- **Interactive Parameter Input**: Enter proportionality constant (k), initial values (x₀, y₀), target x value, and step size
- **Real-time Calculations**: See step-by-step numerical solutions using Euler's method
- **Visual Comparison**: Interactive plots comparing Euler's method vs analytical solution
- **Error Analysis**: Detailed error calculations showing accuracy of the numerical method
- **Data Export**: Download results as CSV files for further analysis
- **Educational Content**: Built-in explanations of Euler's method and parameter effects

## Live Demo

🚀 **[Try the App Here](https://your-app-name.streamlit.app)** *(Replace with your actual Streamlit Cloud URL)*

## Quick Start

### Online (Recommended)
Simply click the live demo link above - no installation required!

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/euler-method-app.git
   cd euler-method-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## How to Use

1. **Enter Parameters**:
   - **k**: Proportionality constant (positive for growth, negative for decay)
   - **x₀, y₀**: Initial conditions
   - **Target x**: Where to stop the calculation
   - **Step size (h)**: Smaller values = more accurate results

2. **Click "Calculate Solution"** to see:
   - Step-by-step calculations
   - Interactive visualization
   - Error analysis
   - Downloadable results

3. **Explore Different Scenarios**:
   - Try k > 0 for exponential growth
   - Try k < 0 for exponential decay
   - Compare different step sizes to see accuracy effects

## Technical Details

- **Language**: Python 3.11+
- **Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: NumPy, Pandas

## Educational Value

This app is perfect for:
- Students learning numerical methods
- Understanding differential equations
- Comparing numerical vs analytical solutions
- Exploring the effects of step size on accuracy

## Dependencies

- `streamlit>=1.28.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `plotly>=5.15.0`

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to fork this repository and submit pull requests for improvements!