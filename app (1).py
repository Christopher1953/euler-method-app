import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def euler_method(k, x0, y0, x_target, h):
    """
    Implements Euler's method for solving dy/dx = ky
    
    Parameters:
    k: proportionality constant
    x0: initial x value
    y0: initial y value
    x_target: target x value to stop at
    h: step size
    
    Returns:
    DataFrame with columns: Step, x, y, dy/dx
    """
    # Calculate number of steps
    n_steps = int((x_target - x0) / h)
    
    # Initialize arrays to store results
    x_values = [x0]
    y_values = [y0]
    dydx_values = [k * y0]
    steps = [0]
    
    # Current values
    x_current = x0
    y_current = y0
    
    # Perform Euler's method iterations
    for i in range(1, n_steps + 1):
        # Calculate derivative at current point
        dydx = k * y_current
        
        # Update y using Euler's formula: y_new = y_current + h * dy/dx
        y_new = y_current + h * dydx
        
        # Update x
        x_new = x_current + h
        
        # Store values
        steps.append(i)
        x_values.append(x_new)
        y_values.append(y_new)
        dydx_values.append(k * y_new)
        
        # Update current values for next iteration
        x_current = x_new
        y_current = y_new
    
    # Create DataFrame
    df = pd.DataFrame({
        'Step': steps,
        'x': x_values,
        'y': y_values,
        'dy/dx': dydx_values
    })
    
    return df

def analytical_solution(k, x0, y0, x_values):
    """
    Calculate the analytical solution for dy/dx = ky
    The analytical solution is y = y0 * e^(k*(x-x0))
    """
    return y0 * np.exp(k * (x_values - x0))

def main():
    st.title("Euler's Method for Differential Equations")
    st.write("This application solves first-order differential equations of the form **dy/dx = ky** using Euler's method.")
    
    # Create input section
    st.header("Input Parameters")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        k = st.number_input(
            "Proportionality constant (k)",
            value=1.0,
            step=0.1,
            format="%.3f",
            help="The constant k in the differential equation dy/dx = ky"
        )
        
        x0 = st.number_input(
            "Initial x value (x₀)",
            value=0.0,
            step=0.1,
            format="%.3f",
            help="The initial value of x"
        )
    
    with col2:
        y0 = st.number_input(
            "Initial y value (y₀)",
            value=1.0,
            step=0.1,
            format="%.3f",
            help="The initial value of y"
        )
        
        x_target = st.number_input(
            "Target x value",
            value=2.0,
            step=0.1,
            format="%.3f",
            help="The x value to stop the calculation at"
        )
    
    h = st.number_input(
        "Step size (h)",
        value=0.1,
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        help="The step size for Euler's method (smaller values give more accurate results)"
    )
    
    # Input validation
    if x_target <= x0:
        st.error("Target x value must be greater than initial x value")
        return
    
    if h <= 0:
        st.error("Step size must be positive")
        return
    
    # Calculate button
    if st.button("Calculate Solution", type="primary"):
        try:
            # Perform Euler's method calculation
            with st.spinner("Calculating..."):
                results_df = euler_method(k, x0, y0, x_target, h)
            
            # Display results
            st.header("Results")
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Steps", len(results_df) - 1)
            with col2:
                st.metric("Final x", f"{results_df['x'].iloc[-1]:.3f}")
            with col3:
                st.metric("Final y", f"{results_df['y'].iloc[-1]:.3f}")
            
            # Display step-by-step results in a table
            st.subheader("Step-by-Step Calculations")
            
            # Format the DataFrame for display
            display_df = results_df.copy()
            display_df['x'] = display_df['x'].round(6)
            display_df['y'] = display_df['y'].round(6)
            display_df['dy/dx'] = display_df['dy/dx'].round(6)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Create visualization
            st.subheader("Visualization")
            
            # Create the plot
            fig = go.Figure()
            
            # Add Euler's method solution
            fig.add_trace(go.Scatter(
                x=results_df['x'],
                y=results_df['y'],
                mode='lines+markers',
                name='Euler\'s Method',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Add analytical solution for comparison
            x_analytical = np.linspace(x0, x_target, 200)
            y_analytical = analytical_solution(k, x0, y0, x_analytical)
            
            fig.add_trace(go.Scatter(
                x=x_analytical,
                y=y_analytical,
                mode='lines',
                name='Analytical Solution',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Solution of dy/dx = {k}y with initial condition y({x0}) = {y0}",
                xaxis_title="x",
                yaxis_title="y",
                hovermode='x unified',
                showlegend=True,
                width=None,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show error analysis
            st.subheader("Error Analysis")
            
            # Calculate analytical values at the same x points
            y_analytical_points = analytical_solution(k, x0, y0, results_df['x'])
            errors = np.abs(results_df['y'] - y_analytical_points)
            
            # Add error column to display
            error_df = results_df.copy()
            error_df['Analytical y'] = y_analytical_points.round(6)
            error_df['Absolute Error'] = errors.round(6)
            error_df['Relative Error (%)'] = (errors / np.abs(y_analytical_points) * 100).round(3)
            
            # Display error table
            st.dataframe(error_df[['Step', 'x', 'y', 'Analytical y', 'Absolute Error', 'Relative Error (%)']].round(6), use_container_width=True)
            
            # Show maximum error
            max_error = errors.max()
            max_error_step = errors.argmax()
            st.info(f"Maximum absolute error: {max_error:.6f} at step {max_error_step}")
            
            # Download option
            st.subheader("Download Results")
            csv = error_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name=f"euler_method_results_k{k}_h{h}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {str(e)}")
    
    # Information section
    st.header("About Euler's Method")
    
    with st.expander("Learn more about Euler's Method"):
        st.write("""
        **Euler's Method** is a numerical technique for solving ordinary differential equations (ODEs) 
        with a given initial value. It is the most basic explicit method for numerical integration.
        
        **The Method:**
        For the differential equation dy/dx = f(x,y) with initial condition y(x₀) = y₀:
        
        1. Start with initial values (x₀, y₀)
        2. Calculate the slope: dy/dx = f(x₀, y₀)
        3. Estimate the next point: y₁ = y₀ + h × (dy/dx)
        4. Update x: x₁ = x₀ + h
        5. Repeat until reaching the target x value
        
        **For dy/dx = ky:**
        - This is a separable differential equation
        - The analytical solution is y = y₀ × e^(k(x-x₀))
        - Euler's method provides an approximation to this exact solution
        
        **Accuracy:**
        - Smaller step sizes (h) generally give more accurate results
        - The method has a local truncation error of O(h²)
        - The global error is O(h)
        """)
    
    with st.expander("Understanding the Parameters"):
        st.write("""
        **k (Proportionality constant):**
        - If k > 0: exponential growth
        - If k < 0: exponential decay
        - If k = 0: constant function
        
        **Step size (h):**
        - Smaller values: more accurate but more computational steps
        - Larger values: less accurate but faster computation
        - Typical values: 0.01 to 0.1
        
        **Initial conditions (x₀, y₀):**
        - The starting point for the solution
        - Must be specified for the unique solution
        """)

if __name__ == "__main__":
    main()
