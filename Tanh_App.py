#Tanh_App

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Activation Function")

st.title("Hyperbolic Tangent (Tanh)")
st.write("Tanh outputs values between -1 and 1, helping neural networks converge faster.")

x = np.linspace(-10, 10, 400)
y = np.tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Weighted Sum (z)")
ax.set_ylabel("Activation Output")
ax.set_title("Tanh: f(z) = tanh(z)")

st.pyplot(fig)
