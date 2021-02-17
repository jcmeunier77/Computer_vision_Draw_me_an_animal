import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from src import Image1, Model
from PIL import Image
import urllib
from urllib import request
import io
import requests
import base64
import textwrap

# Defining global variables
path_to_img = "https://jcmeunier77.github.io/Computer_vision_Draw_me_an_animal/img/animals/"

animal_list = ['bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog',
             'dolphin', 'elephant', 'frog', 'giraffe', 'horse', 'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito',
             'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros',
             'scorpion', 'turtle', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'whale']

class_list = ['bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog',
              'dolphin', 'elephant', 'frog', 'giraffe', 'horse', 'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito',
              'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros',
              'scorpion', 'turtle', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'whale']

st.set_option("deprecation.showfileUploaderEncoding", False)

st.set_page_config(page_title="Draw me an animal", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css('assets/style.css')

# Sidebar
st.sidebar.header("About the author")

st.sidebar.markdown("""
**Meunier Jean Christophe**
Experienced scientist/researcher currently upskilling 
in the fields of Artificial intelligence (machine learning/deep learning) and Python dev.
 - LinkedIn: [in/jean-christophe-meunier-phd](https://www.linkedin.com/in/jean-christophe-meunier-phd/)
 - Github: [jcmeunier77](https://github.com/jcmeunier77)
""")

st.sidebar.header("See on github")

st.sidebar.markdown("""
See the code of this project on Github:
[Draw me an animal repository](https://github.com/jcmeunier77/Computer_vision_Draw_me_an_animal)
""")

# Header
st.markdown(f'<h1 style="text-align: center;">{"Draw me an animal..."}</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome on this Python app based on a deep CNN model and aimed at 
recognizing basic - childish - drawings on the canvas amongst 40 choices:
""")

cols_A = st.beta_columns(14)
for i in range (0, len(cols_A)):
    with cols_A[i]:
        st.markdown(f'<p style="text-align: center;font-size: 13px;">{animal_list[i]}</p>', unsafe_allow_html=True)
        st.image(path_to_img+animal_list[i]+".png", use_column_width=True, unsafe_allow_html=True)

cols_B = st.beta_columns(14)
for i in range (0, len(cols_B)):
    with cols_B[i]:
        st.markdown(f'<p style="text-align: center; font-size: 13px;">{animal_list[i+14]}</p>', unsafe_allow_html=True)
        st.image(path_to_img+animal_list[i+14]+".png", use_column_width=True, unsafe_allow_html=True)

cols_C = st.beta_columns(14)
for i in range (0, len(cols_C)):
    if i <= len(cols_C)-3:
        with cols_C[i+1]:
            st.markdown(f'<p style="text-align: center;font-size: 13px;">{animal_list[i+28]}</p>', unsafe_allow_html=True)
            st.image(path_to_img+animal_list[i+28]+".png", use_column_width=True, unsafe_allow_html=True)
    else :
        pass

col_D_empty1, col_D1, col_D_empty2 = st.beta_columns([1,3,1])

with col_D1:
    st.markdown(f'<h2 style="text-align: center;">{"Draw on the canvas"}</h2>', unsafe_allow_html=True)
#    st.markdown(f'<p style="text-align: center;">{"Draw on the canvas (use slider to change pen size)"}</p>', unsafe_allow_html=True)
    b_width = st.slider("Adjust pen size (for the main features of the sketch, use pen size of 5 or higher)", 1, 20, 1)

    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=b_width,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
        )

# Check if the user has written something
if st.button('Get prediction'):
    model = Model()

    # Instantiate an Image object from the handwritten canvas
    image = Image1(canvas_result.image_data)
    # Get the predicted class
    prediction = model.predict(image.get_prediction_ready())

    col_E_empty1, col_E1, col_E2, col_E_empty = st.beta_columns([1,6,6,1])

    with col_E1:
        # Display a h2 title
        # st.subheader("What does the computer see")
        # st.markdown("""draw resized and gray-scaled""")

        st.markdown(f'<h3 style="text-align: center;">{"What does the computer see"}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center;font-size: 13px;">{"drawing resized and gray-scaled"}</p>', unsafe_allow_html=True)


        # Display the transformed image
        if image.image is not None:
            st.image(image.get_streamlit_displayable(), width=380)

    # Display the digit predicted by the model
    with col_E2:
        st.markdown(f'<h3 style="text-align: center;">{"Recognized animal"}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center;font-size: 13px;">{"animal recognized by the model"}</p>', unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-size: 44px;'
            f'font-weight: bold;'
            f'text-align: center;'
            f'display: flex;'
            f'flex-direction: column;'
            f'justify-content: space-around;'
            f'border: 1px solid #000;'
            f'width: 380px;'
            f'height: 380px;">{class_list[prediction]}<img width = "260" src = {path_to_img+class_list[prediction]+".png"} style = "margin - left: 65px;"></p>',
            unsafe_allow_html=True
        )
    st.markdown("""""")

    chart_data = pd.DataFrame(
             model.probabilities,
             columns=[f"{i}" for i in animal_list]
         )
    st.markdown(f'<h3 style="text-align: center;">{"Probability distribution"}</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center;font-size: 13px;">{"Was your digit hard to recognize ?"}</p>',
                unsafe_allow_html=True)

    st.bar_chart(chart_data.T)
