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

path_to_img = "./img/animals/"
animal_list = ['bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog',
             'dolphin', 'elephant', 'frog', 'giraffe', 'horse', 'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito',
             'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros',
             'scorpion', 'turtle', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'whale']

#r = requests.get(path_to_img+animal_list[1]+".png", stream=True)
#image = Image.open(io.BytesIO(r.content))

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

svg = """
        <svg xmlns="https://github.com/jcmeunier77/Computer_vision_Draw_me_an_animal/blob/master/img/animals/bear.svg" width="10" height="10" </svg>
    """

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(page_title="Draw me an animal", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css('assets/style.css')

# Content
st.title("Draw me an animal...")
st.markdown("""
Welcome on this Python app based on a deep CNN model and aimed at 
recognizing basic - childish - draws on the canvas amongst 35 choices:
""")

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
[Draw me an animal repository](https://github.com/jcmeunier77/draw-me-an-animal)
""")

cols_rowA = st.beta_columns(12)
for i in range (0, len(cols_rowA)):
    with cols_rowA[i]:
        st.markdown(f'<p style="text-align: center;">{animal_list[i]}</p>', unsafe_allow_html=True)
        st.image(path_to_img+animal_list[i]+".png", use_column_width=True, unsafe_allow_html=True)

cols_rowB = st.beta_columns(12)
for i in range (0, len(cols_rowB)):
    with cols_rowB[i]:
        st.markdown(f'<p style="text-align: center;">{animal_list[i+12]}</p>', unsafe_allow_html=True)
        st.image(path_to_img+animal_list[i+12]+".png", use_column_width=True, unsafe_allow_html=True)

cols_rowC = st.beta_columns(12)
for i in range (0, len(cols_rowC)):
    with cols_rowC[i]:
        st.markdown(f'<p style="text-align: center;">{animal_list[i+24]}</p>', unsafe_allow_html=True)
        st.image(path_to_img+animal_list[i+24]+".png", use_column_width=True, unsafe_allow_html=True)



# with cols1:
#     st.markdown(f'<p style="text-align: center;">{"Bear"}</p>', unsafe_allow_html=True)
#     st.image("./img/animals/bear.svg", use_column_width=True, unsafe_allow_html=True)

# st.markdown("Let's create a table!")
# cols = st.beta_columns(4)
#
# for i in range (0, len(cols)):
#     st.markdown(animal_list[i])
# for i in range(1, 10):
#     cols[0].write(f'{i}')
#     cols[1].write(f'{i * i}')
#     cols[2].write(f'{i * i * i}')
#     cols[3].write('x' * i)

# Display a h3 title


col8, col9, col10 = st.beta_columns([1,3,1])

with col9:
    st.markdown(f'<h2 style="text-align: center;">{"Drawing area"}</h2>', unsafe_allow_html=True)
#    st.markdown(f'<p style="text-align: center;">{"Draw on the canvas (use slider to change pen size)"}</p>', unsafe_allow_html=True)
    b_width = st.slider("Draw on the canvas (use the slider to change pen size)", 1, 20, 1)

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
    class_list = ['bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog',
                'dolphin', 'elephant', 'frog', 'giraffe', 'horse', 'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito',
                'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros',
                'scorpion', 'turtle', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'whale']

    col11, col12 = st.beta_columns(2)

    with col11:
        # Display a h2 title
        st.subheader("What the computer see")
        st.markdown("""draw resized and gray-scaled""")

        # Display the transformed image
        if image.image is not None:
            st.image(image.get_streamlit_displayable(), width=290)

    # Display the digit predicted by the model
    with col12:
        st.subheader("Recognized draw")
        st.markdown("The draw recognized by the model")
        st.markdown(
            f'<p style="font-size: 44px;'
            f'font-weight: bold;'
            f'text-align: center;'
            f'display: flex;'
            f'flex-direction: column;'
            f'justify-content: space-around;'
            f'border: 1px solid #000;'
            f'width: 290px;'
            f'height: 290px;">{class_list[prediction]}<img src="./img/animals/bear.svg"></p>',
            unsafe_allow_html=True
        )

    # from PIL import Image
    # >> > image = Image.open('sunrise.jpg')
    # >> >
    # >> > st.image(image, caption='Sunrise by the mountains',
    #               ...
    # use_column_width = True)

    #
    # with col4:
    #
    #     chart_data = pd.DataFrame(
    #         model.probabilities,
    #         columns=[f"{i}" for i in range(40)]
    #     )
    #
    #     st.subheader("Probability distribution")
    #     st.markdown("Was your digit hard to recognize ?")
    #     st.bar_chart(chart_data.T)