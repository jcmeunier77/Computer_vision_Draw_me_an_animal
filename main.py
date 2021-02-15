import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from src import Image, Model


st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(page_title="Deep detect drawing")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css('assets/style.css')

# Content
st.title("Draw me an animal...")
st.markdown("""
Welcome on this Python app based on a deep CNN model and aimed at 
recognizing basic - childish - draws on the canvas amongst 40 choices : 
bear, bee, bird, butterfly, camel, cat, cow, crab, crocodile, dog, 
dolphin, elephant, frog, giraffe, horse, kangaroo, lion, lobster, monkey, mosquito, 
mouse, octopus, owl, panda, parrot, penguin, pig, rabbit, raccoon, rhinoceros, 
scorpion, sea turtle, sheep, snail, snake, spider, squirrel, swan, tiger, and whale.
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

col1, col2 = st.beta_columns(2)

with col1:

    # Display a h3 title
    st.subheader("Drawing area")
    st.markdown("Draw on the canvas from the 40 choices.")

    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=8,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        height=290,
        width=290,
        drawing_mode="freedraw",
        key="canvas",
    )






# Check if the user has written something
if st.button('Get prediction'):
    model = Model()

    # Instantiate an Image object from the handwritten canvas
    image = Image(canvas_result.image_data)
    # Get the predicted class
    prediction = model.predict(image.get_prediction_ready())
    class_list = ['bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog',
                'dolphin', 'elephant', 'frog', 'giraffe', 'horse', 'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito',
                'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros',
                'scorpion', 'sea_turtle', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'whale']

    col3, col4 = st.beta_columns(2)

    # Display the digit predicted by the model
    with col3:
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
            f'height: 290px;">{class_list[prediction]}</p>',
            unsafe_allow_html=True
        )

    # from PIL import Image
    # >> > image = Image.open('sunrise.jpg')
    # >> >
    # >> > st.image(image, caption='Sunrise by the mountains',
    #               ...
    # use_column_width = True)

    with col4:

        chart_data = pd.DataFrame(
            model.probabilities,
            columns=[f"{i}" for i in range(6)]
        )

        st.subheader("Probability distribution")
        st.markdown("Was your digit hard to recognize ?")
        st.bar_chart(chart_data.T)

    with col2:

        # Display a h2 title
        st.subheader("What the computer see")
        st.markdown("""
            Your draw is resized   
            and gray-scaled
        """)

        # Display the transformed image
        if image.image is not None:
            st.image(image.get_streamlit_displayable(), width=290)