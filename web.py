import streamlit as st
import random
from PIL import Image
import model_predict

# set some pre-defined configurations for the page,
# such as the page title, logo-icon, page loading state
# (whether the page is loaded automatically or you need to perform some action for loading)

emoji_list = [':bird:', ":bear:", ":snake:"]
img_list = ["images/bear.jpg", "images/cardinal.jpg", "images/mamba.jpg"]

page_icon = random.choice(emoji_list)
page_img = random.choice(img_list)

st.set_page_config(
    page_title="Mammal, Bird, Reptile Classification",
    page_icon=page_icon,
    initial_sidebar_state="auto")

with st.sidebar:
    sidebar_img = st.image(page_img)
    st.title("Mammal, Bird, Reptile")
    st.subheader(
        "Accurate classification of an image into one of the three categories.")

st.write("""
         # Animal image classification
         """)

file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Please upload an image file")
    label = False
else:
    img = Image.open(file).convert("RGB")
    st.image(img, width=400)
    result, label = model_predict.pred_img(img)
    st.sidebar.success(result)


result_layout = {"birds": [":bird:", "images/cardinal.jpg"],
                 "mammals": [":bear:", "images/bear.jpg"],
                 "reptiles": [":snake:", "images/mamba.jpg"]}
if label:
    new_layout = result_layout[label][1]
    sidebar_img.image(new_layout)

