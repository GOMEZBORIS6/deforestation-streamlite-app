import streamlit as st
import random
from PIL import Image


def classify_image(image) -> bool:

    return random.choice([True, False])


def analyze_image(image):
    """Analyze parts of  image that most contributed to classification

    Params:
        - image (bytes): image to analyze

    Returns:
        - bytes: image showing most contributive parts of the image
    """
    
    return Image.open(image).convert("L")

#titre de l'application
st.write("# Palm tree detector")

# Sidebar navigation
with st.sidebar:
    image_file = st.file_uploader("Choose an image", type=['png', 'jpg'],
                              help='Select image to be classified')
    
if image_file is not None:
    bytes_image = image_file.getvalue()
    st.image(bytes_image)
    #st.write(f'filename: {image_file}')

    if st.button('Detect Palm trees', disabled=(image_file is  None)):
        result = classify_image(bytes_image)
        if result:
            st.write('Palm trees detected')
        else:
            st.write(f'Palm trees not detected')
            
        with st.expander('Sensitivity Analysis', expanded=False):
            analysis_image = analyze_image(image_file)
            st.image(analysis_image, width=256)