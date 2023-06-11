import streamlit as st
from PIL import Image
from Libs import predict_image

# Define custom CSS styling for Streamlit components
st.markdown("""
<style>
.header {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    margin-bottom: 30px;
}
.title {
    font-size: 48px;
    color: #FF6347;
    padding: 20px;
}
.icon {
    font-size: 64px;
    margin-bottom: 10px;
    color: #FF6347;
}
</style>
""", unsafe_allow_html=True)

# Display the StreamSign title
st.markdown('<div class="header"><h1 class="title">PRNU Predictor</h1></div>', unsafe_allow_html=True)

# Upload an image
uploaded_files = st.file_uploader(
    "Upload images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])


if uploaded_files is not None:
    images = [Image.open(file) for file in uploaded_files]

    # Display the letter predictions for each image
    st.subheader("Predicted Devices and Images")
    for i in range(len(images)):
        prediction = predict_image(images[i])
        st.subheader("Predicted Device")
        st.markdown(
            f'<p class="icon">{prediction}</p>', unsafe_allow_html=True)
        st.subheader("Image")
        st.image(images[i].transpose(Image.ROTATE_180), use_column_width=True,caption=f"Image {i+1}", clamp=True, output_format="JPEG")
        