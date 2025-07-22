import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

st.title("🎨 Image Compression using K-Means")

st.markdown("""
This app compresses an image by reducing the number of colors using **K-Means Clustering**.
- Upload an image
- Choose how many colors to keep
- Download the compressed version
""")

# Image Upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Resize for performance
    resized_image = image.resize((256, 256))
    img_data = np.array(resized_image)
    w, h, d = img_data.shape
    pixels = img_data.reshape((-1, 3))

    k = st.slider("🎨 Number of colors (k)", 2, 64, 16)

    if st.button("🧠 Compress Image"):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
        compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
        compressed_img = compressed_pixels.reshape((w, h, 3)).astype(np.uint8)

        result_image = Image.fromarray(compressed_img)

        st.image(result_image, caption=f"Compressed Image with {k} colors", use_column_width=True)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="💾 Download Compressed Image",
            data=buffer,
            file_name="compressed_image.png",
            mime="image/png"
        )
