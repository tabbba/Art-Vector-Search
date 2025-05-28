import os
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from transformers import AutoImageProcessor, AutoModel
import streamlit as st
import torch
from PIL import Image
import base64
from io import BytesIO
import requests
from urllib.parse import urlparse

# Set page configuration first
st.set_page_config(
    page_title="Art Explorer AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant styling with improved spacing
st.markdown("""
<style>
    .main .block-container {
        padding-top: 0rem; /* Remove top padding completely */
        padding-bottom: 0rem;
        max-width: 800px;
        margin: 0 auto;
    }

    h1 {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
        margin-top: 0; /* Remove top margin for headers */
        text-align: center; /* Center the main heading */
    }
    h2 {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
        margin-top: 0;
    }
    h3 {
        font-size: 1.4rem;
        margin-bottom: 0.2rem;
        margin-top: 0;
    }
    p {
        font-size: 1.0rem;
        line-height: 1.3;
        margin-bottom: 0.4rem;
    }

    /* Default button styling */
    .stButton > button {
        width: 60%;
        margin: 0 auto;
        display: block;
        border-radius: 3px;
        font-weight: 250;
        font-size: 0.65rem;
        padding: 0.1rem 0.3rem;
        border: none;
        color: white;
        background-color: #2d2d2d !important; /* Added !important to ensure this style always applies */
        transition: background-color 0.2s;
    }

    /* Hover styling */
    .stButton > button:hover {
        background-color: #444444 !important; /* Added !important */
        color: white;
    }

    /* Ensure the button reverts to default color after clicking/focus */
    .stButton > button:focus,
    .stButton > button:active {
        background-color: #2d2d2d !important; /* Same as your default color */
        color: white !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* MORE COMPACT LAYOUT: Move caption much closer to image */
    .image-caption {
        text-align: center;
        font-style: italic;
        margin-top: 0; /* Further reduced */
        margin-bottom: 0; /* Further reduced */
        font-size: 0.8rem;
        line-height: 1.1; /* Add tighter line height */
    }
    
    /* Adjust progress bar to be closer to caption */
    .stProgress {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0.1rem !important; /* Reduced margin */
        margin-bottom: 0.1rem !important; /* Reduced margin */
    }
    
    .stProgress > div > div {
        height: 0.2rem !important; /* Even thinner progress bar */
    }
    
    /* Extremely compact similarity info */
    .similarity-info {
        text-align: center;
        font-size: 0.75rem;
        margin-bottom: 0.3rem; /* Reduced */
        margin-top: 0; /* Removed top margin completely */
        line-height: 1; /* Tighter line height */
    }
    
    /* Add space between image cells */
    .image-cell {
        margin-bottom: 0.8rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar-nav-text {
        font-size: 0.8rem !important;
    }
    .highlight-box {
        border-radius: 6px;
        padding: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .stFileUploader > div {
        padding: 0.3rem !important;
    }
    .element-container {
        margin-bottom: 0.2rem !important; /* Further reduced */
    }
    
    /* Tighter layout */
    .tight-layout {
        margin-bottom: 0.4rem !important; /* Further reduced */
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    footer {
        visibility: hidden;
    }
    
    /* FOOTER THAT APPEARS ON SCROLL - not fixed anymore */
    .scroll-footer {
        width: 100%;
        text-align: center;
        padding: 0.1rem 0;
        background-color: rgba(14,17,23);
        border-top: 1px solid #131026;
        margin-top: 0.1rem;
    }
    
    /* Reset button in sidebar styling */
    .reset-btn {
        width: 100% !important;
        background-color: #4a4a4a !important;
        color: white;
        font-size: 0.8rem !important;
        margin-top: 0.5rem !important;
        border-radius: 4px !important;
    }
    
    /* Upload area styling */
    .upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px dashed rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    .centered-upload {
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Emoji icon styling */
    .emoji-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        display: inline-block;
        vertical-align: middle;
    }
    
    /* Header with emoji styling */
    .header-with-emoji {
        display: flex;
        align-items: center;
        margin-bottom: 0.3rem;
    }
    
    .stImage img {
        border-radius: 3px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.1rem !important; /* Add minimal margin after image */
    }
    
    hr {
        margin: 0.3rem 0 0.5rem 0 !important;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stAlert > div {
        padding: 0.3rem 0.5rem !important;
        font-size: 0.85rem !important;
    }
    
    /* Remove extra space above elements */
    .css-1kyxreq {
        margin-top: 0 !important;
    }
    
    /* Image containers - reduce spacing */
    .css-ocqkz7, .css-12w0qpk {
        gap: 0.5rem !important;
    }
    
    /* Column layout spacing adjustments */
    div[data-testid="column"] {
        padding: 0 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

QDRANT_API = os.getenv("QDRANT_API")
QDRANT_URL = "https://cd8db105-544d-457f-aa1a-97d4475c1f56.europe-west3-0.gcp.cloud.qdrant.io"

collection_name = "dino_embedding_collection"

# Initialize session state
if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None
if 'similar_records' not in st.session_state:
    st.session_state.similar_records = None

# Function to create and cache the Qdrant client
@st.cache_resource
def get_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API
    )

def add_header(title, emoji, description=None):
    """
    Displays a header with an emoji and an optional, centered subtitle.
    """
    st.markdown(f"""
    <div class="header-with-emoji" style="justify-content:center;">
        <span class="emoji-icon">{emoji}</span>
        <h1>{title}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if description:
        # Center the subtitle
        st.markdown(f"<p style='font-size:0.9rem; opacity:0.8; margin-top:0; text-align:center;'>{description}</p>", unsafe_allow_html=True)
    
    # Simple horizontal rule
    st.markdown("<hr style='margin:0.2rem 0 0.6rem 0'>", unsafe_allow_html=True)

# Load the DINOv2-large model
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large")
    return processor, model

processor, model = load_model()

# Function to generate an embedding
def generate_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

def load_image(url_or_path, target_size=(180, 180)):
    try:
        parsed = urlparse(url_or_path)
        if parsed.scheme in ('http', 'https'):
            response = requests.get(url_or_path)
            img = Image.open(BytesIO(response.content))
        else:
            if not os.path.isabs(url_or_path):
                url_or_path = os.path.abspath(url_or_path)
            img = Image.open(url_or_path)

        img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        return img
    except Exception as e:
        st.error(f"Error loading image {url_or_path}: {str(e)}")
        return None

# Function to filter unique records
def filter_unique_records(records):
    seen_images = set()
    unique_records = []
    
    for record in records:
        image_url = record.payload.get("image_url", "")
        if image_url and image_url not in seen_images:
            seen_images.add(image_url)
            unique_records.append(record)
    
    return unique_records

# Function to get initial records (randomized)
def get_initial_records():
    with st.spinner("Loading art collection..."):
        client = get_client()
        records, _ = client.scroll(
            collection_name=collection_name,
            with_vectors=False,
            limit=100
        )

        unique_records = filter_unique_records(records)
        random.shuffle(unique_records)  # Shuffle images before displaying
        return unique_records[:30]  # Show up to 30 images

# Function to update the selected record
def set_selected_record(new_record):
    st.session_state.selected_record = new_record
    st.session_state.similar_records = None

# Function to clear selection and reset
def clear_selection():
    st.session_state.selected_record = None
    st.session_state.similar_records = None

# Function to get similar records
def get_similar_records():
    client = get_client()
    if st.session_state.selected_record is not None:
        if st.session_state.similar_records is None:
            with st.spinner("Finding similar artworks..."):
                similar_records = client.recommend(
                    collection_name=collection_name,
                    positive=[st.session_state.selected_record.id],
                    limit=100
                )
                st.session_state.similar_records = filter_unique_records(similar_records)[:15]
        return st.session_state.similar_records
    return get_initial_records()

# Function to search similar paintings
def search_similar_paintings(embedding, top_k=9):
    with st.spinner("Discovering similar artworks..."):
        client = get_client()
        results = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=50
        )
        
        seen_images = set()
        paintings = []
        
        for result in results:
            image_url = result.payload.get("image_url", "")
            if image_url and image_url not in seen_images:
                seen_images.add(image_url)
                author = result.payload.get("author", "Unknown Artist")
                score = result.score
                paintings.append({"image_url": image_url, "author": author, "score": score})
                
                if len(paintings) >= top_k:
                    break
        
        return paintings

# Function to display records in a grid layout
def display_records(records):
    columns = st.columns(5)  # 5 columns for a compact grid
    for idx, record in enumerate(records):
        col_idx = idx % 5
        image_url = record.payload.get("image_url", "")
        if not image_url:
            continue
            
        image = load_image(image_url)
        if image:
            author = record.payload.get("author", "Unknown Artist")
            with columns[col_idx]:
                with st.container():
                    st.image(image=image, use_container_width=True)
                    st.markdown(f'<div class="image-caption">{author}</div>', unsafe_allow_html=True)
                    st.button(
                        label="Find Similar",
                        key=f"btn_{record.id}",
                        on_click=set_selected_record,
                        args=[record]
                    )
                    st.markdown("<div style='margin-bottom:0.4rem'></div>", unsafe_allow_html=True)

# ---------------- Sidebar -----------------

st.sidebar.markdown('<div style="text-align: center; padding-top:0.2rem">', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="header-with-emoji" style="justify-content: center">
    <h2 style="font-size:1.2rem; margin-bottom:0.2rem; margin-left:0.3rem;">Art Explorer AI</h2>
    <span style="font-size:1.4rem;">🎨</span>
</div>
<p style="font-size:0.8rem; font-style:italic; opacity:0.8; text-align:center;">
    Explore Art Through Vector Similarity
</p>
""", unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin:0.3rem 0 0.4rem 0'>", unsafe_allow_html=True)

# Main navigation
st.sidebar.markdown("<h3 style='font-size:0.9rem; margin-bottom:0.2rem'>Navigation</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Painting Collection", "Upload and Discover"], label_visibility="collapsed")

# Reset button in sidebar when an image is selected
if st.session_state.selected_record:
    st.sidebar.markdown("<hr style='margin:0.3rem 0 0.4rem 0'>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='font-size:0.9rem; margin-bottom:0.2rem'>Selected Artwork: </h3>", unsafe_allow_html=True)
    image_url = st.session_state.selected_record.payload.get("image_url", "")
    if image_url:
        selected_image = load_image(image_url, target_size=(140, 140))
        if selected_image:
            st.sidebar.image(selected_image, use_container_width=True)
            author = st.session_state.selected_record.payload.get("author", "Unknown Artist")
            st.sidebar.markdown(f'<div class="image-caption">{author}</div>', unsafe_allow_html=True)
            
            reset_btn = st.sidebar.button("Return to Collection", key="reset_btn", on_click=clear_selection)
            st.sidebar.markdown("""
            <style>
                    /* Return to Collection button styling */
                    div[data-testid="stButton"]:last-child > button {
                        background-color: #2d2d2d !important;
                        color: white !important;
                        font-size: 0.7rem !important; /* Reduced font size */
                        margin-top: 0.2rem !important;
                        width: 80% !important; /* Reduced width */
                    }

                    div[data-testid="stButton"]:last-child > button:hover {
                        background-color: #444444 !important;
                    }
            </style>
            """, unsafe_allow_html=True)

# ---------------- Main Pages -----------------

# Painting Collection Section
if page == "Painting Collection":
    add_header(
        title="Art Collection Explorer",
        emoji="🖼️",
        description="Browse through our collection and discover visually similar artworks."
    )

    # Intro text for the academic version
    st.markdown("""
    <div style="text-align:center;">
        <p style="font-size:0.9rem;">
            This interface provides an overview of a curated set of paintings. Selecting any image
            will retrieve artworks that share visual or stylistic characteristics, as determined by 
            modern representation learning techniques.
        </p>
    </div>
    <hr style='margin:0.5rem 0 0.6rem 0'>
    """, unsafe_allow_html=True)

    try:
        records = get_similar_records() if st.session_state.selected_record else get_initial_records()

        if st.session_state.selected_record:
            st.markdown("""
            <div class="header-with-emoji">
                <span style="font-size:1.1rem;">🔍</span>
                <h3 style='font-size:1.0rem; margin-top:0.3rem; margin-bottom:0.2rem; margin-left:0.3rem;'>Similar Artworks</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.85rem; margin-bottom:0.4rem; opacity:0.8'>Artworks that display a high degree "
                "of visual similarity to the selected image are shown below:</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div class="header-with-emoji">
                <span style="font-size:1.1rem;">✨</span>
                <h3 style='font-size:1.0rem; margin-top:0.3rem; margin-bottom:0.2rem; margin-left:0.3rem;'>Featured Collection</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.85rem; margin-bottom:0.4rem; opacity:0.8'>Select any artwork below to initiate a similarity-based retrieval.</p>",
                unsafe_allow_html=True
            )

        if records:
            display_records(records)
        else:
            st.warning("No images found in the collection. Please try again later.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Upload and Discover Section
elif page == "Upload and Discover":
    add_header(
        title="Upload and Discover",
        emoji="🌐",
        description="Find paintings similar to any image you upload"
    )

    # Centered text, academic tone
    st.markdown("""
    <div style="text-align:center; margin-top:0.2rem; margin-bottom:0.6rem;">
        <p style="font-size:0.9rem;">
            Please upload a JPG or PNG image. Once uploaded, a vector embedding will be generated 
            and used to identify artworks that exhibit similar visual features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Center the upload area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Results section with minimal styling
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("""
        <div class="header-with-emoji" style="margin-top:1rem;">
            <h3 style='font-size:1.0rem; margin-bottom:0.2rem; margin-left:0.3rem;'>Visual Search Results 🔎</h3>

        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.2rem 0 0.4rem 0'>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3], gap="small")

        with col1:
            st.markdown("""
            <div class="header-with-emoji">
                <h3 style='font-size:0.9rem; margin-bottom:0.2rem; margin-left:0.3rem;'>Your Image:</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            try:
                # Request 8 similar images
                embedding = generate_embedding(image)
                results = search_similar_paintings(embedding, top_k=8)

                if results:
                    st.markdown("""
                    <div class="header-with-emoji">
                        <h3 style='font-size:0.9rem; margin-bottom:0.2rem; margin-left:0.3rem;'>Similar Artworks:</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    columns = st.columns(4)
                    for col_idx in range(4):
                        with columns[col_idx]:
                            idx1 = col_idx * 2
                            idx2 = col_idx * 2 + 1

                            if idx1 < len(results):
                                with st.container():
                                    similar_image = load_image(results[idx1]["image_url"], target_size=(256, 256))
                                    st.image(similar_image, use_container_width=True)
                                    st.markdown(
                                        f'<div class="image-caption">{results[idx1]["author"]}</div>',
                                        unsafe_allow_html=True
                                    )
                                    similarity = int(results[idx1]["score"] * 100)
                                    st.progress(similarity / 100)
                                    st.markdown(
                                        f'<div class="similarity-info">Similarity: {similarity}%</div>',
                                        unsafe_allow_html=True
                                    )
                                    st.markdown("<div style='margin-bottom:0.3rem'></div>", unsafe_allow_html=True)

                            if idx2 < len(results):
                                with st.container():
                                    similar_image = load_image(results[idx2]["image_url"], target_size=(100, 100))
                                    st.image(similar_image, use_container_width=True)
                                    st.markdown(
                                        f'<div class="image-caption">{results[idx2]["author"]}</div>',
                                        unsafe_allow_html=True
                                    )
                                    similarity = int(results[idx2]["score"] * 100)
                                    st.progress(similarity / 100)
                                    st.markdown(
                                        f'<div class="similarity-info">Similarity: {similarity}%</div>',
                                        unsafe_allow_html=True
                                    )
                else:
                    st.warning("No similar paintings found. Try uploading a different image.")
            except Exception as e:
                st.error(f"An error occurred while processing your image: {str(e)}")
