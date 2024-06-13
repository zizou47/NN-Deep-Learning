import streamlit as st
import warnings
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import io
import json

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
@st.cache_data
def load_receipts_dataset():
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1")
    return dataset

# Load OCR model
@st.cache_resource
def load_ocr_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    return processor, model

# Process an image with OCR model
def process_image(uploaded_file, processor, model):
    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        return str(e)

# Load text similarity model
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Search documents based on query
def search_documents(query, documents, similarity_model):
    query_embedding = similarity_model.encode(query, convert_to_tensor=True)
    document_embeddings = []
    for doc in documents:
        parsed_data_json = doc["parsed_data"]
        if isinstance(parsed_data_json, str):
            parsed_data = json.loads(parsed_data_json)
        else:
            parsed_data = parsed_data_json
        text_to_compare = json.dumps(parsed_data)  # Adjust this to relevant parsed text
        document_embeddings.append(text_to_compare)
    document_embeddings = similarity_model.encode(document_embeddings, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
    results = []
    for score, doc in zip(scores, documents):
        results.append({"score": score.item(), "document": doc})
    
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

# Main function for Streamlit app
def main():
    st.title("Receipts Organizer")
    st.write("This is a basic Streamlit app for organizing receipts using OCR and language models.")
    
    # Load models and dataset
    processor, model = load_ocr_model()
    similarity_model = load_similarity_model()
    dataset = load_receipts_dataset()
    
    # Display dataset information
    st.write("Dataset Loaded Successfully!")
    st.write("Sample Document from Dataset:")
    st.write(dataset['train'][0])
    
    # Display and process uploaded file
    uploaded_file = st.file_uploader("Upload a receipt", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        extracted_text = process_image(uploaded_file, processor, model)
        st.write("Extracted Text:")
        st.write(extracted_text)
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['extracted_text'] = extracted_text
    
    if 'extracted_text' in st.session_state:
        st.write("Extracted Text:")
        st.write(st.session_state['extracted_text'])
    
    # Search functionality
    query = st.text_input("Enter your query:", value=st.session_state.get('query', ''))
    if query:
        st.session_state['query'] = query
        documents = dataset["train"]
        results = search_documents(query, documents, similarity_model)
        st.write("Search Results:")
        for result in results:
            st.write(f"Score: {result['score']:.4f}")
            st.write(f"Document ID: {result['document']['id']}")
            parsed_data = json.loads(result['document']['parsed_data']['json'])
            st.write(f"Parsed Data: {parsed_data}")
            st.write("Raw OCR Data:")
            st.write(result['document']['raw_data']['ocr_words'])

if __name__ == "__main__":
    main()
