import os
from dotenv import load_dotenv
import streamlit as st
from mysql_access import Database
from mysql.connector import Error
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
# Import the Gemini embeddings model (adjust import as needed)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Get your Gemini API key from the .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# (Replace "gemini-model-name" with the actual model name you want to use)
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Directory where the Chroma vector store will be persisted
CHROMA_PERSIST_DIR = "./chroma_db"

# Database configuration details
db_config = {
    'host': 'localhost',
    'database': 'wedding_planner',
    'user': 'root',
    'password': '1234'
}


# Create an instance of your Database connector
db = Database(**db_config)

def fetch_vendors_data(db):
    """
    Fetch vendor data from the vendors_new table.
    This query retrieves all columns as defined in your table.
    """
    try:
        # Get a connection and create a dictionary cursor.
        conn = db.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT 
                id,
                firstname,
                lastname,
                email,
                category_id,
                place_id,
                password,
                sub_category_id,
                zipcode,
                title,
                description,
                position,
                image,
                altTag,
                location,
                city_id,
                latitude,
                longitude,
                avg_price,
                address,
                state_code,
                tel,
                website,
                is_website,
                is_code_added,
                is_verified,
                is_profile_completed,
                remember_token,
                is_popular,
                services,
                phone,
                is_appointment_only,
                embed_video,
                facebook_link,
                linkedin_link,
                twitter_link,
                instagram_link,
                pininterest_link,
                tiktok_link,
                package,
                allow_platinum_plan,
                annual_revenue,
                year_founded,
                no_employees,
                customer_id,
                is_claimed,
                is_claimed_reminder,
                deleted_at,
                created_at,
                updated_at,
                geocoded,
                geocodedate,
                temp_id,
                stripe_id,
                card_brand,
                card_last_four,
                trial_ends_at,
                vendor_bsc_allowed,
                Comapny_Name,
                Comapny_Number,
                Comapny_Address,
                Comapny_URL,
                Comapny_Email,
                Comapny_payment_card_type,
                Comapny_payment_card_number,
                Comapny_payment_card_expiry,
                Comapny_payment_card_CVC,
                Comapny_insta,
                Comapny_fb,
                tax_id,
                surveyResponse,
                stripe_customer_id,
                stripe_payment_method_id,
                vendor_order_state,
                count_views,
                url,
                rating,
                is_subscribed,
                reason_for_blocking,
                is_blocked,
                re_write,
                description_by_vendor,
                rating_cron
            FROM vendors_new
        """
        cursor.execute(query)
        vendors_data = cursor.fetchall()
        return vendors_data
    except Error as e:
        st.error(f"Error fetching vendor data: {e}")
        return []
    finally:
        if conn.is_connected():
            db.close_connection(conn)

def create_documents(vendors_data):
    """
    Create a list of Document objects from the vendor data.
    In this example, we use a subset of the available fields to form the text.
    Adjust the fields below as needed.
    """
    docs = []
    for vendor in vendors_data:
        text = (
            f"Vendor ID: {vendor['id']}\n"
            f"Name: {vendor['firstname']} {vendor['lastname']}\n"
            f"Email: {vendor['email']}\n"
            f"Title: {vendor['title']}\n"
            f"Description: {vendor['description']}\n"
            f"Company Name: {vendor.get('Comapny_Name', '')}\n"
            f"Address: {vendor.get('address', '')}\n"
            f"Location: {vendor.get('location', '')}\n"
            f"Website: {vendor.get('website', '')}\n"
            f"Phone: {vendor.get('phone', '')}\n"
            f"Services: {vendor.get('services', '')}\n"
        )
        metadata = {
            "id": vendor["id"],
            "firstname": vendor["firstname"],
            "lastname": vendor["lastname"],
            "email": vendor["email"]
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def chunk_documents(documents):
    """
    Split the documents into smaller chunks using a RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def index_documents(document_chunks):
    """
    Generate embeddings for the document chunks using Gemini and
    store them in a persistent Chroma vector store.
    """
    vectordb = Chroma.from_documents(
        documents=document_chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="vendors"
    )
    vectordb.persist()


# Streamlit UI Configuration
st.title("Wedding Planner Vendor Embeddings")
st.markdown("### Process vendor data from the database and store embeddings")

if st.button("Process Vendor Data"):
    # Fetch vendor data from the vendors_new table.
    vendors_data = fetch_vendors_data(db)
    
    if vendors_data:
        st.write("Vendor Data Sample:", vendors_data[:3])
        
        # Create Document objects for each vendor.
        docs = create_documents(vendors_data)
        
        # Split each document into smaller chunks.
        chunks = chunk_documents(docs)
        
        # Generate embeddings and index them in Chroma.
        index_documents(chunks)
        
        st.success("Vendor data processed successfully! Embeddings have been stored in Chroma.")
    else:
        st.error("No vendor data found or an error occurred while fetching data.")