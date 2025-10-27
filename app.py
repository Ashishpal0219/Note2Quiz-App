import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json
import asyncio # Keep if generate_questions_async is async
import spacy # Keep spaCy import

# Import functions from your utility file
# Ensure utils.py is in the same directory and correct
try:
    from utils import (
        read_docx,
        read_pdf,
        segment_text_by_topic, # Expects nlp model
        generate_questions_with_ai, # Expects nlp model, uses Gemini
        save_to_question_bank,
        load_question_bank,
        export_to_docx,
        export_to_csv,
        extract_topics, # Expects nlp model
        gemini_model, # Import gemini_model status for checks
        clear_question_bank # Import the clear function
    )
    print("DEBUG: Successfully imported from utils.py") # Keep for startup check
except ImportError as e:
    # Use st.exception for better error display during development if needed
    st.error(f"FATAL: Failed to import functions from utils.py. Ensure it exists and has no errors. Error: {e}")
    print(f"FATAL: Failed to import from utils.py: {e}")
    st.exception(e) # Show traceback in app
    st.stop() # Stop if utils cannot be imported


# --- Caching Resources ---
@st.cache_resource # Caching the NLP model load is crucial for performance
def load_spacy_model():
    """Loads the spaCy language model, downloading if necessary."""
    model_name = "en_core_web_sm"
    try:
        #if not spacy.util.is_package(model_name):
             # Show spinner in the main app area during download
             #with st.spinner(f"Downloading required spaCy model ('{model_name}')... This may take a moment."):
                #  print(f"DEBUG: Downloading spaCy model {model_name}...")
                 # spacy.cli.download(model_name)
                 # print(f"DEBUG: Finished downloading spaCy model.")
             # Use st.rerun() if download happens mid-session to ensure UI updates
             # st.rerun() # Or show toast and let user retry action
             #st.toast("spaCy model downloaded.", icon="✅")
        print("DEBUG: Loading spaCy model...")
        model = spacy.load(model_name)
        print("DEBUG: spaCy model loaded successfully.")
        return model
    except OSError as e:
        st.error(f"SpaCy model '{model_name}' could not be loaded. "
                 f"Try running 'python -m spacy download {model_name}' manually in your terminal. Error: {e}")
        print(f"FATAL: spaCy OSError: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        print(f"FATAL: Error loading spaCy model: {e}")
        st.exception(e) # Show traceback
        st.stop()

# Load the NLP model once
nlp = load_spacy_model()
if not nlp:
    st.error("Fatal Error: Failed to load spaCy NLP model. Cannot proceed.")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Note2Quiz",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Note2Quiz\nAI-Powered Quiz Generator from Notes. Uses Google Gemini."
    }
)

# --- Define Inline CSS ---
css_rules = """
/* styles/custom.css embedded */
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.stApp { background: linear-gradient(to bottom right, #e9f1ff, #f8f9fa); }
/* Sidebar */
[data-testid="stSidebar"] > div:first-child { background-color: rgba(248, 249, 250, 0.95); border-right: 1px solid #dee2e6; padding-top: 1.5rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { color: #343a40; }
/* Main Widgets */
.stTextArea, .stSelectbox, .stMultiselect, .stSlider, [data-testid="stExpander"], [data-testid="stDataFrame"] {
    background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 8px;
    border: 1px solid #e9ecef; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stTextArea textarea { border: none; background-color: transparent !important; }
/* Buttons */
[data-testid="stButton"] button { border-radius: 20px; padding: 0.5rem 1.5rem; border: none; transition: all 0.3s ease; font-weight: 500; }
[data-testid="stButton"] button[kind="primary"] { background-color: #0d6efd; color: white; }
[data-testid="stButton"] button[kind="primary"]:hover { background-color: #0b5ed7; }
[data-testid="stButton"] button[kind="primary"]:disabled { background-color: #adb5bd; color: #6c757d; }
[data-testid="stButton"] button[kind="secondary"] { border: 1px solid #6c757d; color: #6c757d; background-color: transparent; }
[data-testid="stButton"] button[kind="secondary"]:hover { background-color: #f8f9fa; color: #495057; }
/* Expanders */
[data-testid="stExpander"] summary { font-weight: 500; background-color: rgba(233, 239, 255, 0.8); border-radius: 5px 5px 0 0; padding: 0.6rem 1rem; border-bottom: 1px solid #dee2e6; }
[data-testid="stExpander"] div[role="region"] { border-top: none; border-radius: 0 0 5px 5px; padding: 1rem; }
/* Headers */
h1, h2, h3 { color: #212529; margin-bottom: 1.2rem; }
h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 0.6rem; }
h2 { color: #0d6efd; }
h3, h4, h5 { color: #495057; }
/* Captions */
.stCaption { color: #6c757d; font-size: 0.9em; }
/* DataFrames */
[data-testid="stDataFrame"] { border: none; }
[data-testid="stDataFrame"] thead th { background-color: #e9ecef; color: #343a40; font-weight: 600; border-bottom: 2px solid #adb5bd; }
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background-color: #f8f9fa; }
[data-testid="stDataFrame"] tbody tr:hover { background-color: #e9f1ff; }
/* Links */
a { color: #0d6efd; text-decoration: none; transition: color 0.2s ease; }
a:hover { color: #0a58ca; text-decoration: underline; }
"""
st.markdown(f"<style>{css_rules}</style>", unsafe_allow_html=True)

# --- Title ---
st.title("📝 Note2Quiz: AI Quiz Generator")
st.markdown("Transform notes (DOCX/PDF) or pasted text into questions with Bloom's Taxonomy levels using Google Gemini.")


# --- Initialize Session State ---
def init_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "user": "Guest",
        "input_text": "",
        "generated_data": None,
        "last_uploaded_filename": None,
        "docx_download_data": None,
        "docx_filename": None,
        "confirm_clear": False, # For clear bank confirmation
        "show_bank": False # For toggling bank view
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# --- Sidebar ---
with st.sidebar:
    try:
        st.image("https://placehold.co/150x50/0d6efd/ffffff?text=Note2Quiz", width=150) # Blue background
    except Exception:
        st.write("Note2Quiz")

    st.header("⚙️ Configuration")

    # File Uploader
    uploaded_file = st.file_uploader(
        "1. Upload Notes (DOCX or PDF)", type=["docx", "pdf"], key="file_uploader",
        help="Upload notes file. Text will appear in the main area."
    )
    st.caption("OR paste text directly in the main area.")
    st.markdown("---")

    # Quiz Settings
    num_questions = st.slider(
        "2. Questions per Topic", 1, 10, 5, key="num_q_slider",
        help="Number of questions per type for each topic."
        )
    q_types = st.multiselect(
        "3. Select Question Types", ["MCQ", "Short Answer", "Viva"], default=["MCQ"],
        key="q_types_multiselect", help="Choose one or more formats."
    )

    # Check AI Readiness
    gemini_ready = bool(gemini_model)

    # Generate Button State/Label
    generate_button_disabled = not st.session_state.input_text or not q_types or not gemini_ready
    generate_button_label = "✨ Generate Quiz"
    if not st.session_state.input_text: generate_button_label = "Add Text or Upload File First"
    elif not q_types: generate_button_label = "Select Question Type(s)"
    elif not gemini_ready: generate_button_label = "Check API Key / Init Error"

    # Generate Button
    generate_button = st.button(
        generate_button_label, type="primary", key="generate_btn",
        disabled=generate_button_disabled, use_container_width=True,
        help="Click to generate questions."
        )

    st.markdown("---")
    st.header("📚 Question Bank")
    # View Bank Button - Toggles state
    if st.button("View / Hide Question Bank", key="view_bank_toggle_btn", type="secondary", use_container_width=True):
        st.session_state.show_bank = not st.session_state.get("show_bank", False)
        # Optionally reset filters when toggling view
        if 'filter_topic' in st.session_state: st.session_state.filter_topic = 'All'
        if 'filter_type' in st.session_state: st.session_state.filter_type = 'All'
        if 'filter_diff' in st.session_state: st.session_state.filter_diff = 'All'
        st.rerun() # Rerun immediately

    st.markdown("---")
    st.header("👤 User")
    # User Name Input
    user_name_input = st.text_input("Enter Your Name", value=st.session_state.user, key="user_name_input")
    if user_name_input != st.session_state.user:
        st.session_state.user = user_name_input
        st.toast(f"User name updated to: {st.session_state.user}")


# --- Main Area Layout ---
col_input, col_topics = st.columns([3, 2]) # Input area wider

with col_input:
    st.subheader("📌 Input Text / Notes")

    # --- File Processing Logic ---
    if uploaded_file is not None:
        # Process only if it's a new file or if the main text state is empty
        if st.session_state.get("last_uploaded_filename", None) != uploaded_file.name or not st.session_state.get("input_text", ""):
            st.info(f"Processing uploaded file: {uploaded_file.name}...")
            text_content_from_file = ""
            with st.spinner(f"Reading {uploaded_file.name}..."):
                if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text_content_from_file = read_docx(uploaded_file)
                elif uploaded_file.type == "application/pdf":
                    text_content_from_file = read_pdf(uploaded_file)

            if text_content_from_file:
                # Update the main session state directly
                st.session_state.input_text = text_content_from_file
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.toast("File read successfully! Text updated.", icon="📄")
                # Rerun necessary here to update the text_area's displayed value reliably
                st.rerun()
            else:
                st.error("Could not extract text from the uploaded file.")
                st.session_state.last_uploaded_filename = None
                # Clear state only if extraction failed for a NEW file attempt
                if st.session_state.get("last_uploaded_filename", None) != uploaded_file.name:
                    st.session_state.input_text = ""

    # --- Display Text Input Area ---
    # Use st.session_state.input_text as the key to bind widget state
    st.text_area(
        "Paste notes here OR see text extracted from file above:",
        key="input_text", # Link directly to session state key
        height=400,
        placeholder="Paste your notes or upload a DOCX/PDF via the sidebar."
    )
    # final_text_input is now reliably st.session_state.input_text
    final_text_input = st.session_state.input_text


with col_topics:
    st.subheader("🔍 Identified Topics (Optional Focus)")
    st.caption("Select topics below to guide the AI, or leave blank for general questions.")
    topics_dict = {} # Holds segmented text {topic_name: segment}
    available_topics_for_multiselect = [] # Holds just topic names for UI

    # Perform segmentation and topic extraction if text exists
    if final_text_input:
        with st.spinner("Analyzing text structure..."):
            topics_dict = segment_text_by_topic(final_text_input, nlp) # Pass nlp model
            if len(final_text_input) > 50: # Avoid topic extraction for very short text
                 available_topics_for_multiselect = extract_topics(final_text_input, nlp) # Pass nlp model

        if available_topics_for_multiselect:
             selected_topics = st.multiselect(
                "Select topics to focus on:",
                options=available_topics_for_multiselect,
                key="topic_multiselect",
                help="AI will try to focus generation on these topics if provided."
                )
        else:
             st.caption("No specific topics automatically extracted. Generating general questions.")
             selected_topics = []

        # Ensure topics_dict is valid for fallback if needed
        if not topics_dict or ("Full Document" not in topics_dict and len(topics_dict) < 1) :
             topics_dict = {"Full Document": final_text_input}

    else:
        st.caption("Paste text or upload a file to extract topics.")
        selected_topics = []


# --- Question Generation Logic ---
if generate_button:
    # Use session state directly for checks
    if not st.session_state.input_text: st.error("Input text missing.")
    elif not q_types: st.error("Select question type(s).")
    elif not gemini_ready: st.error("Gemini AI not ready (check API Key).")
    else:
        all_generated_questions = []
        st.session_state.generated_data = None # Clear previous results

        progress_bar = st.progress(0, text="Initializing AI Generation...")
        status_placeholder = st.empty() # Placeholder for status messages

        # Ensure topics_dict exists
        if not topics_dict: topics_dict = {"Full Document": st.session_state.input_text}

        with st.spinner("🤖 Generating questions... Please be patient."):
            total_topics = len(topics_dict)
            for i, (topic_name, text_segment) in enumerate(topics_dict.items()):
                status_text = f"Processing topic: '{topic_name}' ({i+1}/{total_topics})"
                status_placeholder.info(status_text)
                progress_bar.progress((i + 0.5) / total_topics, text=status_text)

                focused_context = text_segment
                relevant_selected_topics_for_prompt = selected_topics

                # --- Call AI function (pass nlp object) ---
                generated = generate_questions_with_ai(
                    context=focused_context,
                    num_questions=num_questions,
                    q_types=q_types,
                    topic=topic_name,
                    nlp_model=nlp
                )
                if generated:
                    all_generated_questions.extend(generated)
                else:
                    st.warning(f"AI generation may have failed or returned empty for topic: {topic_name}. Check terminal logs.")

            progress_bar.progress(1.0, text="Generation Complete!")
            status_placeholder.success("✅ Question generation process finished!")


        if all_generated_questions:
            st.session_state.generated_data = all_generated_questions
            saved_count = save_to_question_bank(all_generated_questions)
            if saved_count > 0: st.toast(f"{saved_count} new questions saved to bank.", icon="💾")
            # No automatic rerun needed, display section below will pick up the state
        else:
            st.error("❌ AI failed to generate any usable questions overall. Check terminal logs for API errors or empty responses.")

# --- Display Generated Questions & Export Options ---
# This section runs AFTER the generation logic (if button was clicked) OR just displays existing state
if st.session_state.generated_data:
    st.markdown("---")
    st.header("✨ Generated Questions")
    st.caption("Review the questions below. Expand each one for details.")

    # Display using expanders with improved formatting
    for i, q_data in enumerate(st.session_state.generated_data):
        expander_title = (
            f"Q{i+1}: {q_data.get('question_type', 'N/A')} | "
            f"{q_data.get('bloom_level', 'N/A')} | "
            f"{q_data.get('question', '')[:60]}..." # Truncate long questions
        )
        with st.expander(expander_title):
            st.markdown(f"**Topic:** {q_data.get('topic', 'N/A')}")
            st.markdown(f"**Bloom's Level:** {q_data.get('bloom_level', 'N/A')}")
            st.caption(f"Reasoning: {q_data.get('reasoning', 'N/A')}")
            st.markdown("**Question:**")
            st.markdown(f"> {q_data.get('question', '')}") # Blockquote for question

            # Display options clearly on separate lines for MCQs
            if q_data.get('question_type') == 'MCQ' and q_data.get('options'):
                st.markdown("**Options:**")
                options = q_data.get('options', {})
                for label in sorted(options.keys()): # Ensure A, B, C, D order
                    st.markdown(f"- {label}) {options[label]}") # Use markdown list format

            # Display Answer Key clearly
            st.markdown(f"**Answer Key:** {q_data.get('answer', 'N/A')}")

    # --- Export Section ---
    st.markdown("---")
    st.subheader("⬇️ Export Options")
    col_exp1, col_exp2 = st.columns(2)
    # Base filename - ensure folder 'exports/' exists or is created by util functions
    export_filename_base = f"exports/Note2Quiz_Export_{st.session_state.user}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with col_exp1:
        st.markdown("##### CSV Export")
        # Button for Unsolved CSV
        if st.button("📄 CSV (Unsolved)", key="csv_unsolved_btn", type="secondary", use_container_width=True):
            with st.spinner("Generating Unsolved CSV..."):
                generated_path = export_to_csv(st.session_state.generated_data, export_filename_base, include_answers=False)
                if generated_path and os.path.exists(generated_path):
                    with open(generated_path, "rb") as file:
                        st.download_button(
                            label="Download Unsolved CSV", data=file,
                            file_name=os.path.basename(generated_path), mime='text/csv',
                            key=f"csv_unsolved_dl_{datetime.now().timestamp()}" # Dynamic key
                        )
                    try: os.remove(generated_path)
                    except: pass
                else: st.error("Failed to generate Unsolved CSV.")

        # Button for Solved CSV
        if st.button("🔑 CSV (Solved)", key="csv_solved_btn", type="secondary", use_container_width=True):
             with st.spinner("Generating Solved CSV..."):
                generated_path = export_to_csv(st.session_state.generated_data, export_filename_base, include_answers=True)
                if generated_path and os.path.exists(generated_path):
                    with open(generated_path, "rb") as file:
                        st.download_button(
                            label="Download Solved CSV", data=file,
                            file_name=os.path.basename(generated_path), mime='text/csv',
                            key=f"csv_solved_dl_{datetime.now().timestamp()}" # Dynamic key
                        )
                    try: os.remove(generated_path)
                    except: pass
                else: st.error("Failed to generate Solved CSV.")


    with col_exp2:
        st.markdown("##### DOCX Export")
        # Button for Unsolved DOCX
        if st.button("📄 DOCX (Unsolved)", key="docx_unsolved_btn", type="secondary", use_container_width=True):
             with st.spinner("Generating Unsolved DOCX..."):
                  generated_path = export_to_docx(st.session_state.generated_data, export_filename_base, include_answers=False)
                  if generated_path and os.path.exists(generated_path):
                       with open(generated_path, "rb") as file:
                            st.download_button(
                                label="Download Unsolved DOCX", data=file,
                                file_name=os.path.basename(generated_path),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"docx_unsolved_dl_{datetime.now().timestamp()}" # Dynamic key
                            )
                       try: os.remove(generated_path)
                       except: pass
                  else: st.error("Failed to generate Unsolved DOCX.")

        # Button for Solved DOCX
        if st.button("🔑 DOCX (Solved)", key="docx_solved_btn", type="secondary", use_container_width=True):
             with st.spinner("Generating Solved DOCX..."):
                  generated_path = export_to_docx(st.session_state.generated_data, export_filename_base, include_answers=True)
                  if generated_path and os.path.exists(generated_path):
                       with open(generated_path, "rb") as file:
                            st.download_button(
                                label="Download Solved DOCX", data=file,
                                file_name=os.path.basename(generated_path),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"docx_solved_dl_{datetime.now().timestamp()}" # Dynamic key
                            )
                       try: os.remove(generated_path)
                       except: pass
                  else: st.error("Failed to generate Solved DOCX.")


# --- View Question Bank Logic ---
# Now uses the session state variable to decide whether to show this section
if st.session_state.get("show_bank", False):
    st.markdown("---")
    st.header("📚 Question Bank")

    with st.spinner("Loading question bank..."):
        bank_df_full = load_question_bank() # Load the full, unsorted bank

    if not bank_df_full.empty:
        st.info(f"Displaying {len(bank_df_full)} total questions. Use filters below. Click headers to sort.")

        # --- Filtering UI ---
        st.markdown("##### Filter Questions")
        filter_cols = st.columns([1, 1, 1]) # Adjust column ratios if needed

        # Ensure columns exist before creating filters
        topics_in_bank = ['All'] + sorted(bank_df_full['topic'].dropna().unique().tolist()) if 'topic' in bank_df_full.columns else ['All']
        types_in_bank = ['All'] + sorted(bank_df_full['question_type'].dropna().unique().tolist()) if 'question_type' in bank_df_full.columns else ['All']
        diff_in_bank = ['All'] + sorted(bank_df_full['difficulty'].dropna().unique().tolist()) if 'difficulty' in bank_df_full.columns else ['All']

        with filter_cols[0]:
            # Use keys for selectboxes to preserve their state across reruns caused by other filters
            selected_topic_filter = st.selectbox("By Topic:", topics_in_bank, key="filter_topic", index=0 if 'filter_topic' not in st.session_state else topics_in_bank.index(st.session_state.filter_topic))
        with filter_cols[1]:
            selected_type_filter = st.selectbox("By Type:", types_in_bank, key="filter_type", index=0 if 'filter_type' not in st.session_state else types_in_bank.index(st.session_state.filter_type))
        with filter_cols[2]:
            selected_diff_filter = st.selectbox("By Difficulty:", diff_in_bank, key="filter_diff", index=0 if 'filter_diff' not in st.session_state else diff_in_bank.index(st.session_state.filter_diff))

        # --- Apply Filters ---
        filtered_df = bank_df_full.copy()
        if selected_topic_filter != 'All' and 'topic' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['topic'].astype(str) == str(selected_topic_filter)]
        if selected_type_filter != 'All' and 'question_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['question_type'].astype(str) == str(selected_type_filter)]
        if selected_diff_filter != 'All' and 'difficulty' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['difficulty'].astype(str) == str(selected_diff_filter)]

        # --- Display Filtered DataFrame ---
        st.dataframe(
            filtered_df.reset_index(drop=True), # Reset index for clean display
            use_container_width=True,
            # Ensure timestamp converts correctly for sorting, handle potential errors
            column_config={
                 "added_timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm")
            }
        )
        st.caption(f"Showing {len(filtered_df)} matching questions.")

        # --- Add Clear Bank Button ---
        st.markdown("---")
        st.markdown("##### Manage Bank")

        # Use checkbox key with session state for persistence across reruns
        st.checkbox("⚠️ Confirm: I want to delete ALL questions.", key="confirm_clear", value=st.session_state.get("confirm_clear", False))

        # Pass current checkbox state to disabled argument
        if st.button("Clear Entire Question Bank", type="secondary", disabled=not st.session_state.confirm_clear, use_container_width=True):
            if st.session_state.confirm_clear:
                with st.spinner("Clearing question bank..."):
                    success = clear_question_bank() # Call the function from utils.py
                if success:
                    st.success("Question bank cleared successfully!")
                    st.session_state.confirm_clear = False # Reset checkbox state
                    st.rerun() # Rerun to refresh the view (bank_df will be empty)
                else:
                    st.error("Failed to clear question bank. Check terminal logs.")
            # No need for else here as button is disabled if not confirmed

    else:
        st.info("Question bank is empty or could not be loaded.")


# --- Footer ---
st.markdown("---")
st.caption("Note2Quiz | Powered by Google Gemini")
print("DEBUG: Reached end of app.py script execution.") # Final debug print