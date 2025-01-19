import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ''

GROQ_API_KEY = st.secrets['GROQ_API_KEY']
llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", streaming = True)


def get_transcript(video_url):
    """Fetch the transcript of a YouTube video."""
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except NoTranscriptFound:
        return "No transcript found for this video."
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except Exception as e:
        return f"Error fetching transcript: {e}"

def find_word_timestamps(transcript, search_word):
    """Find timestamps for a specific word in the transcript."""
    timestamps = []
    for entry in transcript:
        if search_word.lower() in entry["text"].lower():
            timestamps.append(f"{entry['start']:.2f}s - {entry['text']}")
    
    if timestamps:
        return "\n".join(timestamps)
    else:
        return f"No occurrences of the word '{search_word}' found."

def summarize_transcript(transcript):
    # st.write(transcript)
    """Summarize the YouTube transcript using ChatGroq LLM."""
    text = " ".join([entry["text"] for entry in transcript])
    text = text[:5000]
    prompt = f"Based on the following {text},  summarize {text}"
    template = PromptTemplate(
                input_variables=["text"],
                template=prompt,
            )
    lm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", streaming = True)
            # Create the LLMChain to manage the model and prompt interaction
    llm_chain = LLMChain(prompt=template, llm=llm)
    response = llm_chain.invoke({
        "content" : text
    })      
            
         
    return response["text"]


# Streamlit Interface
st.title("YouTube Video Transcript Analysis")
st.write("Analyze YouTube videos by fetching transcripts, finding timestamps for specific words, or summarizing the video.")
st.markdown("Developed by Ravi Shankar Prasad - Data Scientist at Beckman Coulter Life Sciences")
# User Input: YouTube video URL
video_url = st.text_input("Enter YouTube video URL:")
if not video_url:
    st.stop()

# Fetch Transcript
if len(st.session_state['transcript'])<2:
    st.session_state['transcript'] = get_transcript(video_url)
if isinstance(st.session_state['transcript'], str):  # Error or no transcript found
    st.error(st.session_state['transcript'])
else:
    # Select Action
    action = st.radio(
        "What would you like to do?",
        ("Fetch Transcript", "Find Timestamps for a Word", "Summarize the Video")
    )

    # Fetch Transcript
    if action == "Fetch Transcript":
        st.text_area("Transcript", value=st.session_state['transcript'])

    # Find Timestamps for Word
    elif action == "Find Timestamps for a Word":
        search_word = st.text_input("Enter the word to search for:")
        if search_word and st.button("Find Timestamps"):
            timestamps = find_word_timestamps(st.session_state['transcript'], search_word)
            st.write(timestamps)

    # Summarize the Video
    elif action == "Summarize the Video":
        if st.button("Summarize"):
            summary = summarize_transcript(st.session_state['transcript'])
            st.write("### Summary:")
            st.write(summary)
