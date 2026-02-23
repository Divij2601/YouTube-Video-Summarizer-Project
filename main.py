
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, field_validator
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
import re

load_dotenv(override=True)



llm = ChatGroq(
    model= "llama-3.1-8b-instant"
    temperature=0.3  # Lower temperature for deterministic production output
)

class GraphState(BaseModel):
    """
    State object passed between LangGraph nodes.
    All fields are strictly typed for production reliability.
    """

    video_url: HttpUrl = Field(..., description="Valid YouTube video URL")

    video_id: Optional[str] = Field(
        default=None,
        min_length=5,
        description="Extracted YouTube video ID"
    )

    transcript: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Full transcript text"
    )

    summary: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Concise transcript summary"
    )

    keywords: Optional[List[str]] = Field(
        default=None,
        description="List of extracted keywords"
    )

    video_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested related videos"
    )

    questions: Optional[List[str]] = Field(
        default=None,
        description="Generated follow-up questions"
    )

    next_steps: Optional[List[str]] = Field(
        default=None,
        description="Suggested learning next steps"
    )

    # Ensure keywords are not empty strings
    @field_validator("keywords")
    def validate_keywords(cls, v):
        if v:
            for word in v:
                if not word.strip():
                    raise ValueError("Keywords cannot contain empty strings.")
        return v


class ExtractedVideoID(BaseModel):
    video_id: str 

class SummaryOutput(BaseModel):
    summary: str

class KeywordOutput(BaseModel):
    keywords: List[str]

class QuestionsOutput(BaseModel):
    questions: List[str]


class NextStepsOutput(BaseModel):
    next_steps: List[str]



def extract_video_id(state: GraphState):
    """
    Extract YouTube video ID using structured LLM output.
    """

    template = PromptTemplate.from_template(
        """
        Extract the video ID from this YouTube URL:
        {video_url}

        Return ONLY the video ID.
        """
    )

    structured_llm = llm.with_structured_output(ExtractedVideoID)

    chain = template | structured_llm

    result = chain.invoke({"video_url": state.video_url})

    return {"video_id": result.video_id}




def extract_transcript(state: GraphState):
    """
    Fetch transcript using YouTubeTranscriptApi.
    Raises error if transcript not available.
    """

    try:
        transcript_list = YouTubeTranscriptApi().fetch(state.video_id)
    except Exception as e:
        raise RuntimeError(f"Transcript fetch failed: {str(e)}")

    transcript_text = " ".join([snippet.text for snippet in transcript_list])

    if not transcript_text.strip():
        raise ValueError("Transcript is empty.")

    return {"transcript": transcript_text}


def summarize_transcript(state: GraphState):
    """
    Generate structured summary from transcript.
    """

    template = PromptTemplate.from_template(
        """
        Summarize the following transcript concisely.

        Transcript:
        {transcript}
        """
    )

    structured_llm = llm.with_structured_output(SummaryOutput)

    chain = template | structured_llm

    result = chain.invoke({"transcript": state.transcript})

    return {"summary": result.summary}



def find_keywords(state: GraphState):
    """
    Extract structured keyword list.
    """

    template = PromptTemplate.from_template(
        """
        Extract 3-5 important keywords from this transcript.
        Return them as a Python list of strings.

        Transcript:
        {transcript}
        """
    )

    structured_llm = llm.with_structured_output(KeywordOutput)

    chain = template | structured_llm

    result = chain.invoke({"transcript": state.transcript})

    return {"keywords": result.keywords}


def generate_questions(state: GraphState):
    """
    Generate structured follow-up questions.
    """

    template = PromptTemplate.from_template(
        """
        Generate 5 thoughtful questions based on this summary.

        Summary:
        {summary}
        """
    )

    structured_llm = llm.with_structured_output(QuestionsOutput)

    chain = template | structured_llm

    result = chain.invoke({"summary": state.summary})

    return {"questions": result.questions}



def generate_next_steps(state: GraphState):
    """
    Generate structured next steps.
    """

    template = PromptTemplate.from_template(
        """
        Suggest 5 practical next learning steps based on this summary.

        Summary:
        {summary}
        """
    )

    structured_llm = llm.with_structured_output(NextStepsOutput)

    chain = template | structured_llm

    result = chain.invoke({"summary": state.summary})

    return {"next_steps": result.next_steps}



def video_suggestions(state: GraphState):
    """
    Use YouTubeSearchTool safely.
    """

    tool = YouTubeSearchTool()

    query = " ".join(state.keywords)

    try:
        results = tool.invoke(query)
    except Exception as e:
        raise RuntimeError(f"YouTube search failed: {str(e)}")

    return {"video_suggestions": results}



builder = StateGraph(GraphState)

builder.add_node("extract_video_id", extract_video_id)
builder.add_node("extract_transcript", extract_transcript)
builder.add_node("summarize_transcript", summarize_transcript)
builder.add_node("find_keywords", find_keywords)
builder.add_node("generate_questions", generate_questions)
builder.add_node("generate_next_steps", generate_next_steps)
builder.add_node("video_suggestions", video_suggestions)

# Flow definition
builder.add_edge(START, "extract_video_id")
builder.add_edge("extract_video_id", "extract_transcript")
builder.add_edge("extract_transcript", "summarize_transcript")
builder.add_edge("extract_transcript", "find_keywords")
builder.add_edge("summarize_transcript", "generate_questions")
builder.add_edge("summarize_transcript", "generate_next_steps")
builder.add_edge("find_keywords", "video_suggestions")

builder.add_edge("generate_questions", END)
builder.add_edge("generate_next_steps", END)
builder.add_edge("video_suggestions", END)

graph = builder.compile()