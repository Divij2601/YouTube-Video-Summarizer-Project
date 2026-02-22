from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchaingroq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.tools import YouTubeSearchResults

load_dotenv()


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)


def extract_video_id(state: YtGraphState):
    video_url = state.video_url
    template =PromptTemplate(
        template='''
        You are a helpful assistant that extracts the video ID from the YouTube video URL.
        The video URL is {video_url}.''',
        input_variables=['video_url'] 
    )
    llm_with_structured_output = llm.with_structured_output(ExtractVideoId)
    chain = template | llm_with_structured_output
    result = chain.invoke({"video_url": video_url})
    return {"video_id": result.video_id}

def extract_transcript(state: YtGraphState):
    video_id = state.video_id
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript_text = " "
    for snippet in fetched_transcript:
        transcript_text += snippet.text + " "
        return {"transcript": transcript_text}

def summarize_transcript(state: YtGraphState):
    transcript = state.transcript
    template = PromptTemplate(
        template=''' summarize the following transcript: {transcript} in a concise manner.
        ''',
        input_variables=['transcript']
    )
    chain = template | llm
    result = chain.invoke({"transcript": transcript})
    return {"summary": result.content}

def generate_questions(state: YtGraphState):
    summary = state.summary
    template = PromptTemplate(
        template=''' generate 5 questions based on the following summary: {summary}
        ''',
        input_variables=['summary']
    )
    chain = template | llm
    result = chain.invoke({"summary": summary})
    return {"questions": result.content}

def next_steps(state: YtGraphState):
    summary = state.summary
    template = PromptTemplate(
        template=''' generate 5 next steps based on the following summary: {summary}
        ''',
        input_variables=['summary']
    )
    chain = template | llm
    result = chain.invoke({"summary": summary})
    return {"next_steps": result.content}

def find_keywords(state: YtGraphState):
    transcript = state.transcript
    template = PromptTemplate(
        template=''' extract the keywords from the following transcript: {transcript}
        keywords should be a single word or phrase that is relevant to the transcript.
        return only the keywords, no other text.
        ''',
        input_variables=['transcript']
    )
    chain = template | llm
    result = chain.invoke({"transcript": transcript})
    return {"keywords": result.content}

def video_suggestions(state: YtGraphState):
    keywords = state.keywords
    tool = YouTubeSearchResults(max_results=5)
    video_suggestions = tool.invoke(keywords)
    return {"video_suggestions": video_suggestions}



#State of the graph 
# This is the state of the graph, it is a pydantic model that is used to store the state of the graph
class YtGraphState(BaseModel):
    video_url: str = Field(..., description="The URL of the YouTube video to summarize")
    video_id: optional[str] = Field(default=None, description="The ID of the YouTube video to summarize")
    summary: Optional[str] = Field(default=None, description="The summary of the YouTube video")
    Transcript: Optional[str] = Field(default=None, description="The transcript of the YouTube video")
    keywords: Optional[List[str]] = Field(default=None, description="The keywords of the YouTube video")
    video_suggestions: Optional[List[str]] = Field(default=None, description="The suggestions of the YouTube video")
    questions: Optional[str] = Field(default=None, description="The questions of the YouTube video")
    next_steps: Optional[str] = Field(default=None, description="The next steps of the YouTube video")

#This is the model that is used to extract the video ID from the YouTube video URL
class ExtractVideoId(BaseModel):
    video_id: str = Field(..., description="The ID of the YouTube video to summarize")


builder = StateGraph(YtGraphState)

builder.add_node("extract_video_id", extract_video_id)
builder.add_node("extract_transcript", extract_transcript)
builder.add_node("summarize_transcript", summarize_transcript)
builder.add_node("generate_questions", generate_questions)
builder.add_node("next_steps", next_steps)
builder.add_node("find_keywords", find_keywords)
builder.add_node("video_suggestions", video_suggestions)

builder.add_edge(START, "extract_video_id")
builder.add_edge("extract_video_id", "extract_transcript")
builder.add_edge("extract_transcript", "summarize_transcript")
builder.add_edge("extract_transcript", "find_keywords")
builder.add_edge("summarize_transcript", "generate_questions")
builder.add_edge("summarize_transcript", "next_steps")
builder.add_edge("find_keywords", "video_suggestions")
builder.add_edge("video_suggestions", END)
builder.add_edge("generate_questions", END)
builder.add_edge("next_steps", END)

graph = builder.compile()