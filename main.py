from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googleapiclient.discovery import build
import pandas as pd
import re
from transformers import pipeline
import os
from typing import Optional, List

# Initialize FastAPI app
app = FastAPI(title="YouTube Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the model
sentiment_pipeline = None

# ✅ Replace this with your YouTube Data API v3 key
API_KEY = "AIzaSyDHYSiiRKArvvAZ5kxSfAx4J28WJrM8auw"  # Replace with your actual API key

class VideoRequest(BaseModel):
    video_url: str

class Comment(BaseModel):
    author: str
    text: str
    likeCount: int
    publishedAt: str
    sentiment_score: Optional[float] = None

class SentimentResult(BaseModel):
    positive_percentage: float
    negative_percentage: float
    total_comments: int
    most_liked_positive: Optional[Comment]
    most_liked_negative: Optional[Comment]
    top_positive_comments: List[Comment]
    top_negative_comments: List[Comment]

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)

def get_video_comments(video_id: str, max_comments: int = 500) -> pd.DataFrame:
    """Get comments from YouTube video with quota management"""
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        comments = []
        next_page_token = None
        total_fetched = 0

        while total_fetched < max_comments:
            # Calculate how many comments to fetch in this batch
            batch_size = min(100, max_comments - total_fetched)
            
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                textFormat="plainText",
                pageToken=next_page_token
            ).execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": comment["authorDisplayName"],
                    "text": comment["textDisplay"],
                    "likeCount": comment["likeCount"],
                    "publishedAt": comment["publishedAt"]
                })
                total_fetched += 1

            next_page_token = response.get("nextPageToken")
            if not next_page_token or total_fetched >= max_comments:
                break

        return pd.DataFrame(comments)
    
    except Exception as e:
        # Log the exact error for debugging
        print(f"YouTube API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching comments: {str(e)}")

@app.on_event("startup")
async def load_model():
    """Load the model during the startup of the FastAPI application"""
    global sentiment_pipeline
    model_path = "./local_model"  # Path where you saved the model

    # Load the model from the local directory
    sentiment_pipeline = pipeline(
        "zero-shot-classification", 
        model=model_path,
        tokenizer=model_path
    )
    print("Model loaded successfully from local directory.")

def analyze_sentiment(df_comments: pd.DataFrame) -> SentimentResult:
    """Analyze sentiment of comments"""
    if df_comments.empty:
        raise HTTPException(status_code=404, detail="No comments found for this video")
    
    # Run sentiment analysis on all comments
    try:
        results = sentiment_pipeline(df_comments['text'].tolist(), candidate_labels=["POSITIVE", "NEGATIVE"], truncation=True)
        
        # Add sentiment results to dataframe
        df_comments['hf_sentiment'] = [r['labels'][0] for r in results]  # Use the top label as sentiment
        df_comments['hf_score'] = [r['scores'][0] for r in results]  # Use the score for the top label
        
        # Calculate percentages
        sentiment_counts = df_comments['hf_sentiment'].value_counts()
        total_comments = len(df_comments)
        
        positive_count = sentiment_counts.get('POSITIVE', 0)
        negative_count = sentiment_counts.get('NEGATIVE', 0)
        
        positive_percentage = (positive_count / total_comments) * 100
        negative_percentage = (negative_count / total_comments) * 100
        
        # Find most liked comments
        most_liked_positive = None
        most_liked_negative = None
        
        positive_comments = df_comments[df_comments['hf_sentiment'] == 'POSITIVE']
        negative_comments = df_comments[df_comments['hf_sentiment'] == 'NEGATIVE']
        
        if not positive_comments.empty:
            top_positive = positive_comments.sort_values(by='likeCount', ascending=False).iloc[0]
            most_liked_positive = Comment(
                author=top_positive['author'],
                text=top_positive['text'],
                likeCount=int(top_positive['likeCount']),
                publishedAt=top_positive['publishedAt'],
                sentiment_score=float(top_positive['hf_score'])
            )
        
        if not negative_comments.empty:
            top_negative = negative_comments.sort_values(by='likeCount', ascending=False).iloc[0]
            most_liked_negative = Comment(
                author=top_negative['author'],
                text=top_negative['text'],
                likeCount=int(top_negative['likeCount']),
                publishedAt=top_negative['publishedAt'],
                sentiment_score=float(top_negative['hf_score'])
            )
        
        # Get top positive comments by sentiment score
        top_positive_comments = []
        if not positive_comments.empty:
            top_positive_df = positive_comments.sort_values(by='hf_score', ascending=False).head(5)
            for _, row in top_positive_df.iterrows():
                top_positive_comments.append(Comment(
                    author=row['author'],
                    text=row['text'],
                    likeCount=int(row['likeCount']),
                    publishedAt=row['publishedAt'],
                    sentiment_score=float(row['hf_score'])
                ))
        
        # Get top negative comments by sentiment score
        top_negative_comments = []
        if not negative_comments.empty:
            top_negative_df = negative_comments.sort_values(by='hf_score', ascending=False).head(5)
            for _, row in top_negative_df.iterrows():
                top_negative_comments.append(Comment(
                    author=row['author'],
                    text=row['text'],
                    likeCount=int(row['likeCount']),
                    publishedAt=row['publishedAt'],
                    sentiment_score=float(row['hf_score'])
                ))
        
        return SentimentResult(
            positive_percentage=positive_percentage,
            negative_percentage=negative_percentage,
            total_comments=total_comments,
            most_liked_positive=most_liked_positive,
            most_liked_negative=most_liked_negative,
            top_positive_comments=top_positive_comments,
            top_negative_comments=top_negative_comments
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.get("/")
async def root():
    return {"message": "YouTube Sentiment Analysis API"}

@app.post("/analyze", response_model=SentimentResult)
async def analyze_video(request: VideoRequest):
    """Analyze sentiment of YouTube video comments"""
    try:
        # Extract video ID
        video_id = extract_video_id(request.video_url)
        
        # Get comments
        df_comments = get_video_comments(video_id)
        
        # Analyze sentiment
        result = analyze_sentiment(df_comments)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
