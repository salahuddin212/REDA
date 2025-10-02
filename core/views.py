# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django import forms
from django.http import JsonResponse
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import Counter
import base64
from io import BytesIO
import matplotlib
from dotenv import load_dotenv
import os
import config
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import time
import logging
import praw

# Set up logging
logger = logging.getLogger(__name__)

# Configure matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")


class SubredditForm(forms.Form):
    subreddit_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter subreddit name (e.g., python, datascience)',
            'required': True
        }),
        help_text="Enter the name without 'r/' prefix"
    )
    
    posts_limit = forms.IntegerField(
        initial=500,
        min_value=100,
        max_value=2000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '500'
        }),
        help_text="Number of posts to analyze (100-2000)"
    )
    
    def clean_subreddit_name(self):
        subreddit_name = self.cleaned_data['subreddit_name'].strip()
        
        # Remove r/ prefix if user included it
        if subreddit_name.startswith('r/'):
            subreddit_name = subreddit_name[2:]
        
        # Basic validation for subreddit name format
        if not re.match(r'^[A-Za-z0-9_]+$', subreddit_name):
            raise forms.ValidationError(
                "Subreddit name can only contain letters, numbers, and underscores."
            )
        
        if len(subreddit_name) < 2:
            raise forms.ValidationError("Subreddit name must be at least 2 characters long.")
        
        return subreddit_name.lower()


def subreddit_input_view(request):
    """Subreddit input form view"""
    if request.method == 'POST':
        form = SubredditForm(request.POST)
        if form.is_valid():
            subreddit_name = form.cleaned_data['subreddit_name']
            posts_limit = form.cleaned_data['posts_limit']
            # Store in session for later use
            request.session['subreddit_name'] = subreddit_name
            request.session['posts_limit'] = posts_limit
            messages.success(request, f'Starting EDA for r/{subreddit_name} with {posts_limit} posts...')
            return redirect('eda_results', subreddit=subreddit_name)
    else:
        form = SubredditForm()
    
    return render(request, 'subreddit_input.html', {'form': form})




load_dotenv()
def fetch_reddit_data_praw(subreddit_name, limit=500):
    """Fetch Reddit data using PRAW (Python Reddit API Wrapper)"""
    try:
        print(f"Fetching {limit} posts from r/{subreddit_name} using PRAW...")
        
        # Initialize Reddit instance with your credentials
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
        # Get subreddit
        subreddit = reddit.subreddit(subreddit_name)
        
        # Fetch posts
        posts = []
        
        # Use .new() to get recent posts (you can also use .hot(), .top(), etc.)
        for submission in subreddit.new(limit=limit):
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'author': str(submission.author) if submission.author else '[deleted]',
                'selftext': submission.selftext,
                'url': submission.url
            })
            
            # Small delay to respect Reddit's API rate limits
            time.sleep(0.05)
        
        if not posts:
            return None, "No posts found for this subreddit."
        
        # Convert to DataFrame
        df = pd.DataFrame(posts)
        
        # Clean and process data
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        df['title'] = df['title'].astype(str)
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce').fillna(0)
        df['title_length'] = df['title'].str.len()
        df['title_words'] = df['title'].str.split().str.len()
        df['day_of_week'] = df['created_utc'].dt.day_name()
        df['hour'] = df['created_utc'].dt.hour
        df['date'] = df['created_utc'].dt.date
        
        # Remove any rows with invalid data
        df = df.dropna(subset=['title', 'created_utc'])
        
        print(f"Successfully fetched {len(df)} posts from r/{subreddit_name}")
        return df, None
        
    except praw.exceptions.PRAWException as e:
        logger.error(f"PRAW error: {e}")
        return None, f"Reddit API error: {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching Reddit data: {e}")
        return None, f"Error fetching data: {str(e)}"


def generate_plots(df, subreddit_name):
    """Generate all plots for EDA"""
    plots = {}
    
    try:
        # Set consistent style
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
        
        # 1. Posts per day of week
        fig, ax = plt.subplots(figsize=(10, 6))
        day_counts = df['day_of_week'].value_counts()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = day_counts.reindex(day_order, fill_value=0)
        
        bars = ax.bar(day_counts.index, day_counts.values, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_title(f'Posts per Day of Week - r/{subreddit_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Posts')
        ax.set_xlabel('Day of Week')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plots['day_of_week'] = plot_to_base64(fig)
        plt.close()
        
        # 2. Posts per hour heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        hour_day = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        hour_day = hour_day.reindex(day_order, fill_value=0)
        
        if not hour_day.empty:
            sns.heatmap(hour_day, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Posts'})
        ax.set_title(f'Posting Activity Heatmap - r/{subreddit_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Day of Week')
        ax.set_xlabel('Hour of Day (24h format)')
        plt.tight_layout()
        plots['hour_heatmap'] = plot_to_base64(fig)
        plt.close()
        
        # 3. Score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = df['score'][df['score'] >= 0]
        ax.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        ax.set_title(f'Distribution of Post Scores - r/{subreddit_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Score (Upvotes)')
        ax.set_ylabel('Frequency')
        mean_score = scores.mean()
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['score_dist'] = plot_to_base64(fig)
        plt.close()
        
        # 4. Comments distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        comments = df['num_comments'][df['num_comments'] >= 0]
        ax.hist(comments, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.set_title(f'Distribution of Comments per Post - r/{subreddit_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Comments')
        ax.set_ylabel('Frequency')
        mean_comments = comments.mean()
        ax.axvline(mean_comments, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_comments:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['comments_dist'] = plot_to_base64(fig)
        plt.close()
        
        # 5. Posts over time
        fig, ax = plt.subplots(figsize=(12, 6))
        daily_posts = df.groupby('date').size().sort_index()
        if not daily_posts.empty:
            ax.plot(daily_posts.index, daily_posts.values, marker='o', linewidth=2, markersize=4, color='purple')
            ax.set_title(f'Posts Over Time - r/{subreddit_name}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Posts')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        plt.tight_layout()
        plots['posts_over_time'] = plot_to_base64(fig)
        plt.close()
        
        # 6. Word Cloud
        try:
            all_titles = ' '.join(df['title'].dropna().astype(str))
            if len(all_titles.strip()) > 0:
                # Enhanced stopwords
                common_words = {
                    'reddit', 'post', 'posts', 'comment', 'comments', 'sub', 'subreddit', 
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                    'by', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 
                    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
                    'might', 'must', 'shall', 'can', 'is', 'am', 'i', 'you', 'he', 'she', 
                    'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 
                    'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
                    'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now', 
                    'what', "from", "using",
                }
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    stopwords=common_words, 
                    max_words=100,
                    colormap='viridis',
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(all_titles)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Most Common Words in Titles - r/{subreddit_name}', 
                            fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                plots['wordcloud'] = plot_to_base64(fig)
                plt.close()
            else:
                plots['wordcloud'] = None
        except Exception as e:
            print(f"Error generating wordcloud: {e}")
            plots['wordcloud'] = None
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        print(f"Error generating plots: {e}")
    
    return plots


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting plot to base64: {e}")
        return None


def calculate_basic_stats(df):
    """Calculate basic statistics"""
    try:
        stats = {
            'total_posts': len(df),
            'date_range_start': df['created_utc'].min().strftime('%Y-%m-%d'),
            'date_range_end': df['created_utc'].max().strftime('%Y-%m-%d'),
            'avg_score': float(df['score'].mean()),
            'median_score': float(df['score'].median()),
            'avg_comments': float(df['num_comments'].mean()),
            'median_comments': float(df['num_comments'].median()),
            'avg_title_length': float(df['title_words'].mean()),
            'most_active_day': df['day_of_week'].mode().iloc[0] if not df['day_of_week'].mode().empty else 'N/A',
            'most_active_hour': int(df['hour'].mode().iloc[0]) if not df['hour'].mode().empty else 0,
            'top_posts': df.nlargest(5, 'score')[['title', 'score', 'num_comments']].to_dict('records')
        }
        return stats
    except Exception as e:
        logger.error(f"Error calculating basic stats: {e}")
        return {
            'total_posts': 0,
            'date_range_start': 'N/A',
            'date_range_end': 'N/A',
            'avg_score': 0,
            'median_score': 0,
            'avg_comments': 0,
            'median_comments': 0,
            'avg_title_length': 0,
            'most_active_day': 'N/A',
            'most_active_hour': 0,
            'top_posts': []
        }


def get_word_frequency(df, top_n=20):
    """Get most common words in titles"""
    try:
        all_titles = ' '.join(df['title'].dropna().astype(str).str.lower())
        
        if len(all_titles.strip()) == 0:
            return []
        
        # Remove common words and punctuation
        import string
        common_words = {
            'reddit', 'post', 'posts', 'comment', 'comments', 'sub', 'subreddit', 
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
            'might', 'must', 'shall', 'can', 'is', 'am', 'i', 'you', 'he', 'she', 
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 
            'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now'
        }
        
        # Clean and split text
        words = all_titles.translate(str.maketrans('', '', string.punctuation)).split()
        words = [word.strip() for word in words if word.strip() not in common_words and len(word.strip()) > 2]
        
        word_freq = Counter(words).most_common(top_n)
        return word_freq
    except Exception as e:
        logger.error(f"Error calculating word frequency: {e}")
        return []


def eda_results_view(request, subreddit):
    """EDA results view with actual analysis"""
    try:
        # Get posts limit from session
        posts_limit = request.session.get('posts_limit', 500)
        
        # Fetch data using PRAW
        print(f"Starting EDA for r/{subreddit}")
        df, error = fetch_reddit_data_praw(subreddit, posts_limit)
        
        if error or df is None:
            context = {
                'subreddit': subreddit,
                'error': error or 'Failed to fetch data'
            }
            return render(request, 'eda_results.html', context)
        
        if len(df) == 0:
            context = {
                'subreddit': subreddit,
                'error': 'No posts found for this subreddit'
            }
            return render(request, 'eda_results.html', context)
        
        print(f"Analyzing {len(df)} posts for r/{subreddit}")
        
        # Generate analysis
        basic_stats = calculate_basic_stats(df)
        word_frequency = get_word_frequency(df)
        plots = generate_plots(df, subreddit)
        
        print(f"Analysis complete for r/{subreddit}")
        
        context = {
            'subreddit': subreddit,
            'basic_stats': basic_stats,
            'word_frequency': word_frequency,
            'plots': plots,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"EDA analysis failed: {e}")
        context = {
            'subreddit': subreddit,
            'error': f'Analysis failed: {str(e)}'
        }
    
    return render(request, 'eda_results.html', context)