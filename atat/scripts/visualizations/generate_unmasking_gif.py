#!/usr/bin/env python3
"""
ATAT Clean White Background Visualization

Modern minimal design with white background and colored text.
Words are colored by frequency/importance - no subtokens.

Frequency Thresholds:
- GREEN (Low importance): Top ~100 most common English words (function words)
- YELLOW (Medium importance): Next ~200 common words (basic nouns/verbs)
- RED (High importance): Rare/technical words, proper nouns, long words

You can adjust thresholds by modifying IMPORTANCE_THRESHOLDS.
"""

import sys
import os
import numpy as np
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import re


# ============================================================================
# FREQUENCY THRESHOLDS - Adjust these to change color assignments
# ============================================================================

IMPORTANCE_THRESHOLDS = {
    'low': 0.33,      # Below this = GREEN (function words)
    'medium': 0.66,   # Below this but above low = YELLOW (common words)
    # Above medium = RED (rare/content words)
}

# ============================================================================
# WORD FREQUENCY LISTS
# Based on English word frequency from large corpora
# ============================================================================

# TOP ~100 most frequent words (function words) → LOW importance → GREEN
# These carry grammatical meaning but little semantic content
FUNCTION_WORDS = {
    # Articles & determiners
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all',
    'both', 'few', 'many', 'much', 'most', 'other', 'another', 'such',
    
    # Pronouns
    'i', 'me', 'you', 'he', 'him', 'she', 'it', 'we', 'us', 'they', 'them',
    'who', 'whom', 'whose', 'which', 'what', 'whoever', 'whatever',
    
    # Prepositions
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'over', 'out', 'up', 'down', 'off', 'about', 'against', 'among',
    
    # Conjunctions
    'and', 'or', 'but', 'if', 'because', 'although', 'while', 'when', 'where',
    'unless', 'since', 'so', 'yet', 'nor', 'though', 'whereas',
    
    # Auxiliary verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    
    # Common adverbs
    'not', 'very', 'just', 'also', 'only', 'even', 'still', 'already', 'always',
    'never', 'ever', 'often', 'sometimes', 'usually', 'really', 'quite',
    'too', 'here', 'there', 'now', 'then', 'how', 'why', 'well',
}

# MEDIUM frequency words (~200) → MEDIUM importance → YELLOW
# Common nouns and verbs - carry meaning but are predictable
MEDIUM_WORDS = {
    # Common nouns
    'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'world',
    'life', 'hand', 'part', 'place', 'case', 'week', 'thing', 'fact', 'point',
    'home', 'water', 'room', 'mother', 'area', 'money', 'story', 'month',
    'lot', 'right', 'study', 'book', 'eye', 'job', 'word', 'business', 'issue',
    'side', 'kind', 'head', 'house', 'service', 'friend', 'father', 'power',
    'hour', 'game', 'line', 'end', 'member', 'law', 'car', 'city', 'name',
    'president', 'team', 'minute', 'idea', 'kid', 'body', 'information',
    'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health',
    'person', 'art', 'war', 'history', 'party', 'result', 'change', 'morning',
    'reason', 'research', 'moment', 'air', 'teacher', 'force', 'education',
    'student', 'group', 'country', 'problem', 'school', 'state', 'family',
    'government', 'company', 'system', 'program', 'question', 'work', 'number',
    
    # Common verbs
    'say', 'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look',
    'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel',
    'try', 'leave', 'call', 'need', 'become', 'keep', 'let', 'begin', 'show',
    'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring', 'happen',
    'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
    'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch',
    
    # Common adjectives
    'new', 'good', 'first', 'last', 'long', 'great', 'little', 'own', 'old',
    'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
    'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'human',
    'local', 'late', 'hard', 'major', 'better', 'economic', 'strong', 'possible',
    'whole', 'free', 'military', 'true', 'federal', 'international', 'full',
    'special', 'easy', 'clear', 'recent', 'certain', 'personal', 'open', 'red',
    'difficult', 'available', 'likely', 'short', 'single', 'medical', 'current',
}


def get_word_importance(word: str) -> float:
    """
    Calculate importance score based on word frequency.
    
    Logic:
    1. Function words (the, a, is, to) → 0.15 (LOW/GREEN)
    2. Medium frequency words (time, work, make) → 0.5 (MEDIUM/YELLOW)
    3. Proper nouns (capitalized) → 0.85 (HIGH/RED)
    4. Long words (>8 chars) → 0.9 (HIGH/RED) - likely technical
    5. Medium length unknown words → 0.7 (HIGH/RED)
    
    Returns float in [0, 1]
    """
    # Clean word for lookup
    clean = word.lower().strip()
    clean_alpha = re.sub(r'[^a-z]', '', clean)
    
    # Empty or punctuation only
    if not clean_alpha:
        return 0.1  # Treat as low importance
    
    # Check function words (most common) → LOW importance
    if clean_alpha in FUNCTION_WORDS:
        return 0.15  # GREEN
    
    # Check medium frequency words → MEDIUM importance
    if clean_alpha in MEDIUM_WORDS:
        return 0.5  # YELLOW
    
    # === HIGH importance indicators (RED) ===
    
    # Proper nouns (capitalized and not sentence start indicator)
    if word and word[0].isupper() and len(clean_alpha) > 1:
        return 0.85
    
    # Numbers often carry specific meaning
    if any(c.isdigit() for c in word):
        return 0.7
    
    # Very long words are usually technical/specialized
    if len(clean_alpha) > 8:
        return 0.9
    
    # Long words likely content words
    if len(clean_alpha) > 5:
        return 0.75
    
    # Short unknown words - default to medium-high
    return 0.65


def importance_to_color(importance: float) -> str:
    """
    Convert importance score to color.
    
    Uses IMPORTANCE_THRESHOLDS to determine cutoffs.
    """
    if importance < IMPORTANCE_THRESHOLDS['low']:
        return '#27ae60'  # Green - low importance
    elif importance < IMPORTANCE_THRESHOLDS['medium']:
        return '#e67e22'  # Orange - medium importance (visible on white)
    else:
        return '#c0392b'  # Dark red - high importance


def create_white_visualization(
    text: str,
    output_path: str = "atat_white.gif",
    num_steps: int = 60,
    fps: int = 4,
):
    """
    Create clean visualization with white background.
    """
    
    print(f"\n{'='*60}")
    print("ATAT Importance Visualization (White Background)")
    print(f"{'='*60}")
    
    # Split into words (full words, not subtokens)
    words = text.split()
    
    # Calculate importance for each word
    word_data = []
    for word in words:
        imp = get_word_importance(word)
        word_data.append({
            'text': word,
            'importance': imp,
            'color': importance_to_color(imp),
        })
    
    # Statistics
    high = [w for w in word_data if w['importance'] >= IMPORTANCE_THRESHOLDS['medium']]
    med = [w for w in word_data if IMPORTANCE_THRESHOLDS['low'] <= w['importance'] < IMPORTANCE_THRESHOLDS['medium']]
    low = [w for w in word_data if w['importance'] < IMPORTANCE_THRESHOLDS['low']]
    
    print(f"\nThresholds: low < {IMPORTANCE_THRESHOLDS['low']}, "
          f"medium < {IMPORTANCE_THRESHOLDS['medium']}")
    print(f"\nWord breakdown ({len(word_data)} total):")
    print(f"  🔴 HIGH (RED):    {len(high):2d} words - content/rare")
    print(f"  🟡 MEDIUM (ORANGE): {len(med):2d} words - common nouns/verbs")
    print(f"  🟢 LOW (GREEN):   {len(low):2d} words - function words")
    
    # Print word assignments for transparency
    print(f"\nWord assignments:")
    print(f"  RED: {', '.join([w['text'] for w in high][:10])}{'...' if len(high) > 10 else ''}")
    print(f"  ORANGE: {', '.join([w['text'] for w in med][:10]) if med else '(none)'}")
    print(f"  GREEN: {', '.join([w['text'] for w in low][:10])}{'...' if len(low) > 10 else ''}")
    
    # Sort by importance (highest first = revealed first)
    sorted_indices = sorted(range(len(word_data)),
                           key=lambda i: word_data[i]['importance'],
                           reverse=True)
    
    # Create reveal schedule
    reveal_step = {}
    for rank, idx in enumerate(sorted_indices):
        progress = rank / len(sorted_indices)
        step = int(progress * num_steps)
        reveal_step[idx] = step
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_facecolor('white')
        
        # Build display words
        display_words = []
        for i, w in enumerate(word_data):
            if frame >= reveal_step[i]:
                display_words.append({
                    'text': w['text'],
                    'color': w['color'],
                    'revealed': True
                })
            else:
                # Use [MASK] token
                display_words.append({
                    'text': '⬚⬚⬚⬚',
                    # 'text': '⬚█____ [MASK]',
                    'color': "#999999",  # Gray mask
                    'revealed': False
                })
        
        # Wrap into lines
        lines = []
        current_line = []
        current_len = 0
        max_chars = 50
        
        for wd in display_words:
            wlen = len(wd['text']) + 1
            if current_len + wlen > max_chars and current_line:
                lines.append(current_line)
                current_line = []
                current_len = 0
            current_line.append(wd)
            current_len += wlen
        if current_line:
            lines.append(current_line)
        
        # Calculate vertical positioning (center)
        line_height = 0.12
        total_height = len(lines) * line_height
        start_y = 0.5 + total_height / 2 - line_height / 2
        
        # Draw each line
        for line_idx, line_words in enumerate(lines):
            y = start_y - line_idx * line_height
            x = 0.05
            
            for wd in line_words:
                ax.text(x, y, wd['text'],
                       fontsize=22,
                       fontfamily='DejaVu Sans Mono',
                       fontweight='bold',
                       color=wd['color'],
                       va='center',
                       ha='left')
                
                # Advance x position
                x += len(wd['text']) * 0.019 + 0.02
    
    # Create animation
    print(f"\nRendering {num_steps} frames at {fps} FPS...")
    anim = FuncAnimation(fig, update, frames=num_steps,
                        interval=1000//fps, repeat=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=120)
    plt.close(fig)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {size_kb:.1f} KB")
    print(f"   Duration: {num_steps/fps:.1f} seconds")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="ATAT visualization with white background"
    )
    parser.add_argument("--text", type=str, default=None,
                       help="Text to visualize")
    parser.add_argument("--output", type=str,
                       default="results/visualizations/atat_white.gif")
    parser.add_argument("--num_steps", type=int, default=60,
                       help="Number of animation frames")
    parser.add_argument("--fps", type=int, default=4,
                       help="Frames per second (lower = slower)")
    
    args = parser.parse_args()
    
    if args.text is None:
        args.text = (
            "The brilliant scientist discovered a revolutionary "
            "breakthrough in quantum computing that transformed "
            "our understanding of artificial intelligence."
        )
    
    print(f"\nInput text:\n{args.text}")
    
    create_white_visualization(
        text=args.text,
        output_path=args.output,
        num_steps=args.num_steps,
        fps=args.fps
    )
    
    print("\n" + "="*60)
    print("HOW FREQUENCY THRESHOLDS WORK")
    print("="*60)
    print("""
Words are assigned importance based on frequency in English:

1. FUNCTION WORDS (GREEN) - importance < 0.33
   - The ~100 most common words: the, a, is, to, and, of, in...
   - These are grammatical "glue" - predictable from context
   - Revealed LAST

2. COMMON WORDS (ORANGE) - 0.33 <= importance < 0.66  
   - Next ~200 common words: time, work, people, make, think...
   - Basic nouns and verbs - carry meaning but common
   - Revealed SECOND

3. CONTENT WORDS (RED) - importance >= 0.66
   - Rare words, technical terms, proper nouns
   - Also: any word > 8 characters (likely specialized)
   - Also: capitalized words (proper nouns)
   - These carry the most unique information
   - Revealed FIRST

To adjust thresholds, edit IMPORTANCE_THRESHOLDS in the script:
  IMPORTANCE_THRESHOLDS = {
      'low': 0.33,    # Increase to make more words GREEN
      'medium': 0.66, # Increase to make more words ORANGE
  }
""")


if __name__ == "__main__":
    main()
