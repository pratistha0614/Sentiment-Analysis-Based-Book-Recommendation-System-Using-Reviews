import sqlite3
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    g,
    flash,
)
from werkzeug.security import generate_password_hash, check_password_hash
import pickle

import os
import re
import math
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import functools


popular_df = pickle.load(open("model/popular.pkl", "rb"))
books = pickle.load(open("model/books.pkl", "rb"))

# BERT-based sentiment analysis (lazy loading)
# Libraries will only be imported when sentiment analysis is first used
hf_tokenizer = None
hf_model = None
bert_id2label = None
bert_device = None

# Inference and ranking configuration
BERT_TEMP = 1.0  # temperature for probability calibration (T=1 means no scaling)
CONFIDENCE_THRESHOLD = 0.75  # default gating threshold
SHORT_TEXT_MIN_CHARS = 12  # abstain to fallback for very short texts
WL_LAMBDA = 0.7  # blend weight for personalized match vs Wilson lower bound


def load_bert_model():
    """Lazy load BERT model only when needed."""
    global hf_tokenizer, hf_model, bert_id2label, bert_device

    if hf_model is not None:
        return hf_model

    bert_device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    hf_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    hf_model.to(bert_device)
    hf_model.eval()
    id2label = getattr(hf_model.config, "id2label", None) or {}
    norm_map = {}
    for k, v in id2label.items():
        lv = str(v).lower()
        if ("neg" in lv) or lv.endswith("0") or (lv == "label_0"):
            norm_map[int(k)] = "NEGATIVE"
        elif ("neu" in lv) or lv.endswith("1") or (lv == "label_1"):
            norm_map[int(k)] = "NEUTRAL"
        else:
            norm_map[int(k)] = "POSITIVE"
    bert_id2label = norm_map
    return hf_model


def preprocess_text(text):
    t = text or ""
    t = t.strip()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


@functools.lru_cache(maxsize=4096)
def _cached_predict(text_norm):
    load_bert_model()
    inputs = hf_tokenizer(text_norm, return_tensors="pt", truncation=True, max_length=512)
    if bert_device == "cuda":
        inputs = {k: v.to(bert_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits
        # Temperature scaling
        probs_t = torch.softmax(logits / BERT_TEMP, dim=-1)
        probs = probs_t.detach().cpu().numpy()[0]
    idx = int(probs.argmax())
    label = bert_id2label.get(idx, "NEUTRAL")
    conf = float(probs[idx])
    scores = {}
    for i, p in enumerate(probs):
        scores[bert_id2label.get(i, str(i))] = float(p)
    return label, conf, scores


def bert_predict_batch(texts):
    """Batch BERT prediction with temperature scaling.
    Returns list of (label, conf, scores_dict) aligned with input order.
    """
    load_bert_model()
    # Preprocess and keep mapping
    norm_texts = [preprocess_text(t or "") for t in texts]
    if not norm_texts:
        return []
    inputs = hf_tokenizer(norm_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    if bert_device == "cuda":
        inputs = {k: v.to(bert_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits
        probs_t = torch.softmax(logits / BERT_TEMP, dim=-1)
        probs_np = probs_t.detach().cpu().numpy()
    results = []
    for row in probs_np:
        idx = int(row.argmax())
        label = bert_id2label.get(idx, "NEUTRAL")
        conf = float(row[idx])
        scores = {bert_id2label.get(i, str(i)): float(p) for i, p in enumerate(row)}
        results.append((label, conf, scores))
    return results


def bert_predict_label_and_conf(text):
    text_norm = preprocess_text(text or "")
    return _cached_predict(text_norm)




def analyze_sentiment_hybrid_with_probs(text: str, threshold: float = CONFIDENCE_THRESHOLD):
    """Like analyze_sentiment_hybrid but also returns per-class probabilities.
    Returns: (label, bert_conf, pos_p, neu_p, neg_p)
    """
    # Default probabilities when unavailable
    pos_p = neu_p = neg_p = 0.0
    txt_norm = preprocess_text(text or "")
    # Try BERT first (with short-text abstention)
    use_fallback = False
    try:
        if len(txt_norm) < SHORT_TEXT_MIN_CHARS:
            use_fallback = True
            raise ValueError("short_text_abstain")
        label, conf, scores = bert_predict_label_and_conf(txt_norm)
        # Map to probs
        pos_p = float(scores.get("POSITIVE", 0.0))
        neu_p = float(scores.get("NEUTRAL", 0.0))
        neg_p = float(scores.get("NEGATIVE", 0.0))
        if conf < threshold or label == "NEUTRAL":
            use_fallback = True
    except Exception:
        use_fallback = True

    if use_fallback:
        # Enhanced fallback rules
        txt = (text or "").lower()
        positive_keywords = {
            "excellent", "amazing", "great", "love", "loved", "like", "liked",
            "wonderful", "fantastic", "awesome", "brilliant", "enjoyed",
            "recommend", "recommended", "incredible", "superb", "favorite",
            "good", "well", "nice"
        }
        negative_keywords = {
            "bad", "terrible", "awful", "hate", "hated", "dislike", "disliked",
            "boring", "poor", "waste", "worst", "disappointing", "meh",
            "confusing", "predictable", "not good", "not great"
        }
        intensifiers = {"very", "really", "extremely", "so"}
        pos_count = 0.0
        neg_count = 0.0
        for kw in positive_keywords:
            if f"not {kw}" in txt or f"never {kw}" in txt:
                neg_count += 1.0
            elif kw in txt:
                if any(f"{intf} {kw}" in txt for intf in intensifiers):
                    pos_count += 2.0
                else:
                    pos_count += 1.0
        for kw in negative_keywords:
            if f"not {kw}" in txt or f"never {kw}" in txt:
                pos_count += 1.0
            elif kw in txt:
                if any(f"{intf} {kw}" in txt for intf in intensifiers):
                    neg_count += 2.0
                else:
                    neg_count += 1.0
        if "!" in txt:
            if pos_count > neg_count:
                pos_count *= 1.2
            elif neg_count > pos_count:
                neg_count *= 1.2
        if pos_count > neg_count:
            return "POSITIVE", 0.0, 1.0, 0.0, 0.0
        elif neg_count > pos_count:
            return "NEGATIVE", 0.0, 0.0, 0.0, 1.0
        else:
            return "NEUTRAL", 0.0, 0.0, 1.0, 0.0

    # Use BERT result
    return label, float(conf), float(pos_p), float(neu_p), float(neg_p)

def track_search(username, search_query, search_type):
    """Track user search in search_history table"""
    if username:
        db = get_db()
        db.execute(
            "INSERT INTO search_history (username, book_title, search_type) VALUES (?, ?, ?)",
            (username, search_query, search_type),
        )
        db.commit()


def recompute_all_review_sentiments(batch_size: int = 200):
    """
    Recompute and persist BERT sentiment labels for all reviews in the database.
    Processes in simple batches to avoid excessive memory usage.
    """
    db = get_db()
    # Ensure model is loaded
    load_bert_model()
    # Count total
    total = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    offset = 0
    while offset < total:
        rows = db.execute(
            "SELECT id, review_text FROM reviews ORDER BY id LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break
        texts = [(rid, (text or "")) for (rid, text) in rows]
        # Preprocess and select those eligible for BERT batch
        norm_texts = [(rid, preprocess_text(t)) for (rid, t) in texts]
        eligible = [(rid, t) for (rid, t) in norm_texts if len(t) >= SHORT_TEXT_MIN_CHARS]
        ineligible = [(rid, t) for (rid, t) in norm_texts if len(t) < SHORT_TEXT_MIN_CHARS]
        preds_map = {}
        # Batch predict eligible texts
        try:
            if eligible:
                _, texts_only = zip(*eligible)
                batch_results = bert_predict_batch(list(texts_only))
                for (rid, _), (label, conf, scores) in zip(eligible, batch_results):
                    pos_p = float(scores.get("POSITIVE", 0.0))
                    neu_p = float(scores.get("NEUTRAL", 0.0))
                    neg_p = float(scores.get("NEGATIVE", 0.0))
                    # Apply gating; if low confidence or neutral, mark for fallback later
                    if conf < CONFIDENCE_THRESHOLD or label == "NEUTRAL":
                        preds_map[rid] = None  # fallback required
                    else:
                        preds_map[rid] = (label, conf, pos_p, neu_p, neg_p)
        except Exception:
            # If batch fails, we'll handle per-item fallback below
            preds_map = {}
        # Handle ineligible or low-confidence items with fallback
        for rid, t in ineligible + [(rid, txt) for (rid, txt) in eligible if preds_map.get(rid) is None]:
            try:
                flabel, fconf, fpos, fneu, fneg = analyze_sentiment_hybrid_with_probs(t)
                preds_map[rid] = (flabel, fconf, fpos, fneu, fneg)
            except Exception:
                preds_map.setdefault(rid, ("NEUTRAL", 0.0, 0.0, 1.0, 0.0))
        # Persist all
        for rid, _ in texts:
            try:
                label, conf, pos_p, neu_p, neg_p = preds_map.get(rid, ("NEUTRAL", 0.0, 0.0, 1.0, 0.0))
                score = 3 if label == "POSITIVE" else 2 if label == "NEUTRAL" else 1
                db.execute(
                    "UPDATE reviews SET sentiment = ?, confidence = ?, sentiment_score = ?, pos_p = ?, neu_p = ?, neg_p = ? WHERE id = ?",
                    (label, conf, score, pos_p, neu_p, neg_p, rid),
                )
            except Exception:
                # Fallback to minimal update
                try:
                    label, conf, pos_p, neu_p, neg_p = preds_map.get(rid, ("NEUTRAL", 0.0, 0.0, 1.0, 0.0))
                    score = 3 if label == "POSITIVE" else 2 if label == "NEUTRAL" else 1
                    db.execute(
                        "UPDATE reviews SET sentiment = ?, confidence = ?, sentiment_score = ? WHERE id = ?",
                        (label, conf, score, rid),
                    )
                except Exception:
                    pass
        db.commit()
        offset += batch_size

def track_visit(username, book_title):
    """Track user book visit in search_history table"""
    if username:
        db = get_db()
        db.execute(
            "INSERT INTO search_history (username, book_title, search_type) VALUES (?, ?, ?)",
            (username, book_title, "visit"),
        )
        db.commit()


def get_user_search_history(username, limit=10):
    """Get user's search and visit history"""
    if not username:
        return []

    db = get_db()
    history = db.execute(
        """SELECT book_title, search_type, timestamp 
           FROM search_history 
           WHERE username = ? AND search_type IN ('title', 'author', 'visit')
           ORDER BY timestamp DESC 
           LIMIT ?""",
        (username, limit),
    ).fetchall()
    return history


def get_user_recommendations(username):
    """
    Generate sentiment-driven recommendations.

    Logic:
    - For each book with reviews, compute per-book sentiment using stored labels:
      POSITIVE=+1, NEUTRAL=0, NEGATIVE=-1. Overall score is the average.
    - Compute positivity score = fraction of POSITIVE reviews.
    - Exclude books the user already reviewed (to promote discovery) and any they rated negatively.
    - Prioritize books with predominantly positive sentiment (positivity > 0.5) and
      sort by positivity desc, then average score desc.
    Returns a list of items: [title, positivity (0..1), author, image_url, avg_score (-1..1)].
    """
    db = get_db()

    # User history for filtering and dominant sentiment detection
    user_reviews = db.execute(
        "SELECT book_title, sentiment FROM reviews WHERE username = ?", (username,)
    ).fetchall()
    reviewed_books = {b for b, _ in user_reviews}
    disliked_books = {b for b, s in user_reviews if s == "NEGATIVE"}
    # Determine user's dominant sentiment
    u_pos = sum(1 for _, s in user_reviews if s == "POSITIVE")
    u_neu = sum(1 for _, s in user_reviews if s == "NEUTRAL")
    u_neg = sum(1 for _, s in user_reviews if s == "NEGATIVE")
    if u_pos >= u_neu and u_pos >= u_neg:
        user_dom = "POSITIVE"
    elif u_neg >= u_pos and u_neg >= u_neu:
        user_dom = "NEGATIVE"
    else:
        user_dom = "NEUTRAL"

    # Candidate books: any book that has at least one review
    book_rows = db.execute(
        "SELECT DISTINCT book_title FROM reviews"
    ).fetchall()
    if not book_rows:
        return None

    recommendations = []

    for (book_title,) in book_rows:
        # Skip books the user disliked or already reviewed
        if book_title in disliked_books or book_title in reviewed_books:
            continue

        sentiments = db.execute(
            "SELECT sentiment, COALESCE(confidence, 0.0) FROM reviews WHERE book_title = ?",
            (book_title,),
        ).fetchall()
        if not sentiments:
            continue

        # Counts per class
        pos = sum(1 for (s, _) in sentiments if s == "POSITIVE")
        neu = sum(1 for (s, _) in sentiments if s == "NEUTRAL")
        neg = sum(1 for (s, _) in sentiments if s == "NEGATIVE")
        # Confidence sums per class
        pos_conf_sum = sum(conf for (s, conf) in sentiments if s == "POSITIVE")
        neu_conf_sum = sum(conf for (s, conf) in sentiments if s == "NEUTRAL")
        neg_conf_sum = sum(conf for (s, conf) in sentiments if s == "NEGATIVE")
        total = pos + neu + neg
        if total == 0:
            continue

        # Average sentiment score in [-1,1] (for simple badge)
        avg_score = (pos * 1 + neu * 0 + neg * -1) / float(total)
        # Also compute label-scale average in [1..3]: 1=Neg,2=Neu,3=Pos
        label_avg = (neg * 1 + neu * 2 + pos * 3) / float(total)
        # Per-book distributions
        pos_pct = pos / float(total)
        neu_pct = neu / float(total)
        neg_pct = neg / float(total)
        # Per-class average confidences
        avg_conf_pos = (pos_conf_sum / pos) if pos > 0 else 0.0
        avg_conf_neu = (neu_conf_sum / neu) if neu > 0 else 0.0
        avg_conf_neg = (neg_conf_sum / neg) if neg > 0 else 0.0

        # Personalized match confidence using user dominant sentiment
        if user_dom == "POSITIVE":
            match_user = pos_pct
        elif user_dom == "NEUTRAL":
            match_user = neu_pct
        else:  # NEGATIVE
            match_user = 1.0 - neg_pct

        # Wilson lower bound on positive rate for robustness
        p_hat = pos_pct
        n = total
        z = 1.96
        wl = 0.0
        if n > 0:
            wl = (p_hat + (z*z)/(2*n) - z*math.sqrt((p_hat*(1-p_hat) + (z*z)/(4*n))/n)) / (1 + (z*z)/n)
            wl = max(0.0, min(1.0, wl))

        # Blend personalized match with Wilson lower bound
        match_conf = WL_LAMBDA * match_user + (1.0 - WL_LAMBDA) * wl
        match_label = user_dom

        # Filter: keep strong matches when possible
        if match_conf <= 0.5:
            # soft filter; collect but will be ranked lower
            pass

        # Book metadata from books dataframe
        temp_df = books[books["Book-Title"] == book_title]
        if temp_df.empty:
            continue
        book_author = temp_df.drop_duplicates("Book-Title")["Book-Author"].values[0]
        image_url = temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values[0]

        # Return tuple enriched with match label and per-class confidences
        recommendations.append([
            book_title,           # 0 title
            match_conf,           # 1 match confidence (0..1)
            book_author,          # 2 author
            image_url,            # 3 image
            avg_score,            # 4 avg sentiment (-1..1) for badge
            match_label,          # 5 match label string
            pos_pct,              # 6 positive share
            neu_pct,              # 7 neutral share
            neg_pct,              # 8 negative share
            label_avg,            # 9 avg label score in [1..3]
            avg_conf_pos,         # 10 avg confidence for positive reviews
            avg_conf_neu,         # 11 avg confidence for neutral reviews
            avg_conf_neg,         # 12 avg confidence for negative reviews
        ])

    if not recommendations:
        return None

    # Sort by match confidence desc, then avg_score desc
    recommendations.sort(key=lambda x: (x[1], x[4]), reverse=True)
    return recommendations[:8]


app = Flask(__name__)
# Ensure templates and assets reload so UI changes (e.g., removing Similarity labels) take effect
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Make is_admin available in all templates
@app.context_processor
def inject_is_admin():
    return dict(is_admin=is_admin)


# In-memory storage for reviews: {book_title: [(review_text, sentiment_label)]}
reviews = {}

app.secret_key = "your_secret_key_here"
DATABASE = "users.db"

UPLOAD_FOLDER = "static/profile_pics"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            profile_pic TEXT
        )""")
        # Add profile_pic column if it doesn't exist (for upgrades)
        try:
            db.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
        except Exception:
            pass
        # Add email column if it doesn't exist
        try:
            db.execute("ALTER TABLE users ADD COLUMN email TEXT")
        except Exception:
            pass
        # Add phone column if it doesn't exist
        try:
            db.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        except Exception:
            pass
        db.execute("""CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_title TEXT NOT NULL,
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            username TEXT NOT NULL
        )""")
        # Add confidence column if it doesn't exist (for upgrades)
        try:
            db.execute("ALTER TABLE reviews ADD COLUMN confidence REAL")
        except Exception:
            pass
        # Add numeric sentiment_score column if it doesn't exist (1=Neg,2=Neu,3=Pos)
        try:
            db.execute("ALTER TABLE reviews ADD COLUMN sentiment_score INTEGER")
        except Exception:
            pass
        # Add per-class probabilities if they don't exist
        try:
            db.execute("ALTER TABLE reviews ADD COLUMN pos_p REAL")
        except Exception:
            pass
        try:
            db.execute("ALTER TABLE reviews ADD COLUMN neu_p REAL")
        except Exception:
            pass
        try:
            db.execute("ALTER TABLE reviews ADD COLUMN neg_p REAL")
        except Exception:
            pass
        db.execute("""CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            book_title TEXT NOT NULL,
            search_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            book_title TEXT NOT NULL,
            UNIQUE(username, book_title)
        )""")
        db.commit()


# Add to Wishlist route
@app.route("/add_to_wishlist", methods=["POST"])
def add_to_wishlist():
    if "user_id" not in session:
        flash("Please log in to add books to your wishlist.", "warning")
        return redirect(url_for("login"))
    book_title = request.form.get("book_title")
    username = session["username"]
    db = get_db()
    try:
        db.execute(
            "INSERT OR IGNORE INTO wishlist (username, book_title) VALUES (?, ?)",
            (username, book_title),
        )
        db.commit()
        flash(f'"{book_title}" added to your wishlist!', "success")
    except Exception as e:
        flash("Could not add to wishlist.", "danger")
    # Redirect back to previous page
    referrer = request.referrer or url_for("index")
    return redirect(referrer)


# Wishlist page
@app.route("/wishlist")
def wishlist():
    if "user_id" not in session:
        flash("Please log in to view your wishlist.", "warning")
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()
    wishlist_books = db.execute(
        "SELECT book_title FROM wishlist WHERE username = ?", (username,)
    ).fetchall()
    # Get book details for each wishlist book
    wishlist_details = []
    for (book_title,) in wishlist_books:
        book_df = books[books["Book-Title"] == book_title].drop_duplicates("Book-Title")
        if not book_df.empty:
            wishlist_details.append(
                {
                    "title": book_df["Book-Title"].values[0],
                    "author": book_df["Book-Author"].values[0],
                    "image": book_df["Image-URL-M"].values[0],
                }
            )
    return render_template("wishlist.html", wishlist=wishlist_details)


init_db()

# Preload BERT model once at startup to avoid first-request latency (optional)
try:
    load_bert_model()
    # Recompute sentiments for all existing reviews to ensure consistency
    recompute_all_review_sentiments()
except Exception:
    pass


@app.route("/register", methods=["GET", "POST"])
def register():
    db = get_db()
    form_data = {}
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        form_data = {"username": username, "email": email, "phone": phone}

        errors = {}

        if not re.fullmatch(r"^[A-Za-z][A-Za-z0-9_]{2,}$", username):
            errors["username"] = "Username must start with a letter and be at least 3 characters long"

        email_pattern = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
        if not email_pattern.match(email):
            errors["email"] = "Please enter a valid email address"

        if not re.fullmatch(r"\d{10}", phone):
            errors["phone"] = "Phone number must be exactly 10 digits"

        if len(password) < 8:
            errors["password"] = "Password must be at least 8 characters"

        if errors:
            return render_template("register.html", form_data=form_data, errors=errors)

        try:
            db.execute(
                "INSERT INTO users (username, password, role, email, phone) VALUES (?, ?, ?, ?, ?)",
                (username, generate_password_hash(password), "user", email, phone),
            )
            db.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            errors = {"username": "Username already exists."}
            return render_template("register.html", form_data=form_data, errors=errors)

    return render_template("register.html", form_data=form_data)


# Helper to check if current user is admin
def is_admin():
    if "user_id" not in session:
        return False
    db = get_db()
    user = db.execute(
        "SELECT role FROM users WHERE id = ?", (session["user_id"],)
    ).fetchone()
    return user and user[0] == "admin"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            flash("Logged in successfully!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


@app.route("/profile")
def profile():
    if "user_id" not in session:
        flash("Please log in to view your profile.", "warning")
        return redirect(url_for("login"))
    username = session["username"]
    db = get_db()
    # Load profile_pic from users table
    user_row = db.execute(
        "SELECT profile_pic FROM users WHERE id = ?", (session["user_id"],)
    ).fetchone()
    profile_pic = user_row[0] if user_row and user_row[0] else None
    user_reviews = db.execute(
        "SELECT book_title, review_text, sentiment FROM reviews WHERE username = ?",
        (username,),
    ).fetchall()
    user_recommendations = db.execute(
        "SELECT book_title, timestamp FROM search_history WHERE username = ? AND search_type = ? ORDER BY timestamp DESC",
        (username, "recommendation"),
    ).fetchall()
    # Get the latest recommendation search
    latest_search = db.execute(
        "SELECT book_title FROM search_history WHERE username = ? AND search_type = ? ORDER BY timestamp DESC LIMIT 1",
        (username, "recommendation"),
    ).fetchone()
    # Use the same sentiment-based recommendations on profile
    user_recommended_books = get_user_recommendations(username) or []
    # Wishlist details
    wishlist_books = db.execute(
        "SELECT book_title FROM wishlist WHERE username = ?", (username,)
    ).fetchall()
    wishlist_details = []
    for (book_title,) in wishlist_books:
        book_df = books[books["Book-Title"] == book_title].drop_duplicates("Book-Title")
        if not book_df.empty:
            wishlist_details.append(
                {
                    "title": book_df["Book-Title"].values[0],
                    "author": book_df["Book-Author"].values[0],
                    "image": book_df["Image-URL-M"].values[0],
                }
            )
    wishlist_count = len(wishlist_details)
    recent_wishlist = wishlist_details[:5]
    # Admin stats
    total_users = total_reviews = total_wishlists = None
    if is_admin():
        total_users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_reviews = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        total_wishlists = db.execute("SELECT COUNT(*) FROM wishlist").fetchone()[0]
    return render_template(
        "profile.html",
        username=username,
        user_reviews=user_reviews,
        user_recommendations=user_recommendations,
        user_recommended_books=user_recommended_books,
        wishlist=wishlist_details,
        wishlist_count=wishlist_count,
        recent_wishlist=recent_wishlist,
        total_users=total_users,
        total_reviews=total_reviews,
        total_wishlists=total_wishlists,
        profile_pic=profile_pic,
    )


# Simple admin stats page for iframe (can be improved with charts later)
@app.route("/admin_stats")
def admin_stats():
    db = get_db()
    total_users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_reviews = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    total_wishlists = db.execute("SELECT COUNT(*) FROM wishlist").fetchone()[0]
    return f"""
    <div style='font-family:Inter,sans-serif;padding:30px;'>
      <h2 style='color:#764ba2;'>📊 Site Statistics</h2>
      <ul style='font-size:1.2rem;line-height:2;'>
        <li><b>Total Users:</b> {total_users}</li>
        <li><b>Total Reviews:</b> {total_reviews}</li>
        <li><b>Total Wishlist Items:</b> {total_wishlists}</li>
      </ul>
      <div style='color:#888;font-size:0.98rem;'>Charts coming soon!</div>
    </div>
    """


@app.route("/", methods=["GET", "POST"])
def index():
    search_results = None
    search_type = None
    search_query = None
    recommendations = None

    if request.method == "POST":
        search_type = request.form.get("search_type")
        search_query = request.form.get("search_query", "").strip().lower()

        # Track search if user is logged in
        if "username" in session and search_query:
            track_search(session["username"], search_query, search_type)

        if search_type == "author":
            filtered = books[
                books["Book-Author"].str.lower().str.contains(search_query, na=False)
            ]
            search_results = filtered if not filtered.empty else None
        elif search_type == "title":
            filtered = books[
                books["Book-Title"].str.lower().str.contains(search_query, na=False)
            ]
            search_results = filtered if not filtered.empty else None

    # Get recommendations for logged-in users based on their reviews
    recommendations = None
    search_history = None
    if "user_id" in session:
        recommendations = get_user_recommendations(session["username"])
        search_history = get_user_search_history(session["username"], 10)

    return render_template(
        "index.html",
        book_name=list(popular_df["Book-Title"].values),
        author=list(popular_df["Book-Author"].values),
        image=list(popular_df["Image-URL-M"].values),
        votes=list(popular_df["num_ratings"].values),
        rating=list(popular_df["avg_ratings"].values),
        search_results=search_results,
        search_type=search_type,
        search_query=search_query,
        recommendations=recommendations,
        search_history=search_history,
    )


@app.route("/book_details/<book_title>", methods=["GET", "POST"])
def book_details(book_title):
    from urllib.parse import unquote

    book_title = unquote(book_title)  # Decode URL-encoded title
    print(f"DEBUG: Book details requested for: {book_title}")  # Debug line

    # Track visit if user is logged in
    if "username" in session:
        track_visit(session["username"], book_title)

    book_details_df = books[books["Book-Title"] == book_title].drop_duplicates(
        "Book-Title"
    )

    # If exact match not found, try to find a close match
    if book_details_df.empty:
        # Try case-insensitive match
        book_details_df = books[
            books["Book-Title"].str.lower() == book_title.lower()
        ].drop_duplicates("Book-Title")

        # If still not found, try partial match
        if book_details_df.empty:
            book_details_df = books[
                books["Book-Title"].str.contains(book_title, case=False, na=False)
            ].drop_duplicates("Book-Title")

    if book_details_df.empty:
        flash(f"Book '{book_title}' not found", "danger")
        return redirect(url_for("index"))
    db = get_db()
    if request.method == "POST":
        if "user_id" not in session:
            flash("Please log in to submit a review.", "warning")
            return redirect(url_for("login"))
        review_text = request.form.get("review_text")
        if review_text:
            sentiment, confidence, pos_p, neu_p, neg_p = analyze_sentiment_hybrid_with_probs(review_text)
            sentiment_score = 3 if sentiment == "POSITIVE" else 2 if sentiment == "NEUTRAL" else 1
            username = session["username"]
            try:
                db.execute(
                    "INSERT INTO reviews (book_title, review_text, sentiment, confidence, sentiment_score, pos_p, neu_p, neg_p, username) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (book_title, review_text, sentiment, confidence, sentiment_score, pos_p, neu_p, neg_p, username),
                )
            except Exception:
                try:
                    db.execute(
                        "INSERT INTO reviews (book_title, review_text, sentiment, confidence, sentiment_score, username) VALUES (?, ?, ?, ?, ?, ?)",
                        (book_title, review_text, sentiment, confidence, sentiment_score, username),
                    )
                except Exception:
                    # Fallback insert in case column not yet available
                    db.execute(
                        "INSERT INTO reviews (book_title, review_text, sentiment, username) VALUES (?, ?, ?, ?)",
                        (book_title, review_text, sentiment, username),
                    )
            db.commit()
            flash("Thank you for the review!", "success")
        return redirect(url_for("book_details", book_title=book_title))
    book_reviews = db.execute(
        "SELECT review_text, sentiment FROM reviews WHERE book_title = ?", (book_title,)
    ).fetchall()
    sentiment_summary = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for _, sentiment in book_reviews:
        if sentiment in sentiment_summary:
            sentiment_summary[sentiment] += 1
        else:
            sentiment_summary["NEUTRAL"] += 1
    return render_template(
        "book_details.html",
        book_title=book_details_df["Book-Title"].values[0],
        author=book_details_df["Book-Author"].values[0],
        image=book_details_df["Image-URL-M"].values[0],
        publisher=book_details_df["Publisher"].values[0],
        year=book_details_df["Year-Of-Publication"].values[0],
        reviews=book_reviews,
        sentiment_summary=sentiment_summary,
    )


# API endpoint to get reviews (for AJAX, optional)
@app.route("/api/reviews/<book_title>")
def get_reviews(book_title):
    db = get_db()
    reviews_data = db.execute(
        "SELECT review_text, sentiment FROM reviews WHERE book_title = ?", (book_title,)
    ).fetchall()
    return jsonify([(r[0], r[1]) for r in reviews_data])


@app.route("/clear_search_history", methods=["POST"])
def clear_search_history():
    """Clear user's search history"""
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Not logged in"})

    db = get_db()
    db.execute("DELETE FROM search_history WHERE username = ?", (session["username"],))
    db.commit()

    return jsonify({"success": True, "message": "Search history cleared"})


@app.route("/About")
def about():
    return render_template("About.html")


# Admin dashboard route (correct placement)
@app.route("/admin")
def admin_dashboard():
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    return render_template("admin_dashboard.html")


# Admin: View all users
@app.route("/admin_users")
def admin_users():
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    users = db.execute("SELECT id, username, role FROM users").fetchall()
    users = [dict(id=u[0], username=u[1], role=u[2]) for u in users]
    return render_template("admin_users.html", users=users)


# Admin: Edit user role
@app.route("/admin_users/<int:user_id>/edit", methods=["POST"])
def admin_edit_user(user_id):
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    new_role = request.form.get("role")
    db = get_db()
    db.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    db.commit()
    flash("User role updated.", "success")
    return redirect(url_for("admin_users"))


# Admin: Reset user password
@app.route("/admin_users/<int:user_id>/reset_password", methods=["POST"])
def admin_reset_password(user_id):
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    new_password = request.form.get("new_password")
    db = get_db()
    db.execute(
        "UPDATE users SET password = ? WHERE id = ?",
        (generate_password_hash(new_password), user_id),
    )
    db.commit()
    flash("Password reset.", "success")
    return redirect(url_for("admin_users"))


# Admin: Delete user
@app.route("/admin_users/<int:user_id>/delete", methods=["POST"])
def admin_delete_user(user_id):
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    flash("User deleted.", "success")
    return redirect(url_for("admin_users"))


# Admin: View all reviews
@app.route("/admin_reviews")
def admin_reviews():
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    reviews = db.execute(
        "SELECT id, book_title, username, review_text, sentiment FROM reviews"
    ).fetchall()
    reviews = [
        dict(id=r[0], book_title=r[1], username=r[2], review_text=r[3], sentiment=r[4])
        for r in reviews
    ]
    return render_template("admin_reviews.html", reviews=reviews)


# Admin: Delete review
@app.route("/admin_reviews/<int:review_id>/delete", methods=["POST"])
def admin_delete_review(review_id):
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    db.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    db.commit()
    flash("Review deleted.", "success")
    return redirect(url_for("admin_reviews"))


# Admin: View all search history
@app.route("/admin_search_history")
def admin_search_history():
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    history = db.execute(
        "SELECT id, username, book_title, search_type, timestamp FROM search_history"
    ).fetchall()
    history = [
        dict(id=h[0], username=h[1], book_title=h[2], search_type=h[3], timestamp=h[4])
        for h in history
    ]
    return render_template("admin_search_history.html", history=history)


# Admin: View all wishlist items
@app.route("/admin_wishlist")
def admin_wishlist():
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    wishlist = db.execute("SELECT id, username, book_title FROM wishlist").fetchall()
    wishlist = [dict(id=w[0], username=w[1], book_title=w[2]) for w in wishlist]
    return render_template("admin_wishlist.html", wishlist=wishlist)


# Admin: Delete wishlist item
@app.route("/admin_wishlist/<int:wishlist_id>/delete", methods=["POST"])
def admin_delete_wishlist(wishlist_id):
    if not is_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("index"))
    db = get_db()
    db.execute("DELETE FROM wishlist WHERE id = ?", (wishlist_id,))
    db.commit()
    flash("Wishlist item deleted.", "success")
    return redirect(url_for("admin_wishlist"))


@app.route("/upload_profile_pic", methods=["POST"])
def upload_profile_pic():
    if "user_id" not in session:
        flash("Please log in to upload a profile picture.", "warning")
        return redirect(url_for("login"))
    if "profile_pic" not in request.files:
        flash("No file part.", "danger")
        return redirect(url_for("profile"))
    file = request.files["profile_pic"]
    if file.filename == "":
        flash("No selected file.", "danger")
        return redirect(url_for("profile"))
    if file and allowed_file(file.filename):
        filename = secure_filename(
            f"{session['username']}_profile.{file.filename.rsplit('.', 1)[1].lower()}"
        )
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        # Save filename in users table
        db = get_db()
        db.execute(
            "UPDATE users SET profile_pic = ? WHERE id = ?",
            (filename, session["user_id"]),
        )
        db.commit()
        flash("Profile picture updated!", "success")
    else:
        flash("Invalid file type.", "danger")
    return redirect(url_for("profile"))


@app.route("/remove_profile_pic", methods=["POST"])
def remove_profile_pic():
    if "user_id" not in session:
        flash("Please log in to remove your profile picture.", "warning")
        return redirect(url_for("login"))
    db = get_db()
    user_row = db.execute(
        "SELECT profile_pic FROM users WHERE id = ?", (session["user_id"],)
    ).fetchone()
    if user_row and user_row[0]:
        filename = user_row[0]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        db.execute(
            "UPDATE users SET profile_pic = NULL WHERE id = ?", (session["user_id"],)
        )
        db.commit()
        flash("Profile picture removed.", "success")
    else:
        flash("No profile picture to remove.", "info")
    return redirect(url_for("profile"))


if __name__ == "__main__":
    app.run(debug=True)


