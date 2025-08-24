from flask import Flask, render_template, session, request, redirect, url_for, flash
from modules.auth import auth_bp
import os
from dotenv import load_dotenv
from google import genai  # latest Gemini SDK
from datetime import datetime
from markupsafe import Markup, escape
import sqlite3
from flask import g
from ml.infer import predict_sentiment, predict_topics



# ------------------------
# Load environment
# ------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found in .env file")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)



# ------------------------
# Initialize Flask App
# ------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Register Authentication Blueprint
app.register_blueprint(auth_bp)

# ------------------------
# Copilot Tool Definitions
# ------------------------
COPILOT_TOOLS = {
    "market": {
        "title": "AI-Powered Market Analysis & Competitive Insights",
        "cta": "Analyze Market",
        "placeholder": "Describe your startup, target customers, and what problem you solve...",
        "prompt": (
            "You are a senior startup analyst. Provide concise, practical analysis.\n"
            "Return markdown with these sections:\n"
            "## Summary\n"
            "## Market Trends (3–5 bullets)\n"
            "## ICP & Customer Insights (3–5 bullets)\n"
            "## Competitor Benchmark (table with: Competitor | Offering | Strengths | Gaps)\n"
            "## Risks & Mitigations (bulleted)\n"
            "## Actionable Next Steps (numbered, high-impact)\n\n"
            "Startup context:\n{user_input}"
        )
    },
    "fundraising": {
        "title": "Fundraising & Investor Readiness",
        "cta": "Assess Investor Readiness",
        "placeholder": "Paste your pitch summary, traction, revenue model, team, and fundraising goal...",
        "prompt": (
            "Act as a VC analyst preparing a founder for fundraising.\n"
            "Return markdown with:\n"
            "## Readiness Score (0–100) with rationale\n"
            "## Pitch Deck Audit (slide-by-slide: Problem, Solution, Market, Product, GTM, Traction, Business Model, Competition, Team, Financials, Ask)\n"
            "## Risks & Diligence Questions (bullets)\n"
            "## Financial Snapshot Template (table)\n"
            "## Investor Targeting (5–10 relevant theses/firms)\n"
            "## Action Plan (0–30 days, 30–60 days)\n\n"
            "Founder input:\n{user_input}"
        )
    },
    "product": {
        "title": "Product Development & Growth Strategies",
        "cta": "Generate Strategy",
        "placeholder": "Describe your product vision, user journey, and current stage...",
        "prompt": (
            "You are a staff product manager and growth lead.\n"
            "Return markdown with:\n"
            "## MVP Scope (must-have features, acceptance criteria)\n"
            "## Architecture Sketch (high-level components)\n"
            "## Tech Stack Options (table: Layer | Option | Why | Trade-offs)\n"
            "## GTM Plan (channels, ICP messaging, lighthouse use cases)\n"
            "## North Star Metric & KPIs (definitions)\n"
            "## Experiment Backlog (5–8 tests with hypothesis, metric, effort/impact)\n\n"
            "Context:\n{user_input}"
        )
    },
    "mentor": {
        "title": "24/7 AI-Powered Startup Mentorship",
        "cta": "Get Mentor Advice",
        "placeholder": "Ask any startup question (hiring, pricing, legal, growth, ops)...",
        "prompt": (
            "You are a pragmatic startup mentor. Answer clearly and concisely.\n"
            "Return markdown with:\n"
            "## Direct Answer\n"
            "## Decision Factors (bullets)\n"
            "## Pitfalls to Avoid (bullets)\n"
            "## Playbook Steps (numbered checklist)\n\n"
            "Question:\n{user_input}"
        )
    },
    "accelerators": {
        "title": "Optimized for Accelerators & Incubators",
        "cta": "Optimize for YC/Techstars",
        "placeholder": "Paste your accelerator application draft or company summary...",
        "prompt": (
            "You help founders succeed in accelerators (YC, Techstars, Seedcamp).\n"
            "Return markdown with:\n"
            "## Acceptance Likelihood (Low/Med/High) + rationale\n"
            "## Application Edits (bullet improvements; be concrete and concise)\n"
            "## Interview Prep (10 likely questions + strong sample answers)\n"
            "## Milestones to Hit (next 6–8 weeks)\n"
            "## Social Proof Ideas (advisors, pilots, press, metrics)\n\n"
            "Application/company context:\n{user_input}"
        )
    }
}

# ------------------------
# Helpers
# ------------------------
def require_login():
    return "user_id" in session

def to_html_from_markdown(md_text: str) -> str:
    """
    Try to convert Markdown to HTML. If markdown package is unavailable,
    fall back to a safe <pre> block.
    """
    if not md_text:
        return ""
    try:
        import markdown
        html = markdown.markdown(
            md_text,
            extensions=["extra", "tables", "sane_lists", "toc"]
        )
        return html
    except Exception:
        # Safe fallback
        return f'<pre style="white-space:pre-wrap">{escape(md_text)}</pre>'

def call_gemini_markdown(system_prompt: str, user_input: str) -> str:
    """
    Calls Gemini 1.5 Flash and returns markdown text.
    """
    contents = system_prompt.format(user_input=user_input)
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=contents
    )
    return (resp.text or "").strip()

# ✅ Database setup
DATABASE = "database.db"

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    if 'db' in g:
        g.db.close()

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return redirect(url_for('auth.login'))

@app.route('/dashboard')
def dashboard():
    if not require_login():
        return redirect(url_for('auth.login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/home')
def home():
    if not require_login():
        return redirect(url_for('auth.login'))
    return render_template('home.html', username=session['username'])

# ------------------------
# Example Startup Generator
# ------------------------
@app.route('/generate', methods=['POST'])
def generate():
    if not require_login():
        return redirect(url_for('auth.login'))

    idea = request.form.get('idea', '').strip()
    user_label = request.form.get('label', '').strip()  # User may give custom label

    if not idea:
        flash("Please enter your startup idea.", "danger")
        return redirect(url_for('home'))

    try:
        # ✅ Step 1: Generate startup details
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=(
                "Generate a startup name, tagline, and short tech stack for this idea:\n"
                f"{idea}\n\n"
                "Return strictly in this format:\n"
                "Name: <name>\n"
                "Tagline: <tagline>\n"
                "Tech Stack: <comma-separated list>"
            )
        )

        text = (resp.text or "")
        name, tagline, stack = "N/A", "N/A", []

        for line in text.splitlines():
            low = line.lower().strip()
            if low.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif low.startswith("tagline:"):
                tagline = line.split(":", 1)[1].strip()
            elif low.startswith("tech stack:"):
                stack = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]

        if not stack:
            stack = ["Python", "Flask", "SQLite"]

        # ✅ Step 2: Auto-label the idea (semantic classification)
        if not user_label:  
            label_resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=(
                    "Analyze the following startup idea and assign a **concise category label** "
                    "(like FinTech, EdTech, AI/ML, HealthTech, E-commerce, GreenTech, Social Media, etc).\n"
                    f"Idea: {idea}\n\n"
                    "Return only one short label (1–2 words), nothing else."
                )
            )
            auto_label = (label_resp.text or "General").strip()
        else:
            auto_label = user_label  # User overrides AI label

    except Exception as e:
        name = f"{idea.split()[0].capitalize()}X" if idea.strip() else "StarterX"
        tagline = f"Revolutionizing {idea or 'your idea'} with AI"
        stack = ["Python", "Flask", "SQLite", "Tailwind CSS"]
        auto_label = user_label if user_label else "General"
        flash(f"API Error: {e}", "danger")

    return render_template(
        'result.html',
        idea=idea,
        label=auto_label,  # ✅ Now AI-generated if user didn’t provide
        startup_name=name,
        tagline=tagline,
        tech_stack=stack
    )



# ------------------------
# Copilot Routes
# ------------------------
@app.route('/copilot')
def copilot_home():
    if not require_login():
        return redirect(url_for('auth.login'))
    return render_template('copilot.html', tools=COPILOT_TOOLS)

@app.route('/tool/<key>', methods=['GET', 'POST'])
def tool(key):
    if not require_login():
        return redirect(url_for('auth.login'))

    tool_def = COPILOT_TOOLS.get(key)
    if not tool_def:
        flash("Unknown tool.", "danger")
        return redirect(url_for('copilot_home'))

    if request.method == 'GET':
        return render_template(
            'tool_form.html',
            key=key,
            title=tool_def["title"],
            cta=tool_def["cta"],
            placeholder=tool_def["placeholder"]
        )

    # POST
    user_input = request.form.get('user_input', '').strip()
    if not user_input:
        flash("Please provide some input.", "danger")
        return redirect(url_for('tool', key=key))

    try:
        md_result = call_gemini_markdown(tool_def["prompt"], user_input)
        html_result = to_html_from_markdown(md_result)
    except Exception as e:
        md_result = f"## Error\nSorry, something went wrong.\n\n**Details:** {e}"
        html_result = to_html_from_markdown(md_result)

    # Pass BOTH naming conventions so whichever template you have will work
    return render_template(
        'tool_result.html',
        # canonical names (new)
        title=tool_def["title"],
        input_text=user_input,
        markdown_result=Markup(html_result),
        # legacy names (old templates)
        query=user_input,
        result=Markup(html_result),
        # raw in case you need it
        result_raw=md_result,
        timestamp=datetime.utcnow()
    )

# ------------------------
# Save Ideas
# ------------------------

# ---------------- SAVE IDEA ----------------
from sentiment import analyze_sentiment
from labeler import assign_label   # ✅ helper for labeling ideas
import sqlite3
from flask import request, redirect, url_for, session, flash

@app.route("/save_idea", methods=["POST"])
def save_idea():
    if "user_id" not in session:
        flash("You must be logged in to save ideas.", "warning")
        return redirect(url_for("login"))

    idea = request.form.get("idea", "").strip()
    startup_name = request.form.get("startup_name", "").strip()
    tagline = request.form.get("tagline", "").strip()
    tech_stack = request.form.get("tech_stack", "").strip()

    if not idea or not startup_name:
        flash("Missing required fields.", "danger")
        return redirect(url_for("dashboard"))

    try:
        # ✅ Sentiment (Positive / Negative / Neutral)
        sentiment = analyze_sentiment(idea)

        # ✅ AI-powered Label (category like AI, FinTech, HealthTech, etc.)
        label = assign_label(idea)

        # ✅ Save to DB
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("""
            INSERT INTO saved_ideas (user_id, idea, startup_name, tagline, tech_stack, sentiment, label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session["user_id"], idea, startup_name, tagline, tech_stack, sentiment, label))
        conn.commit()
        conn.close()

        flash("Idea saved successfully!", "success")

    except Exception as e:
        flash(f"Error saving idea: {e}", "danger")

    return redirect(url_for("dashboard"))



# ---------------- VIEW IDEAS ----------------
@app.route("/saved_ideas")
def saved_ideas():
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db()
    ideas = db.execute(
        "SELECT * FROM saved_ideas WHERE user_id = ?", (session["user_id"],)
    ).fetchall()

    return render_template("saved_ideas.html", ideas=ideas)

# ---------------- EDIT IDEA ----------------
@app.route("/edit_idea/<int:idea_id>", methods=["GET", "POST"])
def edit_idea(idea_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db()

    if request.method == "POST":
        startup_name = request.form["startup_name"]
        tagline = request.form["tagline"]
        idea = request.form["idea"]
        tech_stack = request.form["tech_stack"]

        db.execute(
            "UPDATE saved_ideas SET startup_name=?, tagline=?, idea=?, tech_stack=? WHERE id=? AND user_id=?",
            (startup_name, tagline, idea, tech_stack, idea_id, session["user_id"]),
        )
        db.commit()
        return redirect(url_for("saved_ideas"))

    idea = db.execute(
        "SELECT * FROM saved_ideas WHERE id=? AND user_id=?",
        (idea_id, session["user_id"]),
    ).fetchone()

    return render_template("edit_idea.html", idea=idea)

# ---------------- DELETE IDEA ----------------
@app.route("/delete_idea/<int:idea_id>", methods=["POST"])
def delete_idea(idea_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db()
    db.execute("DELETE FROM saved_ideas WHERE id=? AND user_id=?", (idea_id, session["user_id"]))
    db.commit()
    return redirect(url_for("saved_ideas"))


# ------------------------
# Run App
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)
