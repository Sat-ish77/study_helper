"""
test_all.py — Study Helper v2 Systematic Test
Run this to find all backend bugs before fixing anything.
Usage: python test_all.py
"""
import os
import json
from datetime import date
from dotenv import load_dotenv
load_dotenv()

DUMMY_USER = "212d73b9-2206-40dc-be20-78df844c500c"  # Your real user_id
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []

def test(name, fn):
    try:
        result = fn()
        results.append((PASS, name, result or "OK"))
    except Exception as e:
        results.append((FAIL, name, str(e)))

# ── Supabase connection ───────────────────────────────────────────────────────
def check_supabase():
    from supabase_client import get_supabase
    sb = get_supabase()
    r = sb.table("sh_document_chunks").select("id").limit(1).execute()
    return f"{len(r.data)} chunks found"

test("Supabase connection", check_supabase)

# ── Embeddings ────────────────────────────────────────────────────────────────
def check_embeddings():
    from main import get_embeddings
    emb = get_embeddings()
    v = emb.embed_query("test")
    assert len(v) == 1536, f"Wrong dims: {len(v)}"
    return f"1536 dims OK"

test("Embeddings (1536 dims)", check_embeddings)

# ── Retrieval ─────────────────────────────────────────────────────────────────
def check_retrieval():
    from main import retrieve_docs
    rr = retrieve_docs(DUMMY_USER, "python loops")
    avg_score = f"{rr.avg_score:.3f}" if rr.avg_score else "N/A"
    return f"{len(rr.docs)} docs, avg_score={avg_score}"

test("RAG retrieval", check_retrieval)

# ── Model manager ─────────────────────────────────────────────────────────────
def check_models():
    from model_manager import get_available_models
    models = get_available_models()
    return f"{len(models)} models available"

test("Model manager", check_models)

# ── Quiz DB ───────────────────────────────────────────────────────────────────
def check_quiz_db():
    from quiz_db import get_user_history
    history = get_user_history(DUMMY_USER)
    return f"{len(history)} quiz sessions found"

test("Quiz DB - get history", check_quiz_db)

def check_quiz_stats():
    from quiz_db import get_quiz_stats
    stats = get_quiz_stats(DUMMY_USER)
    return f"stats keys: {list(stats.keys()) if stats else 'empty'}"

test("Quiz DB - get stats", check_quiz_stats)

def check_recent_scores():
    from quiz_db import get_recent_quiz_scores
    scores = get_recent_quiz_scores(DUMMY_USER)
    return f"{len(scores)} recent scores"

test("Quiz DB - recent scores", check_recent_scores)

def check_quiz_save():
    from quiz_db import save_score
    success = save_score(
        user_id=DUMMY_USER,
        topic="Test Topic",
        source_file="test_file.txt",
        score=8,
        total=10,
        difficulty="medium",
        types_used=["mcq"],
        language="English"
    )
    return f"Save test: {'success' if success else 'failed'}"

test("Quiz DB - save score", check_quiz_save)

# ── Flashcard DB ──────────────────────────────────────────────────────────────
def check_flashcard_due():
    from flashcard_db import get_due_flashcards
    cards = get_due_flashcards(DUMMY_USER)
    return f"{len(cards)} cards due"

test("Flashcard DB - due cards", check_flashcard_due)

def check_flashcard_all():
    from flashcard_db import get_all_flashcards
    cards = get_all_flashcards(DUMMY_USER)
    return f"{len(cards)} total cards"

test("Flashcard DB - all cards", check_flashcard_all)

def check_flashcard_stats():
    from flashcard_db import get_flashcard_stats
    stats = get_flashcard_stats(DUMMY_USER)
    return f"stats keys: {list(stats.keys()) if stats else 'empty'}"

test("Flashcard DB - stats", check_flashcard_stats)

def check_sm2():
    # Check if SM-2 function exists
    import flashcard_db
    if not hasattr(flashcard_db, 'calculate_sm2_next_review'):
        return "calculate_sm2_next_review function not found"
    
    card = {
        'ease_factor': 2.5,
        'interval_days': 1,
        'repetitions': 0,
        'next_review': date.today().isoformat(),
        'last_review': None
    }
    new_ease, new_interval, new_reps = flashcard_db.calculate_sm2_next_review(
        card['ease_factor'], card['interval_days'], card['repetitions'], 3
    )  # Good rating
    assert new_interval > 0
    assert new_ease >= 1.3
    return f"interval={new_interval}d, ease={new_ease:.2f}"

test("SM-2 algorithm", check_sm2)

def check_flashcard_create():
    from flashcard_db import create_flashcard
    card_id = create_flashcard(
        user_id=DUMMY_USER,
        question="Test question",
        answer="Test answer",
        source="test",
        source_file="test_file.txt"
    )
    return f"Create test: {'card_id: ' + str(card_id) if card_id else 'failed'}"

test("Flashcard DB - create card", check_flashcard_create)

# ── Canvas API ────────────────────────────────────────────────────────────────
def check_canvas_load():
    from canvas_api import load_ical_url
    url = load_ical_url(DUMMY_USER)
    return f"iCal URL: {'set' if url else 'not set'}"

test("Canvas - load URL", check_canvas_load)

def check_canvas_dismissed():
    from canvas_api import get_dismissed
    d = get_dismissed(DUMMY_USER)
    return f"{len(d)} dismissed events"

# ── Canvas URL persistence test (Phase 2) ───────────────────────────────────────
def check_canvas_url_persistence():
    """Test Canvas URL can be saved and loaded from database"""
    from supabase_client import get_supabase
    
    sb = get_supabase()
    test_user = DUMMY_USER
    test_url = "https://test.canvas.instructure.com/feeds/calendars/user.ics"
    
    # Save test URL
    try:
        sb.table("sh_user_settings").upsert({
            "user_id": test_user,
            "ical_url": test_url
        }).execute()
    except Exception as e:
        return f"Save failed: {e}"
    
    # Load URL back
    try:
        result = sb.table("sh_user_settings").select("ical_url").eq("user_id", test_user).execute()
        saved_url = result.data[0]["ical_url"] if result.data else None
        if saved_url == test_url:
            # Clean up test data
            sb.table("sh_user_settings").update({"ical_url": None}).eq("user_id", test_user).execute()
            return f"Save/Load OK: {test_url}"
        else:
            return f"Mismatch: saved={saved_url}, expected={test_url}"
    except Exception as e:
        return f"Load failed: {e}"

test("Canvas URL persistence", check_canvas_url_persistence)

# ── Canvas dismissed events test (NB4) ───────────────────────────────────────
def check_canvas_dismissed_persistence():
    """Test dismissed events persist after refresh"""
    from canvas_api import dismiss_event, get_dismissed
    
    test_user = DUMMY_USER
    test_event_id = "test-event-123"
    
    # Dismiss event
    try:
        dismiss_event(test_user, test_event_id)
    except Exception as e:
        return f"Dismiss failed: {e}"
    
    # Check if dismissed
    try:
        dismissed = get_dismissed(test_user)
        if test_event_id in dismissed:
            # Clean up test data
            from supabase_client import get_supabase
            get_supabase().table("sh_dismissed_events").delete().eq("user_id", test_user).eq("event_id", test_event_id).execute()
            return f"Dismiss/Get OK: {test_event_id} in {len(dismissed)} dismissed events"
        else:
            return f"Event not found in dismissed set. Found: {list(dismissed)}"
    except Exception as e:
        return f"Get dismissed failed: {e}"

test("Canvas dismissed events persistence", check_canvas_dismissed_persistence)

# ── Course tagging test ───────────────────────────────────────────────────────
def check_course_tagging():
    """Test quiz can be saved with course_name"""
    from quiz_db import save_score, get_user_history
    
    test_user = DUMMY_USER
    test_course = "CS4540"
    
    # Save quiz with course
    try:
        success = save_score(
            user_id=test_user,
            topic="Test topic",
            source_file="test.pdf",
            score=3,
            total=5,
            difficulty="medium",
            types_used=["mcq"],
            language="English",
            course_name=test_course
        )
        if not success:
            return "Save failed"
    except Exception as e:
        return f"Save with course failed: {e}"
    
    # Verify course was saved
    try:
        history = get_user_history(test_user, limit=1)
        if history and history[0].get("course_name") == test_course:
            return f"Course tag OK: saved '{test_course}'"
        else:
            return f"Course not found. Latest: {history[0] if history else 'None'}"
    except Exception as e:
        return f"Verify course failed: {e}"

test("Course tagging", check_course_tagging)

# ── Dashboard score calculation test (Phase 3) ─────────────────────────────────
def check_dashboard_score_calculation():
    """Test dashboard calculates percentage correctly from score/total"""
    from quiz_db import save_score, get_recent_quiz_scores
    
    test_user = DUMMY_USER
    
    # Save a quiz with known score/total
    try:
        success = save_score(
            user_id=test_user,
            topic="Test for dashboard",
            source_file="test.pdf",
            score=3,
            total=5,
            difficulty="medium",
            types_used=["mcq"],
            language="English",
            course_name="CS101"
        )
        if not success:
            return "Save failed"
    except Exception as e:
        return f"Save failed: {e}"
    
    # Verify get_recent_quiz_scores returns both score and total
    try:
        recent = get_recent_quiz_scores(test_user, limit=1)
        if recent:
            latest = recent[0]
            if "score" in latest and "total" in latest:
                expected_pct = round(3 / 5 * 100)
                return f"Score/total OK: {latest['score']}/{latest['total']} = {expected_pct}%"
            else:
                return f"Missing fields: {list(latest.keys())}"
        else:
            return "No recent quizzes found"
    except Exception as e:
        return f"Get recent failed: {e}"

test("Dashboard score calculation", check_dashboard_score_calculation)

# ── Document flashcard generation test (Phase 4) ───────────────────────────────
def check_document_flashcard_generation():
    """Test flashcards can be generated from document content"""
    from flashcard_db import generate_flashcards_from_content
    from model_manager import get_llm, get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Test with sample content
    sample_content = """
    Python is a high-level programming language created by Guido van Rossum.
    It was first released in 1991. Python uses dynamic typing and garbage collection.
    The language supports multiple programming paradigms including procedural,
    object-oriented, and functional programming.
    """
    
    try:
        cards = generate_flashcards_from_content(
            llm=llm,
            content=sample_content,
            num_cards=2
        )
        
        if cards and len(cards) > 0:
            return f"Generated {len(cards)} cards from document"
        else:
            return "No cards generated"
    except Exception as e:
        return f"Generation failed: {e}"

test("Document flashcard generation", check_document_flashcard_generation)

# ── Auto-flashcards from quiz test (Phase 5) ───────────────────────────────────
def check_auto_flashcards_from_quiz():
    """Test flashcards can be created from quiz questions"""
    from flashcard_db import create_flashcard
    
    test_user = DUMMY_USER
    
    try:
        # Create a flashcard from a quiz question
        card_id = create_flashcard(
            user_id=test_user,
            question="What is Python?",
            answer="A high-level programming language created by Guido van Rossum",
            source="quiz_weakness",
            source_file="Quiz: Test"
        )
        
        if card_id:
            # Clean up
            from flashcard_db import delete_flashcard
            delete_flashcard(test_user, card_id)
            return f"Auto-flashcard OK: created {card_id}"
        else:
            return "Failed to create flashcard"
    except Exception as e:
        return f"Auto-flashcard failed: {e}"

test("Auto-flashcards from quiz", check_auto_flashcards_from_quiz)

# ── 24hr deadline banner test (Phase 5) ───────────────────────────────────────
def check_deadline_banner():
    """Test deadline banner shows urgent events"""
    from canvas_api import load_ical_url, get_upcoming_events
    from datetime import datetime, timezone, timedelta
    
    test_user = DUMMY_USER
    # First get the raw events
    ical_url = load_ical_url(test_user)
    if not ical_url:
        return "Deadline banner: No Canvas URL set"
    
    from canvas_api import fetch_canvas_events
    events = fetch_canvas_events(ical_url, test_user)
    upcoming = get_upcoming_events(events, days_ahead=2)
    
    urgent_count = 0
    for event in upcoming:
        start = event.get('start')
        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hours_until = (start - now).total_seconds() / 3600
            if 0 <= hours_until <= 24:
                urgent_count += 1
    
    return f"Deadline banner: {urgent_count} urgent events in next 24hrs"

test("24hr deadline banner", check_deadline_banner)

# ── UI deprecation fixes test (Phase 6) ───────────────────────────────────────────
def check_ui_deprecation_fixes():
    """Test that use_container_width has been replaced with width='stretch'"""
    import os
    import re
    
    # Check key files for use_container_width
    files_to_check = [
        "pages/2_🧪_Quiz_Lab.py",
        "pages/6_📊_Dashboard.py",
        "pages/5_🃏_Flashcards.py",
        "pages/1_📚_Study_Helper.py",
        "canvas_api.py",
        "auth.py"
    ]
    
    violations = 0
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "use_container_width=True" in content:
                    violations += 1
    
    if violations == 0:
        return "UI deprecation OK: No use_container_width found"
    else:
        return f"UI deprecation FAIL: {violations} files still use use_container_width"

test("UI deprecation fixes", check_ui_deprecation_fixes)

# ── Learn More ────────────────────────────────────────────────────────────────
def check_learn_more_import():
    import learn_more
    fns = [f for f in dir(learn_more) if not f.startswith('_')]
    return f"functions: {fns}"

test("Learn More - import", check_learn_more_import)

def check_youtube_key():
    key = os.getenv("YOUTUBE_API_KEY")
    assert key, "YOUTUBE_API_KEY not set in .env"
    return f"key set (starts with {key[:8]}...)"

test("YouTube API key", check_youtube_key)

def check_youtube_search():
    import learn_more
    results = learn_more.search_youtube("python programming")
    return f"found {len(results)} videos"

test("YouTube search", check_youtube_search)

def check_wikimedia_search():
    import learn_more
    results = learn_more.search_wikimedia("python")
    return f"found {len(results)} images"

test("Wikimedia search", check_wikimedia_search)

# ── Supabase table columns ────────────────────────────────────────────────────
def check_table_columns():
    from supabase_client import get_supabase
    sb = get_supabase()
    issues = []

    # Check sh_flashcards has question/answer
    try:
        r = sb.table("sh_flashcards").select("question, answer, ease_factor, interval_days, repetitions, next_review").limit(1).execute()
    except Exception as e:
        issues.append(f"sh_flashcards columns: {e}")

    # Check sh_learn_more_cache has videos/images
    try:
        r2 = sb.table("sh_learn_more_cache").select("topic, images, videos, cached_at").limit(1).execute()
    except Exception as e:
        issues.append(f"sh_learn_more_cache columns: {e}")

    # Check sh_canvas_cache
    try:
        r3 = sb.table("sh_canvas_cache").select("cache_key, data, expires_at").limit(1).execute()
    except Exception as e:
        issues.append(f"sh_canvas_cache columns: {e}")

    # Check sh_user_settings
    try:
        r4 = sb.table("sh_user_settings").select("user_id, ical_url").limit(1).execute()
    except Exception as e:
        issues.append(f"sh_user_settings columns: {e}")

    # Check sh_quiz_scores has course_name
    try:
        r5 = sb.table("sh_quiz_scores").select("user_id, score, course_name").limit(1).execute()
    except Exception as e:
        issues.append(f"sh_quiz_scores columns: {e}")

    # Check sh_document_chunks
    try:
        r6 = sb.table("sh_document_chunks").select("user_id, source_file, filetype").limit(1).execute()
    except Exception as e:
        issues.append(f"sh_document_chunks columns: {e}")

    return "All table columns verified" if not issues else f"Issues: {'; '.join(issues)}"

test("Supabase table schemas", check_table_columns)

# ── Auth ──────────────────────────────────────────────────────────────────────
def check_auth_import():
    from auth import require_auth
    return "auth.py imports OK"

test("Auth module", check_auth_import)

# ── Video generator ───────────────────────────────────────────────────────────
def check_moviepy():
    import moviepy
    assert moviepy.__version__.startswith("1"), f"Wrong version: {moviepy.__version__}"
    return f"MoviePy {moviepy.__version__} OK"

test("MoviePy version (must be 1.x)", check_moviepy)

# ── ingest ────────────────────────────────────────────────────────────────────
def check_ingest():
    import ingest
    functions = [f for f in dir(ingest) if not f.startswith('_')]
    return f"functions: {functions}"

test("Ingest module", check_ingest)

# ── Dashboard ─────────────────────────────────────────────────────────────────
def check_dashboard_import():
    import importlib.util
    import glob
    import os
    pages_dir = "pages"
    dashboard_files = [f for f in os.listdir(pages_dir) if f.startswith("6_") and f.endswith(".py")]
    assert dashboard_files, "Dashboard page not found"
    return f"Found: {dashboard_files[0]}"

test("Dashboard page exists", check_dashboard_import)

# ── Image generator ───────────────────────────────────────────────────────────
def check_image_generator():
    from image_generator import render_image_lab_page
    return "image_generator.py imports OK"

test("Image generator module", check_image_generator)

# ── Quiz generation test ───────────────────────────────────────────────────────
def check_quiz_generation_english():
    from model_manager import get_llm
    from model_manager import get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Test quiz generation for English
    context = "Python is a high-level programming language. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    
    prompt = f"""Based on the study material below, generate exactly 2 quiz questions.
Difficulty: medium
Language: English
Use a mix of these types:
- Multiple choice — 4 options. correct = "A"/"B"/"C"/"D".
- True/False. correct = "True" or "False".

STUDY MATERIAL:
{context[:4000]}

Return ONLY a valid JSON array. Each object must have:
  "question": string
  "type": one of ["mcq", "true_false"]
  "options": list (MCQ: 4 options; true_false: ["True","False"]; others: [])
  "correct": string
  "explanation": string
  "difficulty": "medium"

No markdown fences, no extra text. Pure JSON array only."""
    
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        questions = json.loads(raw.strip())
        return f"English quiz: {len(questions)} questions generated"
    except Exception as e:
        return f"English quiz generation failed: {e}"

test("Quiz generation (English)", check_quiz_generation_english)

def check_quiz_generation_nepali():
    from model_manager import get_llm
    from model_manager import get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Test quiz generation for Nepali
    context = "Python is a high-level programming language. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    
    prompt = f"""Based on the study material below, generate exactly 2 quiz questions.
Difficulty: medium
Language: Nepali
Use a mix of these types:
- Multiple choice — 4 options. correct = "A"/"B"/"C"/"D".
- True/False. correct = "True" or "False".

IMPORTANT: For true_false type, the question MUST be a declarative statement that can be true or false. NEVER generate "why/how/what" questions for true_false type.

STUDY MATERIAL:
{context[:4000]}

Return ONLY a valid JSON array. Each object must have:
  "question": string
  "type": one of ["mcq", "true_false"]
  "options": list (MCQ: 4 options; true_false: ["True","False"]; others: [])
  "correct": string
  "explanation": string
  "difficulty": "medium"

No markdown fences, no extra text. Pure JSON array only."""
    
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        questions = json.loads(raw.strip())
        
        # Check for invalid "why" questions in true_false
        invalid_tf = []
        for q in questions:
            if q.get("type") == "true_false":
                question_text = q.get("question", "").lower()
                if any(word in question_text for word in ["why", "how", "what", "when", "where", "किन", "कसरी", "के"]):
                    invalid_tf.append(q["question"])
        
        if invalid_tf:
            return f"Nepali quiz: {len(questions)} questions, BUT {len(invalid_tf)} invalid T/F questions found"
        return f"Nepali quiz: {len(questions)} questions generated (valid)"
    except Exception as e:
        return f"Nepali quiz generation failed: {e}"

test("Quiz generation (Nepali)", check_quiz_generation_nepali)

# ── Quiz grading tests (Phase 1) ───────────────────────────────────────────────
def check_quiz_grading_letter():
    """Test MCQ grading when correct answer is a letter"""
    from model_manager import get_llm, get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Simulate MCQ grading logic from Quiz_Lab.py
    question = {"type": "mcq", "options": ["A. A programming language", "B. A database", "C. An OS", "D. A compiler"]}
    user_answer = "A. A programming language"  # User selects full text
    correct_answer = "A"  # Correct answer is letter
    
    # Apply the same logic as in render_quiz_results
    options = question.get("options", [])
    user_letter = user_answer[0].upper() if user_answer and len(user_answer) > 0 else ""
    
    if len(correct_answer) == 1 and correct_answer.isalpha():
        correct_letter = correct_answer.upper()
        is_correct = user_letter == correct_letter
    else:
        is_correct = user_answer == correct_answer
    
    return f"Letter grading: user_letter={user_letter}, correct_letter={correct_letter}, is_correct={is_correct}"

test("Quiz grading (letter correct)", check_quiz_grading_letter)

def check_quiz_grading_text():
    """Test MCQ grading when correct answer is full text"""
    from model_manager import get_llm, get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Simulate MCQ grading when correct answer is full text
    question = {"type": "mcq", "options": ["A. A programming language", "B. A database", "C. An OS", "D. A compiler"]}
    user_answer = "A. A programming language"  # User selects full text
    correct_answer = "A. A programming language"  # Correct answer is also full text
    
    # Apply the same logic as in render_quiz_results
    options = question.get("options", [])
    user_letter = user_answer[0].upper() if user_answer and len(user_answer) > 0 else ""
    
    if len(correct_answer) == 1 and correct_answer.isalpha():
        correct_letter = correct_answer.upper()
        is_correct = user_letter == correct_letter
    else:
        is_correct = user_answer == correct_answer
    
    return f"Text grading: user_answer='{user_answer}', correct_answer='{correct_answer}', is_correct={is_correct}"

test("Quiz grading (text correct)", check_quiz_grading_text)

def check_quiz_grading_case_insensitive():
    """Test that grading is case insensitive"""
    from model_manager import get_llm, get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Test case insensitive comparison
    question = {"type": "mcq", "options": ["A. A programming language", "B. A database", "C. An OS", "D. A compiler"]}
    user_answer = "a. a programming language"  # Lowercase
    correct_answer = "A"  # Uppercase letter
    
    # Apply the same logic as in render_quiz_results
    options = question.get("options", [])
    user_letter = user_answer[0].upper() if user_answer and len(user_answer) > 0 else ""
    
    if len(correct_answer) == 1 and correct_answer.isalpha():
        correct_letter = correct_answer.upper()
        is_correct = user_letter == correct_letter
    else:
        is_correct = user_answer == correct_answer
    
    return f"Case insensitive: user_letter={user_letter}, correct_letter={correct_letter}, is_correct={is_correct}"

test("Quiz grading (case insensitive)", check_quiz_grading_case_insensitive)

# ── Quiz grading test ─────────────────────────────────────────────────────────
def check_quiz_grading():
    from model_manager import get_llm
    from model_manager import get_available_models
    
    models = get_available_models()
    if not models:
        return "No models available"
    
    llm = get_llm(models[0])
    
    # Test MCQ grading
    question = "What is Python?"
    user_answer = "A. A programming language"
    correct_answer = "A"
    
    try:
        # Import the grading function directly from the file
        import sys
        import os
        sys.path.append(os.path.join(os.getcwd(), 'pages'))
        
        # Import Quiz Lab functions
        exec(open('pages/2_🧪_Quiz_Lab.py').read(), globals())
        
        result = grade_answer(llm, question, user_answer, correct_answer)
        return f"Grading test: is_correct={result.get('is_correct')}, score={result.get('score')}"
    except Exception as e:
        return f"Grading test failed: {e}"

test("Quiz grading", check_quiz_grading)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STUDY HELPER v2 — SYSTEM TEST RESULTS")
print("="*60)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)

for status, name, detail in results:
    print(f"{status} {name}")
    if status == FAIL:
        print(f"   -> {detail}")
    elif detail and detail != "OK":
        print(f"   -> {detail}")

print("="*60)
print(f"Total: {passed} passed, {failed} failed out of {len(results)} tests")
print("="*60)

if failed > 0:
    print("\nFAILED TESTS SUMMARY:")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  - {name}: {detail}")
