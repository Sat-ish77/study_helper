"""
pages/5_🃏_Flashcards.py — Study Helper v2
Flashcard system with SM-2 spaced repetition algorithm.
Three modes: Review, Browse, Generate.
"""
import streamlit as st
import time
from datetime import datetime, timezone

from auth import require_auth
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector, get_llm
from flashcard_db import (
    get_due_flashcards,
    update_flashcard_review,
    get_all_flashcards,
    delete_flashcard,
    create_flashcard,
    get_flashcard_stats,
    generate_flashcards_from_content,
    save_flashcard_from_qa,
    QUALITY_RATINGS
)

# Page config
st.set_page_config(
    page_title="Flashcards — Study Helper",
    page_icon="🃏",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Theme + auth
inject_css()
user_id = require_auth()
if not user_id:
    st.stop()

# Sidebar
sidebar_header()
model_label = render_model_selector()
llm = get_llm(model_label)

# Main content
st.markdown(
    '<h2 style="font-family:\'DM Serif Display\',serif; color:#f0ede8;">'
    '🃏 Flashcards</h2>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#6b7280; margin-bottom:1.5rem;">'
    'Spaced repetition flashcards with SM-2 algorithm for optimal learning.</p>',
    unsafe_allow_html=True
)

# Stats widget
stats = get_flashcard_stats(user_id)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total", stats["total"])
with col2:
    st.metric("Due", stats["due"])
with col3:
    st.metric("Learned", stats["learned"])
with col4:
    st.metric("Retention", f"{stats['retention']:.1f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["📖 Review", "📚 Browse", "🤖 Generate"])

with tab1:
    st.markdown("### Review Due Cards")
    
    # Get due cards
    due_cards = get_due_flashcards(user_id, limit=10)
    
    if not due_cards:
        st.info("No cards due for review! 🎉")
        st.caption("Cards will appear here when they're due for spaced repetition.")
    else:
        # Progress bar
        if 'review_progress' not in st.session_state:
            st.session_state.review_progress = 0
        if 'current_card_index' not in st.session_state:
            st.session_state.current_card_index = 0
        
        # Get current card
        if st.session_state.current_card_index >= len(due_cards):
            st.session_state.current_card_index = 0
        
        card = due_cards[st.session_state.current_card_index]
        
        # Progress
        progress = (st.session_state.current_card_index + 1) / len(due_cards)
        st.progress(progress, text=f"Card {st.session_state.current_card_index + 1} of {len(due_cards)}")
        
        # Card display
        with st.container():
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            
            # Front (question)
            if 'show_answer' not in st.session_state:
                st.session_state.show_answer = False
            
            if not st.session_state.show_answer:
                st.markdown("### Question")
                st.markdown(f"**{card['question']}**")
                
                if st.button("Show Answer", type="primary", width='stretch'):
                    st.session_state.show_answer = True
                    st.rerun()
            else:
                # Back (answer)
                st.markdown("### Answer")
                st.markdown(card['answer'])
                
                if card.get('source_file'):
                    st.caption(f"📍 {card['source_file']}")
                
                # Quality rating buttons
                st.markdown("#### How well did you know this?")
                
                cols = st.columns(5)
                for i, (rating, description) in enumerate(QUALITY_RATINGS.items()):
                    with cols[i % len(cols)]:
                        if st.button(str(rating), 
                                   help=description,
                                   width='stretch',
                                   key=f"rating_{rating}"):
                            # Record review
                            update_flashcard_review(card['id'], rating)
                            
                            # Move to next card
                            st.session_state.current_card_index += 1
                            st.session_state.show_answer = False
                            
                            # Reset if done
                            if st.session_state.current_card_index >= len(due_cards):
                                st.session_state.current_card_index = 0
                                st.success("✅ Review session complete!")
                                st.rerun()
                            else:
                                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Browse All Cards")
    
    # Search/filter
    search = st.text_input("Search cards...", placeholder="Search in front/back text...")
    
    all_cards = get_all_flashcards(user_id)
    
    # Filter by search
    if search:
        search_lower = search.lower()
        all_cards = [
            card for card in all_cards
            if search_lower in card['question'].lower() or search_lower in card['answer'].lower()
        ]
    
    if not all_cards:
        st.info("No flashcards yet. Create some in the Generate tab!")
    else:
        for i, card in enumerate(all_cards):
            with st.expander(f"📝 {card['question'][:50]}...", expanded=False):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**Question:** {card['question']}")
                    st.markdown(f"**Answer:** {card['answer']}")
                    
                    if card.get('source_file'):
                        st.caption(f"📍 {card['source_file']}")
                    
                    # SM-2 stats
                    st.caption(
                        f"📊 Repetitions: {card['repetitions']} | "
                        f"Interval: {card['interval_days']} days | "
                        f"Ease: {card['ease_factor']:.2f}"
                    )
                    
                    # Next review
                    if card['next_review']:
                        next_review = datetime.fromisoformat(card['next_review'])
                        days_until = (next_review.date() - datetime.now(timezone.utc).date()).days
                        if days_until <= 0:
                            st.caption("🔴 Due now!")
                        elif days_until == 1:
                            st.caption("🟡 Due tomorrow")
                        else:
                            st.caption(f"🟢 Due in {days_until} days")
                
                with col2:
                    if st.button("🗑️", key=f"delete_{card['id']}", help="Delete card"):
                        delete_flashcard(card['id'], user_id)
                        st.rerun()

with tab3:
    st.markdown("### Generate Flashcards")
    
    gen_method = st.radio("Choose method:", ["From Q&A", "From Document", "Manual"])
    
    if gen_method == "From Q&A":
        st.markdown("#### Save Last Q&A as Flashcard")
        
        # Get last Q&A from session
        messages = st.session_state.get("messages", [])
        last_qa = None
        
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("original_question"):
                last_qa = {
                    "question": msg.get("original_question", ""),
                    "answer": msg["content"]
                }
                break
        
        if last_qa:
            st.markdown(f"**Question:** {last_qa['question']}")
            st.markdown(f"**Answer:** {last_qa['answer'][:200]}...")
            
            if st.button("💾 Save as Flashcard", type="primary"):
                card_id = save_flashcard_from_qa(
                    user_id,
                    last_qa['question'],
                    last_qa['answer']
                )
                if card_id:
                    st.success("✅ Flashcard saved!")
                    st.rerun()
                else:
                    st.error("Failed to save flashcard")
        else:
            st.info("No Q&A found. Ask a question in Study Helper first!")
    
    elif gen_method == "From Document":
        st.markdown("#### Generate from Study Material")
        
        # Get user documents
        from ingest import get_user_documents
        docs = get_user_documents(user_id)
        
        if not docs:
            st.info("No documents uploaded yet. Upload documents in the Upload Docs page.")
        else:
            selected_doc = st.selectbox("Select document:", docs)
            num_cards = st.slider("Number of cards:", 1, 10, 5)
            
            if st.button("🎲 Generate Flashcards", type="primary"):
                with st.spinner("Generating flashcards..."):
                    # Fetch document chunks from Supabase
                    from main import retrieve_docs
                    from flashcard_db import generate_flashcards_from_content, create_flashcard
                    
                    # Get document content (retrieve chunks for this document)
                    query = f"content from {selected_doc}"
                    rr = retrieve_docs(user_id, query)
                    
                    # Filter chunks to only include those from the selected document
                    filtered_chunks = []
                    for doc in rr.docs:
                        if doc.metadata.get("filename") == selected_doc:
                            filtered_chunks.append(doc.page_content)
                    
                    if not filtered_chunks:
                        st.error(f"No content found for {selected_doc}")
                    else:
                        # Combine chunks (up to 3000 chars total)
                        combined_content = ""
                        for chunk in filtered_chunks:
                            if len(combined_content) + len(chunk) <= 3000:
                                combined_content += "\n\n" + chunk
                            else:
                                break
                        
                        # Generate flashcards using LLM
                        from model_manager import get_llm, render_model_selector
                        llm = get_llm(st.session_state.get("model_label", "gpt-3.5-turbo"))
                        
                        generated_cards = generate_flashcards_from_content(
                            user_id=user_id,
                            content=combined_content,
                            num_cards=num_cards,
                            llm=llm
                        )
                        
                        if generated_cards:
                            # Store in session state for preview
                            st.session_state.generated_cards = generated_cards
                            st.session_state.selected_doc = selected_doc
                            st.success(f"Generated {len(generated_cards)} flashcards from {selected_doc}")
                        else:
                            st.error("Failed to generate flashcards. Try switching to a different AI model in the sidebar (Groq or Gemini work well).")
            
            # Show preview if cards were generated
            if "generated_cards" in st.session_state and st.session_state.generated_cards:
                st.markdown("#### Preview Generated Flashcards")
                
                preview_cards = st.session_state.generated_cards[:3]  # Show first 3
                
                for i, card in enumerate(preview_cards):
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05); border:1px solid #4b5563; '
                        f'border-radius:8px; padding:16px; margin:8px 0;">'
                        f'<div style="color:#9ca3af; font-size:0.9rem; margin-bottom:8px;">Card {i+1}</div>'
                        f'<div style="margin-bottom:8px;"><strong>Q:</strong> {card["question"]}</div>'
                        f'<div><strong>A:</strong> {card["answer"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                if len(st.session_state.generated_cards) > 3:
                    st.caption(f"... and {len(st.session_state.generated_cards) - 3} more cards")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save All Cards", type="primary"):
                        saved_count = 0
                        for card in st.session_state.generated_cards:
                            card_id = create_flashcard(
                                user_id=user_id,
                                question=card["question"],
                                answer=card["answer"],
                                source="document",
                                source_file=st.session_state.selected_doc
                            )
                            if card_id:
                                saved_count += 1
                        
                        if saved_count > 0:
                            st.success(f"✅ Saved {saved_count} flashcards!")
                            # Clear generated cards from session
                            del st.session_state.generated_cards
                            del st.session_state.selected_doc
                            st.rerun()
                        else:
                            st.error("Failed to save flashcards")
                
                with col2:
                    if st.button("❌ Cancel"):
                        del st.session_state.generated_cards
                        del st.session_state.selected_doc
                        st.rerun()
    
    else:  # Manual
        st.markdown("#### Create Manual Flashcard")
        
        with st.form("manual_flashcard"):
            question = st.text_area("Question", height=100, key="manual_question")
            answer = st.text_area("Answer", height=100, key="manual_answer")
            source_file = st.text_input("Source (optional)", key="manual_source_file")
            
            submitted = st.form_submit_button("💾 Create Flashcard", type="primary")
            
            if submitted:
                if question.strip() and answer.strip():
                    card_id = create_flashcard(
                        user_id=user_id,
                        question=question.strip(),
                        answer=answer.strip(),
                        source="manual",
                        source_file=source_file.strip()
                    )
                    if card_id:
                        st.success("✅ Flashcard created!")
                        st.rerun()
                    else:
                        st.error("Failed to create flashcard")
                else:
                    st.error("Please fill in both question and answer")
