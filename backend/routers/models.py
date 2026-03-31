"""
backend/routers/models.py
Model health check — admin only.
"""

from fastapi import APIRouter, Depends, HTTPException
from dependencies import get_current_user
from model_manager import get_llm, MODELS

router = APIRouter()
ADMIN_USER_ID = "212d73b9-2206-40dc-be20-78df844c500c"


@router.get("/health")
async def model_health(user_id: str = Depends(get_current_user)):
    import time
    from langchain_core.messages import HumanMessage

    if user_id != ADMIN_USER_ID:
        raise HTTPException(status_code=403, detail="Admin only")

    results = []
    for m in MODELS:
        result = {"model": m["label"], "provider": m["provider"], "status": "", "latency": "—"}
        try:
            llm = get_llm(m["label"])
            start = time.time()
            llm.invoke([HumanMessage(content="Reply: working")])
            result["status"] = "✅ working"
            result["latency"] = f"{round(time.time()-start, 1)}s"
        except Exception as e:
            err = str(e).lower()
            result["status"] = (
                "❌ auth error" if "403" in str(e) or "401" in str(e)
                else "❌ not found" if "404" in str(e)
                else "⚠️ quota" if "quota" in err
                else "🔌 offline" if "connect" in err
                else "❌ error"
            )
        results.append(result)
    return {"results": results}