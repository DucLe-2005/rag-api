from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import core.logger_utils as logger_utils
from core import lib
from core.rag.prompt_templates import SelfQueryTemplate

import os

logger = logger_utils.get_logger(__name__)

class SelfQuery:
    # Date range keywords mapping
    DATE_RANGE_KEYWORDS = {
        # Recent time periods
        "recent": {"days": 7},
        "latest": {"days": 7},
        "current": {"days": 7},
        "today": {"days": 1},
        "yesterday": {"days": 1},
        
        # Weekly periods
        "last week": {"days": 7},
        "this week": {"start": "current_week"},
        "past week": {"days": 7},
        
        # Monthly periods
        "last month": {"days": 30},
        "this month": {"start": "current_month"},
        "past month": {"days": 30},
        "month_to_date": {"start": "current_month"},
        
        # Quarterly periods
        "last quarter": {"days": 90},
        "this quarter": {"start": "current_quarter"},
        "past quarter": {"days": 90},
        "quarter_to_date": {"start": "current_quarter"},
        
        # Yearly periods
        "last year": {"days": 365},
        "this year": {"start": "current_year"},
        "past year": {"days": 365},
        "year_to_date": {"start": "current_year"},
        "ytd": {"start": "current_year"},
        
        # Custom periods
        "last 6 months": {"days": 180},
        "last 3 months": {"days": 90},
        "last 2 years": {"days": 730},
        "last 5 years": {"days": 1825}
    }

    @staticmethod
    def _extract_date_range(query: str) -> Optional[Dict[str, str]]:
        """Extract date range from query text."""
        query_lower = query.lower()
        today = datetime.now()
        
        for keyword, range_info in SelfQuery.DATE_RANGE_KEYWORDS.items():
            if keyword in query_lower:
                if "days" in range_info:
                    end_date = today.strftime("%Y-%m-%d")
                    start_date = (today - timedelta(days=range_info["days"])).strftime("%Y-%m-%d")
                    return {"start_date": start_date, "end_date": end_date}
                elif "start" in range_info:
                    if range_info["start"] == "current_year":
                        return {"start_date": f"{today.year}-01-01", "end_date": today.strftime("%Y-%m-%d")}
                    elif range_info["start"] == "current_month":
                        return {"start_date": f"{today.year}-{today.month:02d}-01", "end_date": today.strftime("%Y-%m-%d")}
                    elif range_info["start"] == "current_quarter":
                        quarter_start = ((today.month - 1) // 3) * 3 + 1
                        return {"start_date": f"{today.year}-{quarter_start:02d}-01", "end_date": today.strftime("%Y-%m-%d")}
                    elif range_info["start"] == "current_week":
                        # Get the start of the current week (Monday)
                        start_of_week = today - timedelta(days=today.weekday())
                        return {"start_date": start_of_week.strftime("%Y-%m-%d"), "end_date": today.strftime("%Y-%m-%d")}
        
        return None

    @staticmethod
    def extract_metadata(query: str) -> Dict[str, Any]:
        """Extract date range from query text and append to query if found."""
        try:
            date_range = SelfQuery._extract_date_range(query)
            
            if date_range:
                date_context = f" between {date_range['start_date']} and {date_range['end_date']}"
                query = query + date_context
                logger.info(
                    "Added date range to query",
                    query=query,
                    date_range=date_range
                )
            
            return {
                "modified_query": query,
                "date_range": date_range
            }
        except Exception as e:
            logger.error(f"Error extracting date range: {str(e)}")
            return {
                "modified_query": query,
                "date_range": None
            }
                