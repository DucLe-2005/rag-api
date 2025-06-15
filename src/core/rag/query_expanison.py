from langchain_openai import ChatOpenAI

from core.rag.prompt_templates import QueryExpansionTemplate

import os
from dotenv import load_dotenv

load_dotenv()

class QueryExpansion:
    @staticmethod
    def generate_response(query: str, to_expand_to_n: int) -> list[str]:
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(to_expand_to_n)
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_ID"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
        chain = prompt | model

        response = chain.invoke({"question": query})
        response = response.content

        queries = response.strip().split(query_expansion_template.separator)
        stripped_queries = [
            stripped_item for item in queries if (stripped_item := item.strip(" \\n"))
        ]

        return stripped_queries