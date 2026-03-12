from openai import OpenAI
from mightydatainc_llm_conversation import llm_submit

client = OpenAI()

story = llm_submit(
    messages=[
        {
            "role": "user",
            "content": "Write a short story about a raccoon who steals the Mona Lisa.",
        }
    ],
    ai_client=client,
)

print(story)
