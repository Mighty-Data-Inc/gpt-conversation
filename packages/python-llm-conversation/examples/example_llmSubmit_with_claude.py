import anthropic
from mightydatainc_llm_conversation import llm_submit

client = anthropic.Anthropic()

reply = llm_submit(
    messages=[
        {
            "role": "user",
            "content": "Summarize the causes of the French Revolution in one concise paragraph.",
        }
    ],
    ai_client=client,
)

print(reply)
