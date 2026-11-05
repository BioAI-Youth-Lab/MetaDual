"""DeepSeek credentials are supplied only through the environment."""
import os


def make_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY before starting an online generation or search command")
    return OpenAI(api_key=key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def chat(client, prompt, temperature, system="You are a senior prompt engineer.", model=None):
    response = client.chat.completions.create(
        model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        timeout=120,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("DeepSeek returned empty content")
    return content.strip()
