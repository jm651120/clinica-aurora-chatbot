"""
generation.py — Builds the RAG prompt and calls the Groq API directly.

Uses the groq SDK instead of langchain-groq to avoid proxy-argument
incompatibilities between langchain-groq and certain groq SDK versions.
"""

import os
from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """És o assistente virtual da Clínica Aurora, uma clínica de medicina estética em Lisboa.

Regras obrigatórias:
1. Responde SEMPRE em português europeu (de Portugal, não do Brasil).
   - NUNCA uses "você" — usa sempre formas impessoais ou "o/a cliente".
   - Usa "está" em vez de "você está", "pode" em vez de "você pode", etc.
   - Vocabulário europeu: "telemovel" não "celular", "autocarro" não "ônibus".
2. Tom: profissional, simpático e direto.
3. Baseia as tuas respostas EXCLUSIVAMENTE nas informações do CONTEXTO fornecido.
4. Se a informação não estiver no contexto, diz: "Não tenho essa informação disponível. Para mais detalhes, contacte-nos pelo 21 345 67 89 ou marcacoes@clinicaaurora.pt."
5. Nunca inventes preços, tratamentos, disponibilidades ou políticas.
6. Para marcações ou casos clínicos específicos, encaminha sempre para a equipa da clínica.
7. Mantém as respostas concisas — máximo 4 parágrafos curtos."""


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file and restart the app."
        )
    return Groq(api_key=api_key)


def generate_response(query: str, context: str, history: list[dict]) -> str:
    """
    Run the full RAG generation step and return the assistant's reply.

    history format: [{"role": "user"|"assistant", "content": "..."}]
    The last 6 turns are included to keep the context window manageable.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"CONTEXTO (informação da Clínica Aurora):\n{context}"
            f"\n\nPERGUNTA DO UTILIZADOR:\n{query}"
        ),
    })

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content
