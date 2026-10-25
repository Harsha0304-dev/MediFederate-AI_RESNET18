import os,json
def generate_report(findings):
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"): return "OPENAI_API_KEY is not configured."
    c=OpenAI()
    r=c.responses.create(model=os.getenv("OPENAI_MODEL","gpt-5.6-mini"),
      input="Explain these research model findings without diagnosing or inventing facts. Label measurements as model-derived:\n"+json.dumps(findings,indent=2))
    return r.output_text
