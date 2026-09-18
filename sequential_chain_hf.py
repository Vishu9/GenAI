import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# HuggingFaceEndpoint needs a Hugging Face token (read from HUGGINGFACEHUB_API_TOKEN
# or HF_TOKEN). Fail fast with a clear message instead of a deep stack trace.
if not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")):
    raise RuntimeError(
        "Missing Hugging Face API token. Add HUGGINGFACEHUB_API_TOKEN=hf_xxx "
        "to your .env file (get one at https://huggingface.co/settings/tokens)."
    )

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"],
) 

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",    
    input_variables=["text"],
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result  = chain.invoke({"topic": "Transformers"})

print(result)

chain.get_graph().print_ascii()


