from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_openai import ChatOpenAI
from langchain_core.promts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence


load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}.',
    input_variables=['topic']
)


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
# model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain  = RunnableSequence(prompt, model, parser)

print(chain.invoke({"topic": "Transformers"}))
