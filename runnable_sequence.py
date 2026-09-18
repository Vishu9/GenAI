from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence

from langchain_core.runnables import RunnableSequence


load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}.',
    input_variables=['topic']
)


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation",
)
# model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt2 = PromptTemplate(

    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain  = RunnableSequence(prompt1 , model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "Transformers"}))
