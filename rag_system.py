from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import streamlit as st

from config import *
from prompt import *
@st.cache_resource
def initializa_rag_system():
    # vector store
    vectorstore=Chroma(
embedding_function=OllamaEmbeddings(model=EMBEDGIND_MODEL),
persist_directory=CHROMA_DB_PATH

    )

    #MODELOS
    llm_queries=ChatOllama(model=QUERY_MODEL,temperature=0)
    llm_generation=ChatOllama(model=GENERATION_MODEL,temperature=0)

    #retriever MMR(MAXIMAÑ MARGIN RELEVANCE)
    base_retriever=vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k":SEARCH_K,
            "lambda_mult":MMR_DIVERSITY_LAMBDA,
            "fetch_k":MMR_FECTH_K


        })
    #Prompt perzonalizado para multiqueryRetrever
    multi_query_prompt=PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    #multiqueryRetriever com prompt perzonalizado
    mmr_multi_retriever=MultiQueryRetriever.from_llm(
            
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt

    )
    prompt=PromptTemplate.from_template(RAG_TEMPLATE)

    #funcion para formatear  y preprocesar los documentos recuperados

    def format_docs(docs):
        formtated=[]
        for i , doc in enumerate(docs,1):
            header=f"[Fragmento {1}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source=doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header+=f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header+=f" - Pagina: {doc.metadata['page']}"
            content=doc.page_content.strip()
            formtated.append(f"{header}\n{content}")
        return "\n\n".join(formtated)



    rag_chain=(

        {
            "context":mmr_multi_retriever | format_docs,
            "question":RunnablePassthrough()
        }
        | prompt 
        | llm_generation 
        | StrOutputParser()
        
        )
    return rag_chain,mmr_multi_retriever


def query_rag(question):
    try:
        rag_chain,retriever=initializa_rag_system()
        #obtener la respouetsa
        response=rag_chain.invoke(question)
        #obtener docum,entos para mostrarlos
        docs=retriever.invoke(question)
        #formatear los documentos para mostrar
        docs_info=[]
        for i ,doc in enumerate(docs[:SEARCH_K],1):
            doc_info={
                "fragmento":i,
                "contenido":doc.page_content[:1000]+"..." if len(doc.page_content)>1000 else doc.page_content,
                "fuente":doc.metadata.get('source','No especificada').split("\\")[-1],
                "pagina":doc.metadata.get('page','No especificada')

            }
            docs_info.append(doc_info)
        return response,docs_info
    except Exception as e:
        error_msg=f"error al procesar la consulta:  {str(e)}"
        return error_msg,[]

def get_retriever_info():
    """Obtiene informacion sobre la configuracion del retriever"""
    return{
        "tipo":f"{SEARCH_TYPE.upper()}",
        "documentos":SEARCH_K,
        "diversidad":MMR_DIVERSITY_LAMBDA,
        "candidatos":MMR_FECTH_K,
        "umbral":None
    }



    
