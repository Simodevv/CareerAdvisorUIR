import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from kotaemon.base import KPipeline


# Set environment variables for Google and Neo4j
os.environ['GOOGLE_API_KEY'] = "AIzaSyBLl7qjrGw4_P9kEofM_hxCav0SI7-GyZk"
os.environ['NEO4J_URL'] = "bolt://localhost:7687"
os.environ['NEO4J_USERNAME'] = "neo4j"
os.environ['NEO4J_PASSWORD'] = "12345678"


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

neo4j_url = os.getenv('NEO4J_URL')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')

graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password, database="hamzadb")

if graph is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        resume = open("cv.pdf", "rb")
        tmp_file.write(resume.read())
        tmp_file_path = tmp_file.name

        reader = PdfReader(tmp_file_path)
        pages = "\n".join([page.extract_text() for page in reader.pages])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        docs = text_splitter.split_documents([Document(page_content=page) for page in pages])

        lc_docs = []
        for doc in docs:
            lc_docs.append(
                Document(page_content=doc.page_content.replace("\n", ""), metadata={'source': resume.name}))

            # Define allowed nodes and relationships
            allowed_nodes = ["Skill", "JobRole", "Certification"]
            allowed_relationships = ["REQUIRES_SKILL", "REQUIRES_CERTIFICATION"]

            nodes_properties = {
                "JobRole": {"description": "string", "averageSalary": "integer", "title": "string"},
                "Skill": {"proficiencyLevel": "string", "popularity": "string", "name": "string"},
                "Certification": {"name": "string", "provider": "string", "id": "integer", "description": "string"}
            }

            relationships_properties = {
                "REQUIRES_SKILL": {"importance": "string", "relevance": "integer"}
            }

            # Transform documents into graph documents
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                node_properties=nodes_properties,
                relationship_properties=relationships_properties
            )

            graph_documents = transformer.convert_to_graph_documents(lc_docs)
            graph.add_graph_documents(graph_documents, include_source=True)

            # Set up the QA Chain with Cypher query generation
            template = """
                Task: Generate a Cypher statement to query the graph database.

                Instructions:
                Use only relationship types and properties provided in schema.
                Do not use other relationship types or properties that are not provided.

                schema:
                {schema}

                Note: Do not include explanations or apologies in your answers.
                Do not answer questions that ask anything other than creating Cypher statements.
                Do not include any text other than generated Cypher statements.
                If you don't know the answer, answer with your own gemini words

                Question: {question}"""

            question_prompt = PromptTemplate(
                template=template,
                input_variables=["schema", "question"]
            )

            qa = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                cypher_prompt=question_prompt,
                verbose=True,
                allow_dangerous_requests=True
            )
            instance = KPipeline(transformer=transformer, qa=qa)

else:
    print("Could not connect to Neo4j!")
