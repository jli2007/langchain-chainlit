from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
import chainlit as cl
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def start():
    await upload_file()

async def upload_file():
    files = None
    
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to start chatting!",
            accept=["application/pdf"],
            max_size_mb=100,
            max_files=1,
        ).send()
    
    file = files[0]
    
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        with open(file.path, "rb") as f:   # ✅ use the file path
            temp.write(f.read())
        temp_path = temp.name
    
    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load_and_split()
        
        # Combine all pages into one text
        text = ' '.join([page.page_content for page in pages])
        
        # Split text into chunks
        texts = text_splitter.split_text(text)
        
        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-chunk"} for i in range(len(texts))]
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )
        
        # Create the retrieval chain (finds relevant text chunks)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(temperature=0, streaming=True),
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Store the chain and texts in user session
        cl.user_session.set("chain", chain)
        cl.user_session.set("texts", texts)
        cl.user_session.set("metadatas", metadatas)
        
        msg.content = f"`{file.name}` processed successfully! You can now ask questions about the document."
        
        # Update the message
        await msg.update() #type: ignore
        
    except Exception as e:
        msg.content = f"Error processing `{file.name}`: {str(e)}"
        await msg.update() #type: ignore
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    
    if not chain:
        await cl.Message(content="Please upload a PDF file first!").send()
        return
    
    # Run the chain
    try:
        cb = cl.AsyncLangchainCallbackHandler()
        res = await chain.acall(message.content, callbacks=[cb])
        
        answer = res["answer"]
        sources = res.get("sources", "").strip()
        source_elements = []
        
        # Get stored data
        texts = cl.user_session.get("texts", [])
        metadatas = cl.user_session.get("metadatas", [])
        all_sources = [m["source"] for m in metadatas] #type: ignore
        
        if sources:
            found_sources = []
            
            # Process each source
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                    text_content = texts[index] #type: ignore
                    found_sources.append(source_name)
                    
                    # Create text element for the source
                    source_elements.append(
                        cl.Text(content=text_content, name=source_name)
                    )
                except (ValueError, IndexError):
                    continue
            
            # Add sources info to answer
            if found_sources:
                answer += f"\n\n**Sources:** {', '.join(found_sources)}"
            else:
                answer += "\n\n*No sources found*"
        
        # Send the response
        await cl.Message(content=answer, elements=source_elements).send()
        
    except Exception as e:
        await cl.Message(content=f"Error processing your question: {str(e)}").send()