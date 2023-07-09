# LMS-chatbot-service
This service is using llama-index to store files into a Pinecone vector database. 
With langchain, an agent is created to retrieve documents from the index and answer the user's questions based on the retrieved documents.
It acts like a tutor of the lecture DBIS, but can be used for other lecture by setting the environment variables.

The service is connected to the [Social Bot Framework](https://github.com/rwth-acis/Social-Bot-Framework) and the [tech4comp Moodle](https://moodle.tech4comp.dbis.rwth-aachen.de). You can test it by contacting the user "Ask a Bot".

## Service Variables
For the service, you can set your individual environment variables. 
The following are the default settings:

| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | OpenAI key from the LLMGroup | A paid OpenAI key is mandatory. |
| PINECONE_API_KEY | Own Pinecone API key | A Pinecone API key is mandatory for the database connection. | 
| PINECONE_ENVIRONMENT | us-west4-gcp | The index env variable from Pinecone. |
| MONGO_CONNECTION_STRING | mongodb://localhost:27017 | Name of your MongoDB database to store the chat conversations. |
| SELECTED_FILES | ./src/documents_2023/pdfs | The path of the files to be stored in the Pinecone database. | 
| PINECONE_INDEX_NAME | dbis-lecture-2023 | Set your own Pinecone Index Name |