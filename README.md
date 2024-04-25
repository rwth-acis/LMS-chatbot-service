# ![MoodleBot](./img/bot.png) LMS-chatbot-service 
This service contains the main functions of MoodleBot. MoodleBot is using Large Language Models to support students of the course Databases and Informationsystems. For building and deploying reliable GenAI, newest frameworks such as [LLamaIndex](https://www.llamaindex.ai/) and [LangChain](https://www.langchain.com/) are used. [Weaviate](https://weaviate.io/), an open source vector database is used to store lecture relevant materials. A LangChain agent answers the user's questions based on the query-relevant documents and can act like a tutor by asking course relevant questions. 

The service is connected to the [Social Bot Framework](https://github.com/rwth-acis/Social-Bot-Framework) and the [tech4comp Moodle](https://moodle.tech4comp.dbis.rwth-aachen.de). You can test it by enrolling to the course MoodleBot.

## Service Variables
For the service, you can set your individual environment variables. 
The following are the default settings:

| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | sk-*** | A paid OpenAI key is mandatory. |
| WEAVIATE_URL | http://localhost:8080 | URL specifying your weaviate database | 
| WEAVIATE_POD | localhost | specifying where your weaviate database is located. Port is by default 8080. |
| WEAVIATE_GRPC | localhost | specifying where your weaviate grpc is located. Port is by default 50051. |
| WCS_USERNAME | username | Weaviate username for authentification |
| WCS_PASSWORD | password | Weaviate password for authentification |
| MONGO_CONNECTION_STRING | mongodb://localhost:27017 | Name of your MongoDB database to store the chat conversations. |
| SELECTED_FILES | ./src/vl | The path of the files to be stored in weaviate. | 
