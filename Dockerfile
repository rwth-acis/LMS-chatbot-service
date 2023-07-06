FROM python:3.9

WORKDIR /app

ENV FILES_NEW=""
ENV OPENAI_API_KEY="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"
ENV PINECONE_API_KEY="470a0520-e69d-4d14-83fa-2a9d3c3d4f4f"
ENV PINECONE_ENVIRONMENT="us-west4-gcp"
ENV MONGO_CONNECTION_STRING="localhost:27017"
ENV SELECTED_FILES="./src/documents_2023/pdfs"
ENV SELECTED_EXERCISES="./src/documents_2023/notebooks"
ENV PINECONE_INDEX_NAME="dbis-lecture-2023"

RUN IFS=',' && for file in $FILES_NEW; do cp $file /app/src/documents_new; done

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]