FROM python:3.9

WORKDIR /app

ENV FILES_NEW=""
ENV MONGO_CONNECTION_STRING="localhost:27017"
ENV SELECTED_FILES="./src/documents_2023/pdfs"
ENV SELECTED_EXERCISES="./src/documents_2023/notebooks"
ENV PINECONE_INDEX_NAME="dbis-ss24"

RUN IFS=',' && for file in $FILES_NEW; do cp $file /app/src/documents_new; done

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]