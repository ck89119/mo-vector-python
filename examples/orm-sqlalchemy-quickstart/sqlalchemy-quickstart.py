import os
import dotenv

from sqlalchemy import Column, Integer, create_engine, Text, URL
from sqlalchemy.orm import declarative_base, Session
from mo_vector.sqlalchemy import VectorType, VectorAdaptor

dotenv.load_dotenv()

# Step 1: Connect to MO using SQLAlchemy.

# Using `pymysql` as the driver.
drivername = 'mysql+pymysql'

engine = create_engine(URL.create(
    drivername=drivername,
    username=os.environ['USERNAME'],
    password=os.environ['PASSWORD'],
    host=os.environ['HOST'],
    port=os.environ['PORT'],
    database=os.environ['DATABASE'],
    query={},
))


# Step 2: Define a table with a vector column.
Base = declarative_base()


class Document(Base):
    __tablename__ = 'sqlalchemy_demo_documents'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(VectorType(3))


# Or add HNSW index when creating table.
class DocumentWithIndex(Base):
    __tablename__ = 'sqlalchemy_demo_documents_with_index'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(VectorType(3))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


def test_document():
    # Step 3: Insert embeddings into the table.
    with Session(engine) as session:
        session.add(Document(content="dog", embedding=[1, 2, 1]))
        session.add(Document(content="fish", embedding=[1, 2, 4]))
        session.add(Document(content="tree", embedding=[1, 0, 0]))
        session.commit()

    # Step 4: Get the 3-nearest neighbor documents.
    print('Get 3-nearest neighbor documents:')
    with Session(engine) as session:
        distance = Document.embedding.cosine_distance([1, 2, 3]).label('distance')
        results = session.query(Document, distance).order_by(distance).limit(3).all()

        for doc, distance in results:
            print(f'  - distance: {distance}\n'
                  f'    document: {doc.content}')

    # Step 5: Get documents within a certain distance.
    print('Get documents within a certain distance:')
    with Session(engine) as session:
        distance = Document.embedding.cosine_distance([1, 2, 3]).label('distance')
        results = session.query(
            Document, distance
        ).filter(distance < 0.2).order_by(distance).limit(3).all()

        for doc, distance in results:
            print(f'  - distance: {distance}\n'
                  f'    document: {doc.content}')


def test_document_with_index():
    VectorAdaptor(engine).create_vector_index(
        DocumentWithIndex.embedding,
        skip_existing=True,
    )

    # Step 3: Insert embeddings into the table.
    with Session(engine) as session:
        session.add(DocumentWithIndex(content="dog", embedding=[1, 2, 1]))
        session.add(DocumentWithIndex(content="fish", embedding=[1, 2, 4]))
        session.add(DocumentWithIndex(content="tree", embedding=[1, 0, 0]))
        session.commit()

    # Step 4: Get the 3-nearest neighbor documents.
    print('Get 3-nearest neighbor documents:')
    with Session(engine) as session:
        distance = DocumentWithIndex.embedding.cosine_distance([1, 2, 3]).label('distance')
        results = session.query(DocumentWithIndex, distance).order_by(distance).limit(3).all()

        for doc, distance in results:
            print(f'  - distance: {distance}\n'
                  f'    document: {doc.content}')

    # Step 5: Get documents within a certain distance.
    print('Get documents within a certain distance:')
    with Session(engine) as session:
        distance = DocumentWithIndex.embedding.cosine_distance([1, 2, 3]).label('distance')
        results = session.query(
            DocumentWithIndex, distance
        ).filter(distance < 0.2).order_by(distance).limit(3).all()

        for doc, distance in results:
            print(f'  - distance: {distance}\n'
                  f'    document: {doc.content}')


test_document()
test_document_with_index()