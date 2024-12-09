import spacy
from neo4j import GraphDatabase
import streamlit as st
from pyvis.network import Network
import networkx as nx

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit App Title
st.title("Paragraph-to-Paragraph Semantic Similarity with Knowledge Graph")

# Neo4j Connection Details
st.sidebar.header("Neo4j Database Details")
neo4j_uri = st.sidebar.text_input("Database URI", value="bolt://localhost:7687")
neo4j_user = st.sidebar.text_input("Username", value="neo4j")
neo4j_password = st.sidebar.text_input("Password", value="", type="password")

# Initialize Streamlit Session State
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "driver" not in st.session_state:
    st.session_state.driver = None
if "jaccard_score" not in st.session_state:
    st.session_state.jaccard_score = None
if "path_score" not in st.session_state:
    st.session_state.path_score = None
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False


def connect_to_database(uri, user, password):
    """
    Connect to Neo4j database and set the connection in session state.
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        st.session_state.db_connected = True
        st.session_state.driver = driver
        st.success("Successfully connected to Neo4j database!")
    except Exception as e:
        st.session_state.db_connected = False
        st.session_state.driver = None
        st.error(f"Failed to connect to Neo4j database: {e}")


def close_database_connection():
    """
    Close Neo4j database connection.
    """
    if st.session_state.driver:
        st.session_state.driver.close()
    st.session_state.db_connected = False
    st.session_state.driver = None
    st.info("Database connection closed.")


def run_neo4j_query(query, parameters=None):
    """
    Run a Neo4j query within a new session.
    """
    if not st.session_state.db_connected:
        raise Exception("Not connected to the database.")
    with st.session_state.driver.session() as session:
        return list(session.run(query, parameters))


# Clear Database
def clear_database():
    """
    Clears all nodes and relationships from the Neo4j database.
    """
    query = "MATCH (n) DETACH DELETE n"
    run_neo4j_query(query)


# Entity Extraction
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Relationship Extraction
def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        entities = [(ent.text, ent.label_) for ent in sent.ents]
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                source = entities[i][0]
                target = entities[i + 1][0]
                relationships.append((source, target, "RELATED_TO"))
    return relationships


# Populate Knowledge Graph
def populate_knowledge_graph(paragraph):
    entities = extract_entities(paragraph)
    relationships = extract_relationships(paragraph)

    for entity_name, entity_type in entities:
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type
        """
        run_neo4j_query(query, {"name": entity_name, "type": entity_type})

    for source, target, relationship_type in relationships:
        query = """
        MATCH (e1:Entity {name: $source})
        MATCH (e2:Entity {name: $target})
        MERGE (e1)-[:RELATES_TO {type: $relationship_type}]->(e2)
        """
        run_neo4j_query(query, {"source": source, "target": target, "relationship_type": relationship_type})


# Jaccard Similarity Calculation
def calculate_jaccard_similarity(entities1, entities2):
    set1 = set([e[0] for e in entities1])
    set2 = set([e[0] for e in entities2])
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0


# Path-Based Similarity Calculation
def calculate_path_similarity(entities1, entities2):
    similarities = []
    for e1 in entities1:
        for e2 in entities2:
            if e1[0] == e2[0]:  # Skip self-loops
                similarities.append(1.0)  # Full similarity for same entities
                continue

            query = """
            MATCH (a:Entity {name: $name1}), (b:Entity {name: $name2})
            WHERE a <> b
            MATCH p = shortestPath((a)-[*..5]-(b))
            RETURN length(p) AS path_length
            """
            result = run_neo4j_query(query, {"name1": e1[0], "name2": e2[0]})
            if result and result[0]["path_length"] is not None:
                similarities.append(1 / (1 + result[0]["path_length"]))  # Inverse of path length
    return sum(similarities) / len(similarities) if similarities else 0


# Knowledge Graph Visualization
def visualize_knowledge_graph():
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS source, type(r) AS relationship, m.name AS target
    """
    results = run_neo4j_query(query)
    edges = [(record["source"], record["target"], record["relationship"]) for record in results]

    # Create a NetworkX graph
    G = nx.DiGraph()
    for source, target, relationship in edges:
        G.add_edge(source, target, label=relationship)

    # Visualize with Pyvis
    net = Network(notebook=False, height="500px", width="100%", directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])

    # Save the graph to HTML
    html_file = "knowledge_graph.html"
    try:
        net.save_graph(html_file)
        return html_file
    except Exception as e:
        st.error(f"Failed to create knowledge graph visualization: {e}")
        return None


# Sidebar Buttons
if st.sidebar.button("Connect to Database"):
    connect_to_database(neo4j_uri, neo4j_user, neo4j_password)

if st.sidebar.button("Disconnect from Database"):
    close_database_connection()

# Main Section
if st.session_state.db_connected:
    paragraph1 = st.text_area("Enter Paragraph 1")
    paragraph2 = st.text_area("Enter Paragraph 2")

    if st.button("Calculate Similarity"):
        if not paragraph1 or not paragraph2:
            st.error("Please enter both paragraphs!")
        else:
            try:
                # Clear the database
                clear_database()

                # Populate Knowledge Graph
                populate_knowledge_graph(paragraph1)
                populate_knowledge_graph(paragraph2)

                # Extract Entities
                entities1 = extract_entities(paragraph1)
                entities2 = extract_entities(paragraph2)

                # Calculate Similarities
                st.session_state.jaccard_score = calculate_jaccard_similarity(entities1, entities2)
                st.session_state.path_score = calculate_path_similarity(entities1, entities2)

                # Display Results
                st.success(f"Jaccard Similarity: {st.session_state.jaccard_score:.2f}")
                st.success(f"Path-Based Similarity: {st.session_state.path_score:.2f}")

                # Enable graph display
                st.session_state.graph_ready = True

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Show Knowledge Graph button only after similarity scores are displayed
    if st.session_state.graph_ready:
        if st.button("Show Knowledge Graph"):
            try:
                graph_html = visualize_knowledge_graph()
                if graph_html:  # Only display if the HTML file was successfully created
                    with open(graph_html, "r") as f:
                        graph_html_content = f.read()
                    st.components.v1.html(graph_html_content, height=600)
                else:
                    st.error("Failed to create graph visualization.")
            except Exception as e:
                st.error(f"Error visualizing graph: {e}")

else:
    st.info("Connect to the Neo4j database to enable paragraph inputs.")