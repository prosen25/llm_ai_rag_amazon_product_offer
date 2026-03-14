import logging
import sys
import chromadb
import os
import json
import numpy as np

from typing import List
from dotenv import load_dotenv
from sklearn.manifold import TSNE

from agents.deals import Opportunity
from agents.planning_agent import PlanningAgent

load_dotenv(override=True)

# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]

def init_logging():
    """
    Configure the logging
    """
    root = logging.getLogger()
    root.setLevel(level=logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z"
    )
    handler.setFormatter(fmt=formatter)
    root.addHandler(handler)

class DealAgentFramework:
    VECTOR_DB = "products_vectorstore"
    COLLECTION_NAME = "products"
    MEMORY_FILE = "memory.json"

    def __init__(self):
        """
        Setup vector database collection and memory
        """
        init_logging()
        vector_client = chromadb.PersistentClient(path=self.VECTOR_DB)
        self.collection = vector_client.get_or_create_collection(name=self.COLLECTION_NAME)
        self.memory = self.read_memory()
        self.planner = None

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def init_agent_as_needed(self) -> None:
        """
        Initialize Planning Agent if not available
        """
        if not self.planner:
            self.log("Agent Framework is initializing")
            self.planner = PlanningAgent(collection=self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        """
        Read the memory json
        """
        if os.path.exists(path=self.MEMORY_FILE):
            with open(file=self.MEMORY_FILE, mode="r") as file:
                data = json.load(file)
            opportunities = [Opportunity(**item) for item in data]
            return opportunities
        return []
    
    def write_memory(self) -> None:
        """
        Write the deal into memory
        """
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(file=self.MEMORY_FILE, mode="w") as file:
            json.dump(data, file, indent=2)
    
    def run(self) -> List[Opportunity]:
        """
        Run Planning Agent to notify user with the best deal from internet
        """
        self.init_agent_as_needed()
        self.log("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        self.log(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory
    
    @classmethod
    def reset_memory(cls) -> None:
        data = []
        if os.path.exists(cls.MEMORY_FILE):
            with open(cls.MEMORY_FILE, "r") as file:
                data = json.load(file)
        truncated = data[:2]
        with open(cls.MEMORY_FILE, "w") as file:
            json.dump(truncated, file, indent=2)

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        client = chromadb.PersistentClient(path=cls.VECTOR_DB)
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors
    
if __name__ == "__main__":
    DealAgentFramework().run()