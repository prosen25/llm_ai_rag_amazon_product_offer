from agents.agent import Agent
from openai import OpenAI
from typing import List, Dict
import re

class FrontierAgent(Agent):

    # Set the name and color code to BLUE for logging
    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        """
        Set up this instance by connecting to OpenAI, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Frontier Agent is initializing")
        self.client = OpenAI()
        self.MODEL = "gpt-5.1"
        self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.embedding_model = "text-embedding-3-small"
        self.log("Frontier Agent is ready")

    def get_price(self, s: str) -> float:
        """
        An utility function that plucks a floating point from a string
        """
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def make_context(self, similar_products: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some of the products that might be similar to the one you need to estimate\n\n"
        for similar_product, price in zip(similar_products, prices):
            message += f"Potential related product:\n{similar_product}\nPrice is ${price:.2f}\n\n"

        return message

    def messages_for(self, description: str, similar_products: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        message = f"Estimate the price of this product. Respond with the price only, no explanation\n\n{description}\n\n"
        message += self.make_context(similar_products=similar_products, prices=prices)

        return [{"role": "user", "content": message}]

    def get_similar(self, description: str):
        """
        Return a list of items similar to the given on by looking in the chroma datastore
        """
        self.log("Frontier Agent is performing a RAG search of chroma datastore to find 5 similar items")
        response = self.client.embeddings.create(input=description, model=self.embedding_model)
        query_embedding = response.data[0].embedding
        results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
        documents = results["documents"][0][:]
        prices = [metadata["price"] for metadata in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices
    
    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        similar_products, prices = self.get_similar(description=description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description=description, similar_products=similar_products, prices=prices),
            seed=42,
            reasoning_effort="none"
        )
        reply = response.choices[0].message.content
        result = self.get_price(s=reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result