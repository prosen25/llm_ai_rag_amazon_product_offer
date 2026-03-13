from agents.preprocessor import Preprocessor
from agents.agent import Agent
from agents.frontier_agent import FrontierAgent
from agents.specialist_agent import SpecialistAgent
from agents.neural_network_agent import NeuralNetworkAgent

class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Create an instance of the Ensemble, by creating each of the models
        """
        self.log("Ensemble Agent is initializing")
        self.preprocessor = Preprocessor()
        self.frontier = FrontierAgent(collection=collection)
        self.specialist = SpecialistAgent()
        self.neural_network = NeuralNetworkAgent()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Run this Ensemble model
        Ask each of the model to price the product
        Then use linear regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - processing text")
        rewrite_description = self.preprocessor.preprocess(text=description)
        self.log(f"Preprocessed text using {self.preprocessor.model_name}")
        price_frontier = self.frontier.price(description=rewrite_description)
        price_specialist = self.specialist.price(description=rewrite_description)
        price_neural_network = self.neural_network.price(description=rewrite_description)
        combined = price_frontier * 0.8 + price_specialist * 0.1 + price_neural_network * 0.1
        self.log(f"Ensemble Agent Completed - returning ${combined:.2f}")
        return combined