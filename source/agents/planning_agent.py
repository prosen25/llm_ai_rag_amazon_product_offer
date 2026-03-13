from agents.agent import Agent
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.deals import Opportunity, Deal
from typing import Optional, List

class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Create instance of 3 agents which this planning agent will coordinates
        """
        self.log("Planning Agent is initializing")
        self.scanner_agent = ScannerAgent()
        self.ensemble_agent = EnsembleAgent(collection=collection)
        self.messaging_agent = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Planning Agent is pricing a potential deal")
        estimate = self.ensemble_agent.price(description=deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount {discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from internet
        2. Use the EnsembleAgent to estimate the true value
        3. Use the MessagingAgent to send a notification of deal
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None 
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner_agent.scan(memory=memory)
        if selection:
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            if best.discount > self.DEAL_THRESHOLD:
                self.messaging_agent.alert(opportunity=best)
            self.log("Planning Agent has completed a run")
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None
