from agents.agent import Agent
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from openai import OpenAI
from typing import List, Optional, Dict
from agents.deals import Opportunity, Deal
import json

class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-5.1"

    def __init__(self, collection):
        """
        Create instance of 3 Agents that this planner cooridnates 
        """
        self.log("Autonomous Planning Agent initializing")
        self.scanner_agent = ScannerAgent()
        self.ensemble_agent = EnsembleAgent(collection=collection)
        self.messsaging_agent = MessagingAgent()
        self.openai = OpenAI()
        self.memory = None
        self.opportunity = None
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the tool to scan internet for deals
        """
        self.log("Autonomous Planning Agent is calling Scanner Agent")
        results =self.scanner_agent.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deal found"

    def estimate_true_value(self, description: str) -> str:
        """
        Run tool to estimate tru value of the product description
        """
        self.log("Autonomous Planning Agent is calling Ensemble Agent")
        estimate = self.ensemble_agent.price(description=description)
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
        """
        Run the tool to notify user
        """
        if self.opportunity:
            self.log("Autonomous Planning Agent is trying to notify the user second time; ignoring")
        else:
            self.log("Autonomous Planning Agent is calling Messaging Agent to notify user")
            self.messsaging_agent.notify(description=description, deal_price=deal_price, estimated_true_value=estimated_true_value, url=url)
            deal = Deal(product_description=description, price=deal_price, url=url)
            discount = estimated_true_value - deal_price
            self.opportunity = Opportunity(deal=deal, estimate=estimated_true_value, discount=discount)

        return "Notification sent ok"

    def get_tools(self) -> List[Dict]:
        """
        Return the json for the tools to be used
        """
        scan_function = {
            "name": "scan_the_internet_for_bargains",
            "description": "Return top burgains scraped from the intenet along with the price each item being offered for",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }

        estimate_function = {
            "name": "estimate_true_value",
            "description": "Given the description of the item, estimate how much it really worth",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The description of the item to be estimated"
                    }
                },
                "required": ["description"],
                "additionalProperties": False
            }
        }

        notify_function = {
            "name": "notify_user_of_deal",
            "description": "Send the user a push notification about the single most compelling deal; only call this tool once",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The description of the item iteslf scraped from the internet"
                    },
                    "deal_price": {
                        "type": "number",
                        "description": "The price offered by this deal scraped from the internet"
                    },
                    "estimated_true_value": {
                        "type": "number",
                        "description": "The estimated actual value of this item"
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL of this deal as scraped from the internet"
                    }
                },
                "required": ["description", "deal_price", "estimated_true_value", "url"],
                "additionalProperties": False
            }
        }

        tools = [
            {"type": "function", "function": scan_function},
            {"type": "function", "function": estimate_function},
            {"type": "function", "function": notify_function}
        ]

        return tools
    
    def handle_tool_call(self, message) -> List[Dict]:
        """
        Actually call tools assciated with the message
        """
        tool_mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = tool_mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})

        return results

    def prepare_llm_messages(self) -> List[Dict]:
        """
        Prepare system and user prompt and create the messages for llm call
        """
        system_prompt = "You find great deals on bargain products using your tools, and notify the user of the best bargain"
        user_prompt = """First use your tool to scan the internet for bargain deals. Then for each deal, use your tool to 
        estimate its true value. Then pick the single most compelling deal where price is much lower than the estimated value, 
        and use your tool to notify the user. Then just reply OK to indicate success."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the full workflow, providing the LLM with tools to surface scraped deals to the user
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.prepare_llm_messages()[:]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message=message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True

        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity