import logging
import queue
import threading
import time

import gradio as gr
import plotly.graph_objects as go

from dotenv import load_dotenv
from log_utils import reformat

from deal_agent_framework import DealAgentFramework

load_dotenv(override=True)

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record) -> None:
        self.log_queue.put(self.format(record))

def html_for(log_data) -> str:
    """
    HTML formatting of log_data
    """
    output = "<br>".join(log_data[-15:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
    {output}
    </div>
    """

def setup_logging(log_queue) -> None:
    """
    Format log output
    """
    handler = QueueHandler(log_queue=log_queue)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class App:

    def __init__(self):
        """
        Declare the Agent Frame
        """
        self.agent_framework = None

    def get_agent_framework(self):
        """
        Create deal agent framework if neeeded
        """
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):
        """
        Run the app
        """
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_data = gr.State([])

            def table_for(opps):
                """
                Format the deal opportunities
                """
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url
                    ]
                    for opp in opps
                ]

            def update_output(log_data, log_queue, result_queue):
                """
                Update the output
                """
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message=message))
                        yield log_data, html_for(log_data=log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield log_data, html_for(log_data=log_data), final_result or initial_result
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)
            
            def get_plot():
                """
                Create the 3D scatter plot
                """
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            z=vectors[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=colors, opacity=0.7)
                        )
                    ]
                )

                fig.update_layout(
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),
                        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))
                    ),
                    height=400,
                    margin=dict(r=5, b=1, l=1, t=2)
                )

                return fig
            
            def do_run():
                """
                Get opportunities from the internet
                """
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table
            
            def run_with_logging(initial_log_data):
                """
                Run agent in logging mode
                """
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue=log_queue)

                def worker():
                    result = do_run()
                    result_queue.put(result)

                thread = threading.Thread(target=worker)
                thread.start()

                for log_data, output, final_result in update_output(
                    log_data=initial_log_data, log_queue=log_queue, result_queue=result_queue
                ):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData):
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.get_agent_framework().planner.messaging_agent.alert(opportunity=opportunity)

            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Description", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[4, 1, 1, 1, 2],
                    row_count=10,
                    col_count=5,
                    max_height=400
                )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(value=get_plot(), show_label=False)

            ui.load(
                fn=run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe]
            )

            timer = gr.Timer(value=300, active=True)
            timer.tick(
                fn=run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe]
            )

            opportunities_dataframe.select(do_select)

        ui.launch(inbrowser=True)

if __name__ == "__main__":
    App().run()