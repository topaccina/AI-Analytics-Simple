from dash import Dash, Input, State, Output, callback, ctx, dcc, dash_table, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# from langchain_community.document_loaders import JSONLoader
import json


#######################################################
import warnings

warnings.filterwarnings(action="ignore")
#######################################################


#######################################################
def getEnvVar():

    load_dotenv(override=True)
    API_KEY = os.environ.get("OPENAI_API_KEY")
    return API_KEY


OPENAI_API_KEY = getEnvVar()
#######################################################


llm = ChatOpenAI(model="gpt-4o", temperature=0)
#######################################################

system_prompt = """You are a senior analyst  and statistician with deep knowledge of python and plotly visualization. You are an expert in get insights from visualization.
you can understand a plotly visualization serialized in json format and converted to a dictionary and your are good in math and provide insigths according with the user questions.
Report statistics numbers when possible. 

"""
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            "can you provide insigths from this plot converted in a dictionary:{data}. List max {number} insights in markdown format. Not exceed 100 words each ",
        ),
    ]
)


df = px.data.tips()
fig = px.box(
    df, x="time", y="total_bill", points="all", title="Restaurant Bill vs Time"
)

chain = final_prompt | llm

jsonFig = pio.to_json(fig, validate=True, pretty=False, remove_uids=True, engine=None)


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("AI Analytics - Simple Test ", className="ms-2 "),
        ]
    ),
    color="dark",
    dark=True,
)
app = Dash(external_stylesheets=[dbc.themes.CERULEAN])
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col([navbar])),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                html.H5("Plotly Chart"),
                                dcc.Graph(figure=fig, id="id-fig"),
                            ],
                            className="mt-5 mb-1 ms-5 me-3",
                        ),
                        dbc.Button(["Get Insights"], id="id-button", className="ms-5"),
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                html.H5("Chart JSON format"),
                                html.Pre(
                                    [
                                        ""
                                        + str(json.dumps(jsonFig, indent="\t")).replace(
                                            "\\", ""
                                        )
                                    ],
                                    style={
                                        "height": 400,
                                        "whiteSpace": "pre-wrap",
                                        "font-size": 10,
                                        "margin": "10px",
                                    },
                                ),
                            ],
                            className="mt-5 mb-1 ms-2 me-5",
                        ),
                    ],
                    width=4,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Container(
                                    [
                                        # html.P(["Data Insights"]),
                                        dcc.Loading(
                                            dcc.Markdown(
                                                [
                                                    """ Click the button to get data insights
"""
                                                ],
                                                id="id-markdown",
                                                className="m-5",
                                            ),
                                        )
                                    ]
                                )
                            ],
                            width=10,
                        ),
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    Output("id-markdown", "children"),
    Input("id-button", "n_clicks"),
    prevent_initial_call=True,
)
def chat(n):
    outLLM = chain.invoke({"data": jsonFig, "number": 2})
    text_out = outLLM.content
    return text_out


if __name__ == "__main__":
    app.run_server(debug=True)
