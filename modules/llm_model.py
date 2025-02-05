from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import json
import pandas as pd


from dotenv import load_dotenv
load_dotenv()

pd.set_option("display.max_columns", None)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

system = '''You are an Assistance bot who is good at understanding text, specifically invoices. The user will provide you with invoice text, and your task is to convert it to JSON format.
extract info like menu, sub_total, total

For eg.

{{
   'menu': [{{'nm': '<>',
   'cnt': {{'nm': '<>',
    'unitprice': '<>'}},
   'unitprice': '<>',
   'price': '<>'}},
  {{'nm': '<>',
   'cnt': {{'unitprice': '<>'}},
   'price': {{'cnt': '<>', 'unitprice': '<>'}},
  {{'nm': '<>',
   'unitprice': '<>',
   'cnt': '<>',
   'price': '<>'}},
  {{'cnt': '<>', 'price': '<>'}}],

 'sub_total': {{'subtotal_price': {{'cnt': '<>'}},

 'total': {{'total_price': '<>',
  'cashprice': '<>',
  'changeprice': '<>',
  'creditcardprice': '<>',
  'menuqty_cnt': '<>'}}
}}

'''


def json2df(json_data) -> pd.DataFrame:
    """
    Convert JSON invoice data into a single Pandas DataFrame.
    """

    dfs = []

    # Extract common fields
    for key, value in json_data.items():
        dfs.append(
            pd.json_normalize(value)
        )

    return dfs


def llm_pipeline_get_df(text: str) -> pd.DataFrame:
    prompt_template = ChatPromptTemplate([
        ("system", system),
        ("human", "invoice data : {invoice_data}")
    ])


    def output(data):
        return json.loads(
            data
            .content
            .replace("```", "")
            .replace("json", "")
        )


    chaining = (
        prompt_template |
        model |
        output
    )

    json_output = chaining.invoke(input={"invoice_data" : text})
    return json2df(json_output)
