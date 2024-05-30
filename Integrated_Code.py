import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import plotly.express as px
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import base64
import pandasql as ps
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from io import BytesIO
import io
from devices_library import *
global history

history = []

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"
global model
model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2024-03-01-preview',temperature = 0.0)
####################################################################################################################----------------Copilot-------------------#####################################################################################################

Copilot_Sentiment_Data  = pd.read_csv("New_CoPilot_Data.csv")

st.markdown("""
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
</head>
""", unsafe_allow_html=True)

def Sentiment_Score_Derivation(value):
    try:
        if value == "positive":
            return 1
        elif value == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Copilot_Sentiment_Data["Sentiment_Score"] = Copilot_Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Copilot_Sentiment_Data["Review_Count"] = 1.0

overall_net_sentiment = round(sum(Copilot_Sentiment_Data["Sentiment_Score"])*100/sum(Copilot_Sentiment_Data["Review_Count"]),1)
overall_review_count = sum(Copilot_Sentiment_Data["Review_Count"])


def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err


# In[5]:


def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        
        if '!=' in query or '=' in query:
            query = query.replace("!="," NOT LIKE ")
            query = query.replace("="," LIKE ")
            
            pattern = r"LIKE\s'([^']*)'"
            def add_percentage_signs(match):
                return f"LIKE '%{match.group(1)}%'"
            query = re.sub(pattern, add_percentage_signs, query)
        
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err
        
 
#-------------------------------------------------------------------------------------------------------Summarization-------------------------------------------------------------------------------------------------------#
 
def get_conversational_chain_quant():
    global model
    try:
        prompt_template = """
        
        You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below.
            1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
            2. There is only one table with table name Copilot_Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                Geography: From which Country or Region the review was given. It contains different Geography.
                           list of Geographies in the table - Values in this column [China,France,Japan,US,Brazil,Canada,Germany,India,Mexico,UK,Australia,Unknown,Venezuela,Vietnam,Cuba,Colombia,Iran,Ukraine,Northern Mariana Islands,Uruguay,Taiwan,Spain,Russia,Bolivia,Argentina,Lebanon,Finland,Saudi Arabia,Oman,United Arab Emirates,Austria,Luxembourg,Macedonia,Puerto Rico,Bulgaria,Qatar,Belgium,Italy,Switzerland,Peru,Czech Republic,Thailand,Greece,Netherlands,Romania,Indonesia,Benin,Sweden,South Korea,Poland,Portugal,Tonga,Norway,Denmark,Samoa,Ireland,Turkey,Ecuador,Guernsey,Botswana,Kenya,Chad,Bangladesh,Nigeria,Singapore,Malaysia,Malawi,Georgia,Hong Kong,Philippines,South Africa,Jordan,New Zealand,Pakistan,Nepal,Jamaica,Egypt,Macao,Bahrain,Tanzania,Zimbabwe,Serbia,Estonia,Jersey,Afghanistan,Kuwait,Tunisia,Israel,Slovakia,Panama,British Indian Ocean Territory,Comoros,Kazakhstan,Maldives,Kosovo,Ghana,Costa Rica,Belarus,Sri Lanka,Cameroon,San Marino,Antigua and Barbuda]
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "COPILOT"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Product Names  - It contains following Values - [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                
        IMPORTANT : User won't exactly mention the exact Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
                    Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                        We know that Chinanews in not any of the DataSource, Geography and so on.
                        So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues when we pull SQL Queries
                    
                    Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                        We know that USA in not any of the Geography, Data Source and so on.
                        So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                        
                        Same goes for all the columns
                
        3. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Copilot_Sentiment_Data
                    ORDER BY Net_Sentiment DESC
        4. If an user is asking for Summarize reviews of any product. Note that user is not seeking for reviews, user is seeking for all the Quantitative results product(Net Sentiment & Review Count) and also (Aspect wise sentiment and Aspect wise review count). So choose to Provide Net Sentiment and Review Count and Aspect wise sentiment and their respective review count and Union them in single table
        
        5. IMPORTANT : CoPilot is Overall Product and Product_Family are different versions of CoPilot.
        
        6. Example : If the user Quesiton is "Summarize reviews of Github Copilot"
        
                6.1 "Summarize reviews of CoPilot for Mobile" - User seeks for net sentiment and aspect wise net sentiment of "CoPilot for Mobile" Product Family and their respective aspect review count in a single table
                    
                    The Query has to be like this 
                        
                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Aspect
                        ORDER BY Review_Count DESC
                    
                6.2 And if user specifies any Geography/DataSource. Make sure to apply those filters in the SQL Query response
                
                    if Geography is included:
                    
                            SELECT 'TOTAL' AS Aspect, 
                            ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                            SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            WHERE Product_Family LIKE '%CoPilot for Mobile%'
                            GEOGRAPHY LIKE '%US%'

                            UNION

                            SELECT Aspect, 
                            ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                            SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            WHERE Product_Family LIKE '%CoPilot for Mobile%' AND
                            GEOGRAPHY LIKE '%US%'
                            GROUP BY Aspect
                            ORDER BY Review_Count DESC

                    
                    
                6.3 IMPORTANT : if any particular Aspect "Code Generation" in user question:
                    

                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Copilot_Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Aspect
                        HAVING Aspect LIKE %'Code Generation'%
                        ORDER BY Review_Count DESC
            
        7. It the user wants to compare features of 2 different ProductFamily, let's say "Github CoPilot" and "CoPilot for Microsoft 365". I want the aspect wise sentiment of both the devices in one table.
        
                IMPORTANT - DO NOT ORDER BY REVIEW COUNT in this case.
        
                7.1 IMPORTANT : Example USE THIS Query for COMPARISION -  "Compare different features of CoPilot for Mobile and GitHub CoPilot" 
        
                        Query: 
        
                            SELECT 'GITHUB COPILOT' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'

                            UNION All

                            SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'
                            GROUP BY ASPECT

                            UNION All

                            SELECT 'COPILOT FOR MICROSOFT 365' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'

                            UNION All

                            SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'
                            GROUP BY ASPECT
                    
                7.2 IMPORTANT : Example USE THIS Query for COMPARISION Query - :  if only one aspect (Use always 'LIKE' OPERATOR) for ASPECT, GEOGRAPHY, PRODUCT_FAMILY, PRODUCT and so on while performing where condition. 
        
                            SELECT 'GITHUB COPILOT' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'

                            UNION All

                            SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'
                            GROUP BY ASPECT
                            HAVING ASPECT LIKE '%CODE GENERATION%'

                            UNION All

                            SELECT 'COPILOT FOR MICROSOFT 365' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'

                            UNION All

                            SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                            FROM Copilot_Sentiment_Data
                            WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'
                            GROUP BY ASPECT
                            HAVING ASPECT LIKE '%CODE GENERATION%'
                    
                    
                VERY VERY IMPORTANT : Do not use Order in this comparision of two devices.
                VERY VERY IMPORTANT : Do not use Order in this comparision of two devices.
                    
        8. IMPORTANT : USE UNION ALL Everytime instead of UNION
                    
                8.1  If the user question is : Compare "Interface" feature of CoPilot for Mobile and GitHub CoPilot or "Compare the reviews for Github Copilot and Copilot for Microsoft 365 for Interface"
                   
                    DO NOT respond like :
                       
                       
                        SELECT 'COPILOT FOR MOBILE' AS PRODUCT_FAMILY, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
                        FROM Copilot_Sentiment_Data 
                        WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MOBILE%' 
                        AND ASPECT = 'Interface'

                        UNION ALL

                        SELECT 'GITHUB COPILOT' AS PRODUCT_FAMILY, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
                        FROM Copilot_Sentiment_Data WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%' 
                        AND ASPECT = 'Interface'
                        
                 Instead respond like : 
                       
                        SELECT 'COPILOT FOR MOBILE' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
                        FROM Copilot_Sentiment_Data 
                        WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MOBILE%' 
                        AND ASPECT = '%Interface%'

                        UNION ALL

                        SELECT 'GITHUB COPILOT' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
                        FROM Copilot_Sentiment_Data WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%' 
                        AND ASPECT LIKE '%Interface%'
                        
                    IMPORTANT : Do not use Order By here. Donot order by based on review count for comparision case
            
                    CHANGES MADE : USE OF LIKE OPERATOR, ASPECT as alias instead of Product_Family


        9. Generic Queries: 
                
            9.1. Sentiment mark is calculated by sum of Sentiment_Score.
            9.2. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Copilot_Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            9.3. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Copilot_Sentiment_Data
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            9.4. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Copilot_Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            9.5. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Copilot_Sentiment_Data) AS Total_Reviews 
                    FROM Copilot_Sentiment_Data 
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            9.6. If the user asks for net sentiment across any country: example : Net sentiment of Windows Copilot in US geography
                   SELECT ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                   FROM Copilot_Sentiment_Data
                   WHERE Geography LIKE "%US%"
            
            REMEBER TO USE LIKE OPERATOR whenever you use 'where' clause
                     
        10. Points to remember :  
            10.1. Convert numerical outputs to float upto 1 decimal point.
            10.2. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            10.3 Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            10.4 Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            10.5 If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            10.6 Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            10.7 Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            10.8 Important: User can ask question about any categories including Aspects, Geograpgy, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            10.9 Important: You Response should directly starts from SQL query nothing else.
            10.10 Important: Always use LIKE keyword instead of '=' symbol while generating SQL query.
            10.11 Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            10.12 Sort all Quantifiable outcomes based on review count.

        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question,history,vector_store_path="faiss_index_CopilotSample"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Copilot_Sentiment_Data")
        print(SQL_Query)
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err



def get_conversational_chain_aspect_wise_detailed_summary():
    global model, history
    try:
        prompt_template = """
        
        1. Your Job is to analyse the Net Sentiment, Aspect wise sentiment and Key word regarding the different aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        
        Your will receive Aspect wise net sentiment of the Product. you have to concentrate on top 4 Aspects based on ASPECT_RANKING.
        For that top 4 Aspect you will get top 2 keywords for each aspect. You will receive each keywords' contribution and +ve mention % and negative mention %
        You will receive reviews of that devices focused on these aspects and keywords.

        For Each Aspect

        Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that Product. In this case provide why the aspect sentiment is lower than net sentiment.
        Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is higher than net sentiment. 

        IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.

            Your response should be : 

            For Each Aspect 
                    Net Sentiment of the Product and aspect sentiment of that aspect of the Product (Mention Code Generation, Aspect Sentiment) . 
                    Top Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                    Top 2nd Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                       Limit yourself to top 3 keywords and don't mention as top 1, top 2, top 3 and all. Mention them as pointers
                    Overall Summary

            IMPORTANT : Example Template :

            ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below:

            Response : "BOLD ALL THE NUMBERS"

            IMPOPRTANT : Start with : "These are the 4 major aspects users commented about" and mention their review count contributions. These top 4 shold be based on ASPECT_RANKING Column

                           These are the 4 top ranked aspects users commented about - IMPORTANT : These top 4 should be from Aspect Ranking:
                           
                           IMPORTANT : DO NOT CONSIDER GENERIC AS ONE OF THE ASPECTS

                        - Total Review for CoPilot for Mobile Product is 1200
                        - Code Generarion: 4.82% of the reviews mentioned this aspect
                        - Ease of Use: 6% of the reviews mentioned this aspect
                        - Compatibility: 9.71% of the reviews mentioned this aspect
                        - Interface: 7.37% of the reviews mentioned this aspect

                        Code Generation:
                        - The aspect sentiment for price is 52.8%, which is higher than the net sentiment of 38.5%. This indicates that the aspect of price is driving the net sentiment higher for the Vivobook.
                        -  The top keyword for price is "buy" with a contribution of 28.07%. It has a positive percentage of 13.44% and a negative percentage of 4.48%.
                              - Users mentioned that the Vivobook offers good value for the price and is inexpensive.
                        - Another top keyword for price is "price" with a contribution of 26.89%. It has a positive percentage of 23.35% and a negative percentage of 0.24%.
                            - Users praised the affordable price of the Vivobook and mentioned that it is worth the money.

                        Ease of use:
                        - The aspect sentiment for performance is 36.5%, which is lower than the net sentiment of 38.5%. This indicates that the aspect of performance is driving the net sentiment lower for the Vivobook.
                        - The top keyword for performance is "fast" with a contribution of 18.24%. It has a positive percentage of 16.76% and a neutral percentage of 1.47%.
                            - Users mentioned that the Vivobook is fast and offers good speed.
                        - Another top keyword for performance is "speed" with a contribution of 12.06%. It has a positive percentage of 9.12% and a negative percentage of 2.06%.
                            - Users praised the speed of the Vivobook and mentioned that it is efficient.


                        lIKE THE ABOVE ONE EXPLAIN OTHER 2 ASPECTS

                        Overall Summary:
                        The net sentiment for the Vivobook is 38.5%, while the aspect sentiment for price is 52.8%, performance is 36.5%, software is 32.2%, and design is 61.9%. This indicates that the aspects of price and design are driving the net sentiment higher, while the aspects of performance and software are driving the net sentiment lower for the Vivobook. Users mentioned that the Vivobook offers good value for the price, is fast and efficient in performance, easy to set up and use in terms of software, and has a sleek and high-quality design.

                        Some Pros and Cons of the device, 


           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.

           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick Summary of whether the aspect is driving the sentiment high or low
               4. Top Keyword: "Usable" (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the usable experience on the Cobilot for Mobile, with many mentioning the smooth usage and easy to use
                    - Some users have reported experiencing lag while not very great to use, but overall, the gaming Ease of use is highly rated.

                Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer

                5. IMPORTANT : Pros and Cons in pointers (overall, not related to any aspect)
                6. Overall Summary
                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.""" + history + """\n
            
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_aspect_wise_detailed_summary(user_question,vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_aspect_wise_detailed_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
        
def get_conversational_chain_aspect_wise_detailed_summary():
    global model, history
    try:
        prompt_template = """
        
        1. Your Job is to analyse the Net Sentiment, Aspect wise sentiment and Key word regarding the different aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        
        Your will receive Aspect wise net sentiment of the Product. you have to concentrate on top 4 Aspects based on ASPECT_RANKING.
        For that top 4 Aspect you will get top 2 keywords for each aspect. You will receive each keywords' contribution and +ve mention % and negative mention %
        You will receive reviews of that devices focused on these aspects and keywords.

        For Each Aspect

        Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that Product. In this case provide why the aspect sentiment is lower than net sentiment.
        Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is higher than net sentiment. 

        IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.

            Your response should be : 

            For Each Aspect 
                    Net Sentiment of the Product and aspect sentiment of that aspect of the Product (Mention Code Generation, Aspect Sentiment) . 
                    Top Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                    Top 2nd Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                       Limit yourself to top 3 keywords and don't mention as top 1, top 2, top 3 and all. Mention them as pointers
                    Overall Summary

            IMPORTANT : Example Template :

            ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below:

            Response : "BOLD ALL THE NUMBERS"

            IMPOPRTANT : Start with : "These are the 4 major aspects users commented about" and mention their review count contributions. These top 4 shold be based on ASPECT_RANKING Column

                           These are the 4 top ranked aspects users commented about - IMPORTANT : These top 4 should be from Aspect Ranking:
                           
                           IMPORTANT : DO NOT CONSIDER GENERIC AS ONE OF THE ASPECTS

                        - Total Review for CoPilot for Mobile Product is 1200
                        - Code Generarion: 4.82% of the reviews mentioned this aspect
                        - Ease of Use: 6% of the reviews mentioned this aspect
                        - Compatibility: 9.71% of the reviews mentioned this aspect
                        - Interface: 7.37% of the reviews mentioned this aspect

                        Code Generation:
                        - The aspect sentiment for price is 52.8%, which is higher than the net sentiment of 38.5%. This indicates that the aspect of price is driving the net sentiment higher for the Vivobook.
                        -  The top keyword for price is "buy" with a contribution of 28.07%. It has a positive percentage of 13.44% and a negative percentage of 4.48%.
                              - Users mentioned that the Vivobook offers good value for the price and is inexpensive.
                        - Another top keyword for price is "price" with a contribution of 26.89%. It has a positive percentage of 23.35% and a negative percentage of 0.24%.
                            - Users praised the affordable price of the Vivobook and mentioned that it is worth the money.

                        Ease of use:
                        - The aspect sentiment for performance is 36.5%, which is lower than the net sentiment of 38.5%. This indicates that the aspect of performance is driving the net sentiment lower for the Vivobook.
                        - The top keyword for performance is "fast" with a contribution of 18.24%. It has a positive percentage of 16.76% and a neutral percentage of 1.47%.
                            - Users mentioned that the Vivobook is fast and offers good speed.
                        - Another top keyword for performance is "speed" with a contribution of 12.06%. It has a positive percentage of 9.12% and a negative percentage of 2.06%.
                            - Users praised the speed of the Vivobook and mentioned that it is efficient.


                        lIKE THE ABOVE ONE EXPLAIN OTHER 2 ASPECTS

                        Overall Summary:
                        The net sentiment for the Vivobook is 38.5%, while the aspect sentiment for price is 52.8%, performance is 36.5%, software is 32.2%, and design is 61.9%. This indicates that the aspects of price and design are driving the net sentiment higher, while the aspects of performance and software are driving the net sentiment lower for the Vivobook. Users mentioned that the Vivobook offers good value for the price, is fast and efficient in performance, easy to set up and use in terms of software, and has a sleek and high-quality design.

                        Some Pros and Cons of the device, 


           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.

           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick Summary of whether the aspect is driving the sentiment high or low
               4. Top Keyword: "Usable" (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the usable experience on the Cobilot for Mobile, with many mentioning the smooth usage and easy to use
                    - Some users have reported experiencing lag while not very great to use, but overall, the gaming Ease of use is highly rated.

                Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer

                5. IMPORTANT : Pros and Cons in pointers (overall, not related to any aspect)
                6. Overall Summary
                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.\n Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
          Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_aspect_wise_detailed_summary(user_question,vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_aspect_wise_detailed_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err

#----------------------------------------------------------------------------------------------------------------------Visualization------------------------------------------------------------------------#

def get_conversational_chain_quant_classify2():
    global model
    try:
        prompt_template = """

                You are an AI Chatbot assistant. Understand the user question carefully and follow all the instructions mentioned below.
                    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
                    2. There is only one table with table name Copilot_Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                        Review: Review of the Copilot Product
                        Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                        Geography: From which Country or Region the review was given. It contains different Geography.
                                   list of Geographies in the table - Values in this column [China,France,Japan,US,Brazil,Canada,Germany,India,Mexico,UK,Australia,Unknown,Venezuela,Vietnam,Cuba,Colombia,Iran,Ukraine,Northern Mariana Islands,Uruguay,Taiwan,Spain,Russia,Bolivia,Argentina,Lebanon,Finland,Saudi Arabia,Oman,United Arab Emirates,Austria,Luxembourg,Macedonia,Puerto Rico,Bulgaria,Qatar,Belgium,Italy,Switzerland,Peru,Czech Republic,Thailand,Greece,Netherlands,Romania,Indonesia,Benin,Sweden,South Korea,Poland,Portugal,Tonga,Norway,Denmark,Samoa,Ireland,Turkey,Ecuador,Guernsey,Botswana,Kenya,Chad,Bangladesh,Nigeria,Singapore,Malaysia,Malawi,Georgia,Hong Kong,Philippines,South Africa,Jordan,New Zealand,Pakistan,Nepal,Jamaica,Egypt,Macao,Bahrain,Tanzania,Zimbabwe,Serbia,Estonia,Jersey,Afghanistan,Kuwait,Tunisia,Israel,Slovakia,Panama,British Indian Ocean Territory,Comoros,Kazakhstan,Maldives,Kosovo,Ghana,Costa Rica,Belarus,Sri Lanka,Cameroon,San Marino,Antigua and Barbuda]
                        Title: What is the title of the review
                        Review_Date: The date on which the review was posted
                        Product: Corresponding product for the review. It contains following values: "COPILOT"
                        Product_Family: Which version or type of the corresponding Product was the review posted for. Different Product Names  - It contains following Values - [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
                        Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                        Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.
                        Keyword: What are the keywords mentioned in the product
                        Review_Count - It will be 1 for each review or each row
                        Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                        
                IMPORTANT : User won't exactly mention the exact Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
                            Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                                We know that Chinanews in not any of the DataSource, Geography and so on.
                                So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues when we pull SQL Queries
                            
                            Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                                We know that USA in not any of the Geography, Data Source and so on.
                                So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                                
                                Same goes for all the columns

                    1. If the user asks for count of column 'X', the query should be like this:
                            SELECT COUNT('X') 
                            FROM Copilot_Sentiment_Data
                    2. If the user asks for count of column 'X' for different values of column 'Y', the query should be like this:
                            SELECT 'Y', COUNT('X') AS Total_Count
                            FROM Copilot_Sentiment_Data 
                            GROUP BY 'Y'
                            ORDER BY TOTAL_COUNT DESC
                    3. If the user asks for Net overall sentiment the query should be like this:
                            SELECT ((SUM(Sentiment_Score))/(SUM(Review_Count))) * 100 AS Net_Sentiment,  SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            ORDER BY Net_Sentiment DESC
                    

                    4. If the user asks for Net Sentiment for column "X", the query should be exactly like this: 

                            SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                            FROM Copilot_Sentiment_Data
                            GROUP BY X
                            ORDER BY Review_Count DESC


                    5. If the user asks for overall review count, the query should be like this:
                            SELECT SUM(Review_Count) 
                            FROM Copilot_Sentiment_Data
                    6. If the user asks for review distribution across column 'X', the query should be like this:
                            SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Copilot_Sentiment_Data) AS Review_Distribution
                            FROM Copilot_Sentiment_Data 
                            GROUP BY 'X'
                            ORDER BY Review_Distribution DESC
                    7. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                            SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Copilot_Sentiment_Data) AS Distribution_PCT
                            FROM Copilot_Sentiment_Data 
                            GROUP BY 'Y'
                            ORDER BY Distribution_PCT DESC
                    8. If the user asks for net sentiment across any country: example : Net sentiment of Windows Copilot in US geography
                               SELECT ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                               FROM Copilot_Sentiment_Data
                               WHERE Geography LIKE "%US%"
                               
                   9. IMPORTANT NOTE :
                   
                        THIS IS THE ONLY WAY TO CALCULATE NET SENTIMENT : ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100
                        
                    Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Copilot_Sentiment_Data
                    ORDER BY Net_Sentiment DESC

                    Important: While generating SQL query to calculate net_sentiment across column 'X' and 'Y', if 'Y' has less distinct values, keep your response like this - SELECT 'Y','X', ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count FROM Copilot_Sentiment_Data GROUP BY 'Y','X'
                    
                    Important: Always replace '=' operator with LIKE keyword and add '%' before and after filter value for single or multiple WHERE conditions in the generated SQL query . For example, if the query is like - 'SELCT * FROM Copilot_Sentiment_Data WHERE PRODUCT='ABC' AND GEOGRAPHY='US' ORDER BY Review_Count' , you should modify the query and share the output like this - 'SELCT * FROM Copilot_Sentiment_Data WHERE PRODUCT LIKE '%ABC%' AND GEOGRAPHY LIKE '%US%' ORDER BY Review_Count'

                    Important: Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
                    Important: Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
                    Important: You Response should directly start from SQL query nothing else.
                    Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
                    
                    Enhance the model’s comprehension to accurately interpret user queries by:
                      Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
                      Understanding product family names even when written in reverse order or missing connecting words such as HP Laptop 15, Lenovo Legion 5 15 etc
                      Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
                      Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
                      Generate acurate response only, do not provide extra information.

                Context:\n {context}?\n
                Question: \n{question}\n

                Answer:
                """
         
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

def query_quant_classify2(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_classify2()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        # st.write(SQL_Query)
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Copilot_Sentiment_Data")
        print(SQL_Query)
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err


def get_conversational_chain_detailed_summary():
    global model, history
    try:
        prompt_template = f"""
            Important: You are provided with an input dataset, also you know that the Overall Net Sentiment for all reviews is {overall_net_sentiment} and total reviews are {overall_review_count}.
            Your Job is to analyse the Net Sentiment, Geo-Wise wise sentiment of particular product or Product-wise sentiment and summarize the reviews that user asks, utilizing the reviews and numbers you get from the input data. Ensure maximum utility of the numbers and justify them using the reviews.
            For example, if the data you receive is Geography wise net sentiment data for a particular product-
            First give an overall summary of the data like, from which Geography most of the reviews are and which geographies have the most and least net sentiment, etc. Then, with the help of the reviews, summarize reviews from each geography and provide Pros and Cons about that Product in each Geography. 

            Your response should follow the below templates, based on input data:
                1. Geography wise summary for a particular product -
                    Based on the provided sentiment data for Github CoPilot reviews from different geographies, here is a summary:
     
                    - Total Sentiment: The overall net sentiment for Github CoPilot is 6.9, based on a total of 3,735 reviews.
     
                    - 1st Geography: The net sentiment for reviews with unknown geography is 5.2, based on 2,212 reviews. ('Driving Net sentiment High' if net sentiment for this Geography is greater than {overall_net_sentiment}, else 'Driving Net sentiment Low')
     
                        Provide Overall summary of the Product from that Geography in 5 to 6 lines
                        Give Some Pros and Cons of the Product from the reviews from this Geography
     
                    - 2nd Geography: The net sentiment for reviews from the United States is 8.1, based on 1,358 reviews.  ('Driving Net sentiment High' if net sentiment for this Geography is greater than {overall_net_sentiment}, else 'Driving Net sentiment Low')
     
                        Provide Overall summary of the Product from that Geography in 5 to 6 lines
                        Give Some Pros and Cons of the Product from the reviews from this Geography
     
                    - 3rd Geography: The net sentiment for reviews from Japan is 20.0, based on 165 reviews.  ('Driving Net sentiment High' if net sentiment for this Geography is greater than {overall_net_sentiment}, else 'Driving Net sentiment Low')
     
                        Overall summary of the Product from that Geography in 5 to 6 lines
                        Give Some Pros and Cons of the Product from the reviews from this Geography

                2. Product Family wise summary -

                    Based on the provided sentiment data for different Product Families, here is a summary:
     
                    - Total Sentiment: The overall net sentiment for all the reviews is {overall_net_sentiment}, based on a total of {overall_review_count} reviews.
     
                    - Copilot for Mobile: The net sentiment for reviews of Copilot for Mobile is 29.5, based on 18,559 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Copilot for Mobile: Users have highly positive reviews for Copilot for Mobile, praising its functionality and ease of use. They find it extremely helpful in their mobile development tasks and appreciate the regular updates and additions to the toolkit.
     
                    - Copilot: The net sentiment for reviews of Copilot is -8.0, based on 10,747 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Copilot: Reviews for Copilot are mostly negative, with users expressing dissatisfaction with its performance and suggesting improvements. They mention issues with suggestions and accuracy, leading to frustration and disappointment.
     
                    - Copilot in Windows 11: The net sentiment for reviews of Copilot in Windows 11 is 8.3, based on 6,107 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Copilot in Windows 11: Users have positive reviews for Copilot in Windows 11, highlighting its compatibility and ease of use. They find it helpful in their development tasks and appreciate the integration with the Windows 11 operating system.
     
                    - Copilot Pro: The net sentiment for reviews of Copilot Pro is 12.7, based on 5,075 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Copilot Pro: Users have highly positive reviews for Copilot Pro, praising its advanced features and capabilities. They find it valuable for their professional development tasks and appreciate the additional functionalities offered in the Pro version.
     
                    - Github Copilot: The net sentiment for reviews of Github Copilot is 6.9, based on 3,735 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Github Copilot: Users have generally positive reviews for Github Copilot, mentioning its usefulness in their coding tasks. They appreciate the AI-powered suggestions and find it helpful in improving their productivity.
     
                    - Microsoft Copilot: The net sentiment for reviews of Microsoft Copilot is -2.4, based on 2,636 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Microsoft Copilot: Reviews for Microsoft Copilot are mostly negative, with users expressing dissatisfaction with its performance and suggesting improvements. They mention issues with accuracy and compatibility, leading to frustration and disappointment.
     
                    - Copilot for Security: The net sentiment for reviews of Copilot for Security is 9.4, based on 2,038 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Copilot for Security: Users have positive reviews for Copilot for Security, mentioning its effectiveness in enhancing security measures. They find it valuable for protecting sensitive information and appreciate the various customization options offered.
     
                    Copilot for Microsoft 365: The net sentiment for reviews of Copilot for Microsoft 365 is 4.0, based on 2,031 reviews. (Mention 'Driving Net sentiment High' if net sentiment for this Product Family is greater than {overall_net_sentiment}, else mention 'Driving Net sentiment Low')
     
                       Overall summary of Copilot for Microsoft 365: Reviews for Copilot for Microsoft 365 are mostly neutral, with users expressing mixed opinions about its functionality. Some find it helpful in their Microsoft 365 tasks, while others mention limitations and suggest improvements.
     
                    Based on the sentiment data, it can be observed that Copilot for Mobile, Copilot in Windows 11, Copilot Pro, and Copilot for Security have higher net sentiments, indicating positive user experiences. On the other hand, Copilot, Microsoft Copilot, and Copilot for Microsoft 365 have lower net sentiments, indicating negative or mixed user experiences.

            Important: Modify the Geography, Product Family or Product names in the prompt as per given dataset values            
            Important: Enhance the model’s comprehension to accurately interpret user queries by:
              - Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
              - Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
              - Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references]
             Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
                    Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
            Context:\n {context}?\n
            Question: \n{question}\n
     
            Answer:
            """
            
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_summary(dataframe_as_dict,user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = generate_chart_insight_llm(dataframe_as_dict)
        return err
        
def generate_chart_insight_llm(user_question):
    global model, history
    try:
        prompt_template = """
        1.Based on the data available in the input, generate meaningful insights using the numbers and summarize them. 
        2.Ensure to include all possible insights and findings that can be extracted, which reveals vital trends and patterns in the data. 
        3.Share the findings or insights in a format which makes more sense to business oriented users, and can generate vital action items for them. 
        4.If any recommendations are possible based on the insights, share them as well - primarily focusing on the areas of concern.
        5.For values like Net_Sentiment score, positive values indicate positive overall sentiment, negative values indicate negative overall sentiment and 0 value indicate neutral overall sentiment. For generating insights around net_sentiment feature, consider this information.
        IMPORTANT: If the maximum numerical value is less than or equal to 100, then the numerical column is indicating percentage results - therefore while referring to numbers in your insights, add % at the end of the number.
        IMPORTANT : Use the data from the input only and do not give information from pre-trained data.
        IMPORTANT : Dont provide any prompt message written here in the response, this is for your understanding purpose \n Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        #st.write("\n\n",response["output_text"])
        return response["output_text"]
            
    except Exception as e:
        err = "Apologies, unable to generate insights based on the provided input data. Kindly refine your search query and try again!"
        return err


def quantifiable_data(user_question):
    try:
        #st.write("correct_func")
        response = query_quant_classify2(user_question)
        
        return response
    except Exception as e:
        err = f"An error occurred while generating quantitative review summarization: {e}"
        return err
        
def generate_chart(df):
    global full_response
    # Determine the data types of the columns
    try:
        df=df.drop('Impact',axis=1)
    except:
        pass
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    
    if len(num_cols)>0:
        for i in range(len(num_cols)):
            df[num_cols[i]]=round(df[num_cols[i]],1)
            
    if len(df.columns)>3:
        try:
            cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
            df.drop(columns=cols_to_drop, inplace=True)
        except:
            pass
        
        df=df.iloc[:, :3]
        
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    #st.write(num_cols,cat_cols,len(num_cols),len(cat_cols))
    # Simple heuristic to determine the most suitable chart
    if len(df.columns)==2:
        
        if len(num_cols) == 1 and len(cat_cols) == 0:

            plt.figure(figsize=(10, 6))
            sns.histplot(df[num_cols[0]], kde=True)
            plt.title(f"Frequency Distribution of '{num_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        elif len(num_cols) == 2:
   
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        elif len(cat_cols) == 1 and len(num_cols) == 1:
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                st.plotly_chart(fig)
                # try:
                    # chart = fig.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")

            else:
                num_categories=df[cat_cols[0]].nunique()
                width = 800
                height = max(600,num_categories*50)
                
                bar=px.bar(df,x=num_cols[0],y=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'",text=num_cols[0])
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                st.plotly_chart(bar)
                # try:
                    # chart = bar.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")


        elif len(cat_cols) == 2:

            plt.figure(figsize=(10, 6))
            sns.countplot(x=df[cat_cols[0]], hue=df[cat_cols[1]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        elif len(date_cols) == 1 and len(num_cols) == 1:
   
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=df[date_cols[0]], y=df[num_cols[0]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)
            # try:
                # chart = plt.to_html()
                # full_response += chart
            # except:
                # st.write("Error in converting chart to html")


        else:
            sns.pairplot(df)
            st.pyplot(plt)
            
    
            
    elif len(df.columns)==3 and len(cat_cols)>=1:
        
        col_types = df.dtypes

#         cat_col = None
#         num_cols = []

#         for col in df.columns:
#             if col_types[col] == 'object' and df[col].nunique() == len(df):
#                 categorical_col = col
#             elif col_types[col] in ['int64', 'float64']:
#                 num_cols.append(col)
#         st.write(cat_cols,num_cols,len(cat_cols),len(num_cols))
#         st.write(type(cat_cols))
        # Check if we have one categorical and two numerical columns
        if len(cat_cols)==1 and len(num_cols) == 2:
#             df[cat_cols[0]]=df[cat_cols[0]].astype(str)
#             df[cat_cols[0]]=df[cat_cols[0]].fillna('NA')
            
            
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                fig2 = px.pie(df, names=cat_cols[0], values=num_cols[1], title=f"Distribution of '{num_cols[1]}' across '{cat_cols[0]}'")
                st.plotly_chart(fig)
                st.plotly_chart(fig2)
                # try:
                    # chart = fig.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                # try:
                    # chart = fig2.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                

            else:
                num_categories=df[cat_cols[0]].nunique()
                width = 800
                height = max(600,num_categories*50)
                
                bar=px.bar(df,x=num_cols[0],y=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'",text=num_cols[0])
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                st.plotly_chart(bar)
                
                bar2=px.bar(df,x=num_cols[1],y=cat_cols[0],title=f"Distribution of '{num_cols[1]}' across '{cat_cols[0]}'",text=num_cols[1])
                bar2.update_traces(textposition='outside', textfont_size=12)
                bar2.update_layout(width=width, height=height)
                st.plotly_chart(bar2)
                # try:
                    # chart = bar.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                # try:
                    # chart = bar2.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")
                
        elif len(cat_cols)==2 and len(num_cols) == 1:
            df[cat_cols[0]]=df[cat_cols[0]].astype(str)
            df[cat_cols[1]]=df[cat_cols[1]].astype(str)
            df[cat_cols[0]]=df[cat_cols[0]].fillna('NA')
            df[cat_cols[1]]=df[cat_cols[1]].fillna('NA')
            
            list_cat=df[cat_cols[0]].unique()
            st.write("\n\n")
            for i in list_cat:
                st.markdown(f"* {i} OVERVIEW *")
                df_fltr=df[df[cat_cols[0]]==i]
                df_fltr=df_fltr.drop(cat_cols[0],axis=1)
                num_categories=df_fltr[cat_cols[1]].nunique()
#                 num_categories2=df[cat_cols[1]].nunique()
                height = 600 #max(80,num_categories2*20)
                width=800

                bar=px.bar(df_fltr,x=num_cols[0],y=cat_cols[1],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[1]}'",text=num_cols[0],color=cat_cols[1])
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                st.plotly_chart(bar)
                # try:
                    # chart = bar.to_html()
                    # full_response += chart
                # except:
                    # st.write("Error in converting chart to html")

def generate_chart_insight_llm(user_question):
    global model, history
    try:
        prompt_template = """
        1.Based on the data available in the input, generate meaningful insights using the numbers and summarize them. 
        2.Ensure to include all possible insights and findings that can be extracted, which reveals vital trends and patterns in the data. 
        3.Share the findings or insights in a format which makes more sense to business oriented users, and can generate vital action items for them. 
        4.If any recommendations are possible based on the insights, share them as well - primarily focusing on the areas of concern.
        5.For values like Net_Sentiment score, positive values indicate positive overall sentiment, negative values indicate negative overall sentiment and 0 value indicate neutral overall sentiment. For generating insights around net_sentiment feature, consider this information.
        IMPORTANT: If the maximum numerical value is less than or equal to 100, then the numerical column is indicating percentage results - therefore while referring to numbers in your insights, add % at the end of the number.
        IMPORTANT : Use the data from the input only and do not give information from pre-trained data.
        IMPORTANT : Dont provide any prompt message written here in the response, this is for your understanding purpose \n Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        #st.write("\n\n",response["output_text"])
        return response["output_text"]
            
    except Exception as e:
        err = "Apologies, unable to generate insights based on the provided input data. Kindly refine your search query and try again!"
        return err
        
#------------------------------------------------------------------------------------- Generic  ---------------------------------------------------------------------------------------#

def get_conversational_chain_generic():
    global model, history
    try:
        prompt_template = """
        You are an AI ChatBot where you will get the data of Copilot products and you should generate the response based on that Dataset only for user question by following the below instructions.
        Context Interpretation:
        1. Recognize abbreviations for country names (e.g., 'DE' for Germany, 'USA' for the United States).
        2. Understand product family names even if written in reverse order or with missing words.
        3. If the user asks to summarize attributes of a product (e.g., "Summarize the attributes of Microsoft Copilot Pro in the USA"), relate the attributes to the corresponding dataset columns and provide the response accordingly.
            For example:
            Attributes: Aspect
            Product: Product_Family
            Location: Geography
            Based on these relations, generate the summary using the relevant data from the dataset.

        Data Utilization:
        IMPORTANT: 1. Use only the provided dataset for generating responses.
        IMPORTANT: 2. Do not use or rely on pre-trained information other than Copilot Product Data which is given in Dataset. Limit Yourself to data you are provided with.

        Dataset Columns:
        Review: This column contains the opinions and experiences of users regarding different product families across geographies, providing insights into customer satisfaction or complaints and areas for improvement.
        Data_Source: This column indicates the platform from which the user reviews were collected, such as Reddit, Play Store, App Store, Tech Websites, or YouTube videos.
        Geography: This column lists the countries of the users who provided the reviews, allowing for an analysis of regional preferences and perceptions of the products.
        Product_Family: This column identifies the broader category of products to which the review pertains, enabling comparisons and trend analysis across different product families.
                List of Product_Families : ["Windows Copilot" , "Microsoft Copilot" , "Github Copilot" , "Copilot Pro" , "Copilot for Security" , "Copilot for Mobile", "Copilot for Microsoft 365"]
        Sentiment: This column reflects the overall tone of the review, whether positive, negative, or neutral, and is crucial for gauging customer sentiment.
        Aspect: This column highlights the particular features or attributes of the product that the review discusses, pinpointing areas of strength or concern.

        Tasks:
        1. Review Summarization: Summarize reviews by filtering relevant Aspect, Geography, Product_Family, Sentiment, or Data_Source. Provide insights based on available reviews and sentiments of Copilot Products.
        2. Aspect Comparison: Summarize comparisons for overlapping features between product families or geographies. Highlight key differences with positive and negative sentiments.
        3. New Feature Suggestion/Recommendation: Generate feature suggestions or improvements based on the frequency and sentiment of reviews and mentioned aspects. Provide detailed responses by analyzing review sentiment and specific aspects.
        4. Hypothetical Reviews: Create hypothetical reviews for feature updates or new features, simulating user reactions. Include realistic reviews with all types of sentiments. Provide solutions for negative hypothetical reviews.
        5. Response Criteria: Minimum of 300 words. Provide as much detail as possible. Generate accurate responses without extra information.
        
        Understanding User Queries:
        1. Carefully read and understand the full user's question.
        2. If the question is outside the scope of the dataset, respond with: "Sorry! I do not have sufficient information. Can you provide more details?"
        3. Respond accurately based on the provided Copilot Products Data only.\n Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err
      

def query_detailed_generic(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_generic()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err

#----------------------------------------------------------Split Table--------------------------------------------------------------#

def split_table(data,device_a,device_b):
    # Initialize empty lists for each product
    copilot_index = data[data["ASPECT"] == str(device_b).upper()].index[0]
    if copilot_index != 0:
        device_a_table = data.iloc[:copilot_index]
        device_b_table = data.iloc[copilot_index:]
    else:
        copilot_index = data[data["ASPECT"] == str(device_a).upper()].index[0]
        device_a_table = data.iloc[:copilot_index]
        device_b_table = data.iloc[copilot_index:]

    return device_a_table, device_b_table
    
#-----------------------------------------------------------Miscellaneous----------------------------------------------------------------#

def make_desired_df(data):
    try:
        # Create DataFrame from the dictionary
        df1 = pd.DataFrame(data)
        df = pd.DataFrame(data)
        
        # Ensure the necessary columns are present
        if 'ASPECT_SENTIMENT' not in df.columns or 'REVIEW_COUNT' not in df.columns:
            raise ValueError("Input data must contain 'ASPECT_SENTIMENT' and 'REVIEW_COUNT' columns")
        
        df = df[df['ASPECT_SENTIMENT'] != 0]
        df = df[df['ASPECT'] != 'Generic']
        df = df[df['ASPECT'] != 'TOTAL']

        # Compute min and max values for normalization
        min_sentiment = df['ASPECT_SENTIMENT'].min(skipna=True)
        max_sentiment = df['ASPECT_SENTIMENT'].max(skipna=True)
        min_review_count = df['REVIEW_COUNT'].min(skipna=True)
        max_review_count = df['REVIEW_COUNT'].max(skipna=True)

        # Apply min-max normalization for ASPECT_SENTIMENT
        df['NORMALIZED_SENTIMENT'] = df.apply(
            lambda row: (row['ASPECT_SENTIMENT'] - min_sentiment) / (max_sentiment - min_sentiment)
            if pd.notnull(row['ASPECT_SENTIMENT'])
            else None,
            axis=1
        )

        # Apply min-max normalization for REVIEW_COUNT
        df['NORMALIZED_REVIEW_COUNT'] = df.apply(
            lambda row: (row['REVIEW_COUNT'] - min_review_count) / (max_review_count - min_review_count)
            if pd.notnull(row['REVIEW_COUNT'])
            else None,
            axis=1
        )

        # Calculate the aspect ranking based on normalized values
        weight_for_sentiment = 1
        weight_for_review_count = 3
        
        df['ASPECT_RANKING'] = df.apply(
            lambda row: (weight_for_sentiment * (1 - row['NORMALIZED_SENTIMENT']) + weight_for_review_count * row['NORMALIZED_REVIEW_COUNT'])
            if pd.notnull(row['NORMALIZED_SENTIMENT']) and pd.notnull(row['NORMALIZED_REVIEW_COUNT'])
            else None,
            axis=1
        )
        # Assign integer rankings based on the 'Aspect_Ranking' score
        df['ASPECT_RANKING'] = df['ASPECT_RANKING'].rank(method='max', ascending=False, na_option='bottom').astype('Int64')

        # Sort the DataFrame based on 'Aspect_Ranking' to get the final ranking
        df_sorted = df.sort_values(by='ASPECT_RANKING')
        
        # Extract and display the net sentiment and overall review count
        try:
            total_row = df1[df1['ASPECT'] == 'TOTAL'].iloc[0]
            net_sentiment = str(int(total_row["ASPECT_SENTIMENT"])) + '%'
            overall_review_count = int(total_row["REVIEW_COUNT"])
        except (ValueError, TypeError, IndexError):
            net_sentiment = total_row["ASPECT_SENTIMENT"]
            overall_review_count = total_row["REVIEW_COUNT"]

        st.write(f"Net Sentiment: {net_sentiment}")
        st.write(f"Overall Review Count: {overall_review_count}")
        df_sorted = df_sorted.drop(columns=["NORMALIZED_SENTIMENT", "NORMALIZED_REVIEW_COUNT", "ASPECT_RANKING"])
        return df_sorted
    except Exception as e:
        st.error(f"Error in make_desired_df: {str(e)}")
        return pd.DataFrame()


import numpy as np

def custom_color_gradient(val, vmin=-100, vmax=100):
    green_hex = '#347c47'
    middle_hex = '#dcdcdc'
    lower_hex = '#b0343c'
    
    # Adjust the normalization to set the middle value as 0
    try:
        # Normalize the value to be between -1 and 1 with 0 as the midpoint
        normalized_val = (val - vmin) / (vmax - vmin) * 2 - 1
    except ZeroDivisionError:
        normalized_val = 0
    
    if normalized_val <= 0:
        # Interpolate between lower_hex and middle_hex for values <= 0
        r = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
    else:
        # Interpolate between middle_hex and green_hex for values > 0
        r = int(np.interp(normalized_val, [0, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
    # Convert interpolated RGB values to hex format for CSS color styling
    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
    return f'background-color: {hex_color}; color: black;'


# In[11]:


def get_final_df(aspects_list,device):
    final_df = pd.DataFrame()
    device = device
    aspects_list = aspects_list

    # Iterate over each aspect and execute the query
    for aspect in aspects_list:
        # Construct the SQL query for the current aspect
        query = f"""
        SELECT Keywords,
               COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
               COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
               COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
               COUNT(*) as Total_Count
        FROM Copilot_Sentiment_Data
        WHERE Aspect = '{aspect}' AND Product_Family LIKE '%{device}%'
        GROUP BY Keywords
        ORDER BY Total_Count DESC;
        """

        # Execute the query and get the result in 'key_df'
        key_df = ps.sqldf(query, globals())

        # Calculate percentages and keyword contribution
        total_aspect_count = key_df['Total_Count'].sum()
        key_df['Positive_Percentage'] = (key_df['Positive_Count'] / total_aspect_count) * 100
        key_df['Negative_Percentage'] = (key_df['Negative_Count'] / total_aspect_count) * 100
        key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / total_aspect_count) * 100
        key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100

        # Drop the count columns
        key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)

        # Add the current aspect to the DataFrame
        key_df['Aspect'] = aspect

        # Sort by 'Keyword_Contribution' and select the top 2 for the current aspect
        key_df = key_df.sort_values(by='Keyword_Contribution', ascending=False).head(2)

        # Append the results to the final DataFrame
        final_df = pd.concat([final_df, key_df], ignore_index=True)
        
    return final_df
    
#-----------------------------------------------Classify Flow---------------------------------------------------#

def classify(user_question):
    global model
    try:
        prompt_template = """
        
            Given an input, classify it into one of two categories:
            
            ProductFamilies = Microsoft Copilot, Copilot in Windows 11, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile
            
            1stFlow: [Summarization of any Product Family or Product Family Name]. This flow is just for summarization of reviews of only one product/product Family name.
                    Choose 1st flow, if the user seeks for summarization of only one product choose this flow.
                    Eg: "Summarize reviews of copilot mobile in USA", "Summarize reviews of copilot mobile in USA"
                        
            
            
            2ndFlow: User is seeking any other information like geography wise performance or any quantitative numbers like what is net sentiment for different product families then categorize as 2ndFlow. It should even choose 2nd flow, if it asks for Aspect wise sentiment of one Product.
            
            Example - Geography wise how products are performing or seeking for information across different product families/products.
            What is net sentiment for any particular product/geography
            
        IMPORTANT : Only share the classified category name, no other extra words.
        IMPORTANT : Don't categorize into 1stFlow or 2ndFlow based on number of products, categorize based on the type of question the user is asking
        Input: User Question
        Output: Category (1stFlow or 2ndFlow)
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        if "1stflow" in response["output_text"].lower():
            return "1"
        elif "2ndflow" in response["output_text"].lower():
            return "2"
        else:
            return "Others"+"\nPrompt Identified as:"+response["output_text"]+"\n"
    except Exception as e:
        err = f"An error occurred while generating conversation chain for identifying nature of prompt: {e}"
        return err
        

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("672370cd6ca440f2a0327351d4f4d2bf"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("https://hulk-openai.openai.azure.com/")
    )
    
deployment_name='SurfaceGenAI'

context_Prompt = """


You are an AI Chatbot assistant. Carefully understand the user's question and follow these instructions to categorize their query into one of four features.

Features:
    Summarization, Quantifiable and Visualization, Comparison, Generic

Instructions:


-There is only one table with table name Copilot_Sentiment_Data where each row is a user review, using that data we developed these functionalities. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                Geography: From which Country or Region the review was given. It contains different Geography.
                           list of Geographies in the table - Values in this column [China,France,Japan,US,Brazil,Canada,Germany,India,Mexico,UK,Australia,Unknown,Venezuela,Vietnam,Cuba,Colombia,Iran,Ukraine,Northern Mariana Islands,Uruguay,Taiwan,Spain,Russia,Bolivia,Argentina,Lebanon,Finland,Saudi Arabia,Oman,United Arab Emirates,Austria,Luxembourg,Macedonia,Puerto Rico,Bulgaria,Qatar,Belgium,Italy,Switzerland,Peru,Czech Republic,Thailand,Greece,Netherlands,Romania,Indonesia,Benin,Sweden,South Korea,Poland,Portugal,Tonga,Norway,Denmark,Samoa,Ireland,Turkey,Ecuador,Guernsey,Botswana,Kenya,Chad,Bangladesh,Nigeria,Singapore,Malaysia,Malawi,Georgia,Hong Kong,Philippines,South Africa,Jordan,New Zealand,Pakistan,Nepal,Jamaica,Egypt,Macao,Bahrain,Tanzania,Zimbabwe,Serbia,Estonia,Jersey,Afghanistan,Kuwait,Tunisia,Israel,Slovakia,Panama,British Indian Ocean Territory,Comoros,Kazakhstan,Maldives,Kosovo,Ghana,Costa Rica,Belarus,Sri Lanka,Cameroon,San Marino,Antigua and Barbuda]
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "COPILOT"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names  - It contains following Values - [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.

        IMPORTANT : User won't exactly mention the Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context and then provide SQL Query.
        Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                We know that Chinanews in not any of the DataSource, Geography and so on.
                So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues in understanding

        Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                We know that USA in not any of the Geography, Data Source and so on.
                So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding

Summarization:
        -Summarizes reviews for a specific Product Family.
        -Choose this if the user asks for a summary of aspects for one Product Family.
        -Do not choose this for general pros and cons or top verbatims.
        -If user mention 2 devices for summarization, Go with comparision
        -If user mention 3 devices for summarization, Go with Generic
        -Don't select this, if the question ask to summarize reviews for a feature.
Quantifiable and Visualization:
       - Provides data retrieval and visualization for any Product Family.
       - Choose this for queries related to net sentiment, aspect sentiment, and review count.
       - Examples:
            "What is the net sentiment of [Product Family]?"
            "Give me the aspect sentiment of [Product Family]."
            "Which Product Family has the highest review count?" 
       - Do not choose this for general questions.
       - Whenver user asks about top 20 aspects, keywords choose this function
       
Comparison:

        - Compares two different Product Families based on user reviews.
        - Choose this if exactly two Product Families are mentioned. 
       Examples:
        - "Compare [Product Family 1] and [Product Family 2] on performance."
        - Do not choose Comparison if more than two Product Families are mentioned.

Generic:

        -For general questions about any Product Family.
        -Choose this for queries about pros and cons, common complaints, and top verbatims.
        -Also, choose this if the question involves more than two Product Families.
        -Choose this, if the question ask to summarize reviews for a feature.
        -Examples:
        -"What do people think about the pricing of [Product Family] in the US?"
        -"Compare the interface of all Copilot Products."


Important Notes:

    -Do not choose Comparison if more than two Product Families are mentioned.
    -Do not choose Quantifiable and Visualization for general questions.
    -Generic should be chosen for any query not specific to net sentiment, aspect sentiment, or comparing exactly two Product Families.

Your response should be one of the following:

“Summarization”
“Quantifiable and Visualization”
“Comparison”
“Generic”
"""

def classify_prompts(user_question):
    global context_Prompt
    # Append the new question to the context
    full_prompt = context_Prompt + "\nQuestion:\n" + user_question + "\nAnswer:"
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    user_query = response.choices[0].text.strip()
    
    # Update context with the latest interaction
    context_Prompt += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    
    return user_query   
        
        
#-----------------------------------------------------------------------------------------Comparision------------------------------------------------------------------------------------#


def get_conversational_chain_detailed_compare():
    global model, history
    try:
        prompt_template = """
        
            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.
            
        Product = Microsoft Copilot, Copilot in Windows 11, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile

        
        1. Your Job is to Summarize the user reviews and sentiment data you get as an input for 2 different Product that user mentioned.
        
        IMPORTANT : Mention their Positive and Negative of each Product for each aspects (What consumer feels) for each aspect.
        
        Example :
        
        Summary of CoPilot for Mobile and Github Copilot:
        

        Positive:

        CoPilot for Mobile has a high sentiment score for aspects such as Productivity, Ease of Use, and Accessibility.
        Users find it helpful for various tasks and appreciate its quick and informative responses.
        The app is praised for its usefulness in both work and everyday life.
        Negative:

        Some users have reported issues with Connectivity and Reliability, mentioning network problems and automatic closing of the app.
        There are concerns about Security/Privacy, with users mentioning the potential for data misuse.
        Compatibility with certain devices and interfaces is also mentioned as an area for improvement.
        Summary of GitHub CoPilot:

        Positive:

        GitHub CoPilot receives positive sentiment for aspects such as Interface and Innovation.
        Users appreciate its code generation capabilities and find it helpful for their programming tasks.
        The app is praised for its accuracy and ability to provide quick and relevant responses.
        Negative:

        Some users have reported issues with Reliability and Compatibility, mentioning problems with generating images and recognizing certain commands.
        There are concerns about Security/Privacy, with users mentioning the potential for data misuse.
        Users also mention the need for improvements in the app's interface and connectivity.
        Overall, both CoPilot for Mobile and GitHub CoPilot have received positive feedback for their productivity and code generation capabilities. However, there are areas for improvement such as connectivity, reliability, compatibility, and security/privacy. Users appreciate the ease of use and quick responses provided by both apps.
     
        
        IMPORTANT : If user asks to compare any specific aspects of two device, Give detailed summary like how much reviews is being spoken that aspect in each device, net sentiment and theire Pros and cons on that device (Very Detailed).
        
            Summary of Code Generation feature for CoPilot for Mobile:

                    Positive:

                    Users have praised the Code Generation feature of CoPilot for Mobile, with a high sentiment score of 8.5.
                    The feature is described as helpful and efficient in generating code that aligns with project standards and practices.
                    Users appreciate the convenience and time-saving aspect of the Code Generation feature.
                    Negative:

                    No negative reviews or concerns were mentioned specifically for the Code Generation feature of CoPilot for Mobile.
                    Summary of Code Generation feature for GitHub CoPilot:

                    Positive:

                    Users have a positive sentiment towards the Code Generation feature of GitHub CoPilot, with a sentiment score of 5.4.
                    The feature is described as a game-changer for developer productivity.
                    Users appreciate the ability of GitHub CoPilot to generate code that aligns with project standards and practices.
                    Negative:

                    No negative reviews or concerns were mentioned specifically for the Code Generation feature of GitHub CoPilot.
                    Overall, both CoPilot for Mobile and GitHub CoPilot have received positive feedback for their Code Generation capabilities. Users find the feature helpful, efficient, and a game-changer for developer productivity. No negative reviews or concerns were mentioned for the Code Generation feature of either product.
        
        Give a detailed summary for each aspects using the reviews. Use maximum use of the reviews. Do not use your pretrained data. Use the data provided to you. For each aspects. Summary should be 3 ro 4 lines

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + str(history) + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_compare(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_compare()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
        
def check_history_length(a,last_response):
    if len(a) > 3:
        a.pop(0)
    else:
        a.append(last_response)
    return a
        
#------------------------------------------------------Rephrase Prompts------------------------------------------------#

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("672370cd6ca440f2a0327351d4f4d2bf"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("https://hulk-openai.openai.azure.com/")
    )
    
deployment_name='SurfaceGenAI'

rephrase_Prompt = """


You are an AI Chatbot assistant. Carefully understand the user's question and follow these instructions.

Your Job is to correct the user Question if user have misspelled any device names, Geographies, DataSource names... etc
Device Family Names, Geographies and DataSources names are clearly mentioned below

Below are the available column names and values from the Copilot sentiment data: 

-There is only one table with table name Copilot_Sentiment_Data where each row is a user review, using that data we developed these functionalities. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers - It contains following values : [chinatechnews, DigitalTrends, Engadget, clubic, g2.com, gartner, JP-ASCII, Jp-Impresswatch, Itmedia, LaptopMag, NotebookCheck, PCMag, TechAdvisor, TechRadar, TomsHardware, TechCrunch, Verge, ZDNET, PlayStore, App Store, AppStore, Reddit, YouTube, Facebook, Instagram, X, VK, Forums, News, Print, Blogs/Websites, Reviews, Wordpress, Podcast, TV, Quora, LinkedIn, Videos]
                Geography: From which Country or Region the review was given. It contains different Geography.
                           list of Geographies in the table - Values in this column [China,France,Japan,US,Brazil,Canada,Germany,India,Mexico,UK,Australia,Unknown,Venezuela,Vietnam,Cuba,Colombia,Iran,Ukraine,Northern Mariana Islands,Uruguay,Taiwan,Spain,Russia,Bolivia,Argentina,Lebanon,Finland,Saudi Arabia,Oman,United Arab Emirates,Austria,Luxembourg,Macedonia,Puerto Rico,Bulgaria,Qatar,Belgium,Italy,Switzerland,Peru,Czech Republic,Thailand,Greece,Netherlands,Romania,Indonesia,Benin,Sweden,South Korea,Poland,Portugal,Tonga,Norway,Denmark,Samoa,Ireland,Turkey,Ecuador,Guernsey,Botswana,Kenya,Chad,Bangladesh,Nigeria,Singapore,Malaysia,Malawi,Georgia,Hong Kong,Philippines,South Africa,Jordan,New Zealand,Pakistan,Nepal,Jamaica,Egypt,Macao,Bahrain,Tanzania,Zimbabwe,Serbia,Estonia,Jersey,Afghanistan,Kuwait,Tunisia,Israel,Slovakia,Panama,British Indian Ocean Territory,Comoros,Kazakhstan,Maldives,Kosovo,Ghana,Costa Rica,Belarus,Sri Lanka,Cameroon,San Marino,Antigua and Barbuda]
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "COPILOT"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names  - It contains following Values - [Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile]
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.

        IMPORTANT : User won't exactly mention the Geography Names, Product Names, Product Families, Data Source name, Aspect names. Please make sure to change/correct to the values that you know from the context
        Exmaple : User Question : "Summarize the reviews of Copilot from Chinanews"
                We know that Chinanews in not any of the DataSource, Geography and so on.
                So Change it to "Summarize the reviews of Copilot from Chinatechnews" as this is more relevant and faces no issues in understanding

        Exmaple : User Question : "Summarize the reviews of Copilot from USA"
                We know that USA in not any of the Geography, Data Source and so on.
                So Change it to "Summarize the reviews of Copilot from US" as this is more relevant and faces no issues in understanding
                
                Exmaple : User Question : "Summarize the reviews of mobile copilot from USA"
                We know that USA in not any of the Geography, Data Source and so on. and mobile copilot is not in Product Family, Geography and so on.
                So Change it to "Summarize the reviews of Copilot for Mobile from US"
                
        Mapping file for Products:

        Copilot in Windows 11 -> Windows Copilot
        Copilot for Security -> Copilot for Security
        Copilot Pro -> Copilot Pro
        Microsoft Copilot -> Microsoft Copilot
        Copilot for Microsoft 365 -> Copilot for Microsoft 365
        Github Copilot -> Github Copilot
        Copilot for Mobile -> Copilot for Mobile
        Windows Copilot -> Windows Copilot
        Copilot for Windows -> Windows Copilot
        Copilot Windows -> Windows Copilot
        Win Copilot -> Windows Copilot
        Security Copilot -> Copilot for Security
        Privacy Copilot -> Copilot for Security
        M365 -> Copilot for Microsoft 365
        Microsoft 365 -> Copilot for Microsoft 365
        Office copilot -> Copilot for Microsoft 365
        Github -> Github Copilot
        MS Office -> Copilot for Microsoft 365
        MSOffice -> Copilot for Microsoft 365
        Microsoft Office -> Copilot for Microsoft 365
        Office Product -> Copilot for Microsoft 365
        Mobile -> Copilot for Mobile
        App -> Copilot for Mobile
        ios -> Copilot for Mobile
        apk -> Copilot for Mobile
        Copilot -> Microsoft Copilot

IMPORTANT: If the input sentence mentions a device(Laptop or Desktop) instead of Copilot, keep the device name as it is.

"""

def rephrase_prompt(user_question):
    global rephrase_Prompt
    # Append the new question to the context
    full_prompt = rephrase_Prompt + "\nQuestion:\n" + user_question + "\nAnswer:"
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    user_query = response.choices[0].text.strip()
    
    # Update context with the latest interaction
    rephrase_Prompt += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    
    return user_query   
        
#--------------------------------------------------------Main Function-------------------------------------------------#


def user_ques(user_question_1, user_question, classification):
    global full_response, history
    if user_question_1:
        device_list = Copilot_Sentiment_Data['Product_Family'].to_list()
        sorted_device_list_desc = sorted(device_list, key=lambda x: len(x), reverse=True)

    # Convert user question and product family names to lowercase for case-insensitive comparison
        user_question_lower = user_question_1.lower()

        # Initialize variables for device names
        device_a = None
        device_b = None

        # Search for product family names in the user question
        for device in sorted_device_list_desc:
            if device.lower() in user_question_lower:
                if device_a is None and device != 'Copilot':
                    device_a = device
                else:
                    if device_a != device and device != 'Copilot':
                        device_b = device
                        break# Found both devices, exit the loop

        # st.write(device_a)
        # st.write(device_b)

        if device_a != None and device_b != None:
            try:
                col1,col2 = st.columns(2) 
                data = query_quant(user_question_1,[])
                device_a_table,device_b_table = split_table(data,device_a,device_b)   
                with col1:
                    device_a_table = device_a_table.dropna(subset=['ASPECT_SENTIMENT'])
                    device_a_table = device_a_table[~device_a_table["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
                    device_a_table = device_a_table[device_a_table['ASPECT_SENTIMENT'] != 0]
                    device_a_table = device_a_table[device_a_table['ASPECT'] != 'Generic']
                    device_a_table = device_a_table.sort_values(by='REVIEW_COUNT', ascending=False)
                    styled_df_a = device_a_table.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                    data_filtered = device_a_table[(device_a_table["ASPECT"] != device_a) | (device_a_table["ASPECT"] != device_b) & (device_a_table["ASPECT"] != 'Generic')]
                    top_four_aspects = data_filtered.head(4)
                    c = device_a_table.to_dict(orient='records')
                    st.dataframe(styled_df_a)
                    first_table = styled_df_a.to_html(index = False)
                    full_response += first_table
                    history = check_history_length(history,first_table)

                with col2:

                    device_b_table = device_b_table.dropna(subset=['ASPECT_SENTIMENT'])
                    device_b_table = device_b_table[~device_b_table["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
                    device_b_table = device_b_table[device_b_table['ASPECT_SENTIMENT'] != 0]
                    device_b_table = device_b_table[device_b_table['ASPECT'] != 'Generic']
                    device_b_table = device_b_table.sort_values(by='REVIEW_COUNT', ascending=False)
                    styled_df_b = device_b_table.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                    data_filtered = device_b_table[(device_b_table["ASPECT"] != device_b) | (device_b_table["ASPECT"] != device_a) & (device_b_table["ASPECT"] != 'Generic')]
                    top_four_aspects = data_filtered.head(4)
                    d = device_b_table.to_dict(orient='records')
                    st.dataframe(styled_df_b)
                    second_table = styled_df_b.to_html(index = False)
                    full_response += second_table
                    history = check_history_length(history,second_table)
                try:
                    user_question_1 = user_question_1.replace("Compare", "Summarize reviews of")
                except:
                    pass
                comparision_summary = query_detailed_compare(user_question + "Which have the following sentiment data" + str(c)+str(d))
                st.write(comparision_summary)
                full_response += comparision_summary
                history = check_history_length(history,comparision_summary)
            except:
                st.write(f"Unable to fetch relevant details based on the provided input. Kindly refine your search query and try again!")


        elif (device_a != None and device_b == None) | (device_a == None and device_b == None):
        
            try:

                data = query_quant(user_question_1,[])
                print(data)
                try:
                    total_reviews = data.loc[data.iloc[:, 0] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
                except:
                    pass
                # total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
                try:
                    data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
                except:
                    pass
                dataframe_as_dict = data.to_dict(orient='records')

                # classify_function = classify(user_question_1+str(dataframe_as_dict))
                if classification == "Summarization":
                    classify_function = "1"
                else:
                    classify_function = "2"

                if classify_function == "1":
                    data_new = data
                    data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
                    data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
                    data_new = make_desired_df(data_new)
                    styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                    data_filtered = data_new[(data_new['ASPECT'] != 'TOTAL') & (data_new['ASPECT'] != 'Generic')]
                    top_four_aspects = data_filtered.head(4)
                    dataframe_as_dict = data_new.to_dict(orient='records')
                    aspects_list = top_four_aspects['ASPECT'].to_list()
            #         formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
                    key_df = get_final_df(aspects_list, device)
                    b =  key_df.to_dict(orient='records')
                    summary_ans = query_aspect_wise_detailed_summary(user_question+"which have the following sentiment :" + str(dataframe_as_dict) + "these are the imporatnt aspect based on aspect ranking : " + str(aspects_list) + "and their respective keywords" + str(b))
                    st.write(summary_ans)
                    full_response += summary_ans
                    history = check_history_length(history,summary_ans)
                    st.dataframe(styled_df)
                    styled_df_html = styled_df.to_html(index=False)
                    full_response += styled_df_html  # Initialize full_response with the HTML table
                    history = check_history_length(history,styled_df_html)
                                
                elif classify_function == "2":
                    data= quantifiable_data(user_question_1)
                    if len(data)>0:
                        numerical_cols = data.select_dtypes(include='number').columns

        # Round float values in numerical columns to one decimal place
                        data[numerical_cols] = data[numerical_cols].apply(lambda x: x.round(1) if x.dtype == 'float' else x)
                        st.dataframe(data)
                        data_1 = data.to_html(index = False)
                        full_response += data_1
                        history = check_history_length(history,data_1)
                        if 'NET_SENTIMENT' in data.columns:
                            # st.write(f" Overall Net Sentiment is {overall_net_sentiment} for {overall_review_count} reviews *")
                            data['Impact']=np.where(data['NET_SENTIMENT']<overall_net_sentiment,'Driving Overall Net Sentiment LOW','Driving Overall Net Sentiment HIGH')

                        dataframe_as_dict = data.to_dict(orient='records')

                        try:
                            data = data.dropna()
                        except:
                            pass
                        try:
                            user_question = user_question.replace("What is the", "Summarize reviews of")
                        except:
                            pass
                        qunat_summary = query_detailed_summary(str(dataframe_as_dict),user_question + "Which have the following sentiment data : " + str(dataframe_as_dict),[])
                        st.write(qunat_summary)
                        full_response += qunat_summary
                        history = check_history_length(history,qunat_summary)
                        if(len(data))>1:
                            generate_chart(data)
                    else:
                        st.write(f"Unable to fetch relevant details based on the provided input. Kindly refine your search query and try again!")
            except Exception as e:
                st.write(f"Unable to fetch relevant details based on the provided input. Kindly refine your search query and try again!")
                print(e)    
        else:
            print('No Flow')
            
global full_response
if __name__ == "__main__":
    global full_response
    if st.sidebar.subheader("Select an option"):
        options = ["Copilot", "Devices"]
        selected_options = st.sidebar.selectbox("Select product", options)
        if selected_options == "Copilot":
            # st.session_state['messages'] = []
            # st.session_state['chat_initiated'] = False
            st.header("Copilot Review Synthesis Tool")
            if "messages" not in st.session_state:
                st.session_state['messages'] = []
            if "chat_initiated" not in st.session_state:
                st.session_state['chat_initiated'] = False
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and "is_html" in message and message["is_html"]:
                        st.markdown(message["content"], unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
            if user_question := st.chat_input("Enter the Prompt: "):
                st.chat_message("user").markdown(user_question)
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("assistant"):
                    full_response = ""
                    try:
                        user_question = user_question.replace("Give me", "What is").replace("Give", "What is")
                    except:
                        pass
                    classification = classify_prompts(user_question)
                    print(classification)
                    if classification != 'Generic':
                        user_question_1 = rephrase_prompt(str(user_question))
                        print(type(user_question_1))
                        print(user_question_1)
                        user_ques(str(user_question_1), user_question, classification)
                    else:
                        user_question_1 = user_question
                        Gen_Ans = query_detailed_generic(user_question_1)
                        st.write(Gen_Ans)
                        history = check_history_length(history,Gen_Ans)
                        full_response += Gen_Ans
                        
                    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_html": True})
                st.session_state['chat_initiated'] = True
            if st.session_state['chat_initiated'] and st.button("New Chat"):
                st.session_state['messages'] = []
                st.session_state['chat_initiated'] = False
                st.experimental_rerun()
        
        elif selected_options == "Devices":
            st.header("Devices Review Synthesis Tool")
            if "messages" not in st.session_state:
                st.session_state['messages'] = []
            if "chat_initiated" not in st.session_state:
                st.session_state['chat_initiated'] = False
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and "is_html" in message and message["is_html"]:
                        st.markdown(message["content"], unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
            if user_question := st.chat_input("Enter the Prompt: "):
                st.chat_message("user").markdown(user_question)
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("assistant"):
                    full_response = ""
                    classification = identify_prompt(user_question)
                    if classification == 'summarization':
                        device = identify_devices(user_question)
                        if device == "Device not available":
                            Gen_Ans = query_devices_detailed_generic(user_question)
                            st.write(Gen_Ans)
                            full_response += Gen_Ans
                        else:
                            device1 = get_sentiment_device_name(device)
                            if not device1:
                                st.write(f"Device {device} not present in the data. Please try again!")
                            else:
                                device_summarization(device1)
                    elif classification == 'comparison':
#                         st.write(f"Flow : {classification}")
                        devices = extract_comparison_devices(user_question)
#                         st.write(f"Devices from User Input: {devices}")
                        if len(devices) == 2:
                            device1 = identify_devices(devices[0])
                            device2 = identify_devices(devices[1])

                            if device1 == "Device not available":
                                st.write(f"Device {devices[0]} not present in the data. Please try again!")
                            elif device2 == "Device not available":
                                st.write(f"Device {devices[1]} not present in the data. Please try again!")
                            else:
                                #st.write(f"Devices Identified as {device1} and {device2}")
                                device1 = get_sentiment_device_name(device1)
                                device2 = get_sentiment_device_name(device2)
                                if not device1:
                                    st.write(f"Device {devices[0]} not present in the data. Please try again!")
                                elif not device2:
                                    st.write(f"Device {devices[1]} not present in the data. Please try again!")
                                else:
                                    #st.write(f"Detailed name for the devices: {device1} and {device2}")
                                    comparison_view(device1,device2)
                    else:
                        Gen_Ans = query_devices_detailed_generic(user_question)
                        st.write(Gen_Ans)
                        full_response += Gen_Ans
                        
                    st.session_state.messages.append({"role": "assistant", "content": full_response, "is_html": True})
                st.session_state['chat_initiated'] = True
            if st.session_state['chat_initiated'] and st.button("New Chat"):
                st.session_state['messages'] = []
                st.session_state['chat_initiated'] = False
                st.experimental_rerun()
