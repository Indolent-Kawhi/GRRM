seqrec_prompt = []

SYN_SYSTEM_PROMPT = """ You are an AI recommendation model. You will receive two inputs:  
- The user's purchase history, which includes a list of previously purchased items along with their metadata.  
- The metadata of the actual next item the user purchased.  

Your task is to analyze the purchase history and recommend the next item the user would purchase, but you must act as if you do not know the actual next item. Base your entire reasoning and recommendation solely on the purchase history—do not use any information from the actual next item in your analysis.  

Your output must be structured into four sequential steps:  
- **Step 1: Analyze the purchase history:** Examine the items in the purchase history, including their metadata, to identify patterns, trends, or any other insights.  
- **Step 2: Build user preferences:** Based on the analysis, infer and summarize the user's preferences.  
- **Step 3: Uncover purchase intent:** Deduce the user's potential purchase intent or next likely item by connecting the history and preferences.  
- **Step 4: Recommend the item:** Provide a clear recommendation for the next item the user would purchase, including its metadata. This should be presented as a prediction derived from your analysis, not from the actual next item.  

Ensure your response is concise, logical, and strictly based on the purchase history. Output only the structured steps as described, without additional commentary."""

SFT_SYSTEM_PROMPT = """You are an AI recommendation model. You task is to analyze the user's purchase history to recommend what they would likely purchase next.

Your output must be structured into following format:
<think>
- **Step 1: Analyze the purchase history:** Examine the items in the purchase history, including their metadata, to identify patterns, trends, or any other insights.  
- **Step 2: Build user preferences:** Based on the analysis, infer and summarize the user's preferences.  
- **Step 3: Uncover purchase intent:** Deduce the user's potential purchase intent or next likely item by connecting the history and preferences.  
</think>
Then rewrite the user purchase history and recommend the next item the user may buy based on your analysis. Enclose the item in <answer> </answer> xml tags."""

#####——0
prompt = "The user has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"

seqrec_prompt.append(prompt)

#####——1
prompt = "I find the user's historical interactive items: {inters}, and I want to know what next item the user needs. Can you help me decide?"

seqrec_prompt.append(prompt)

#####——2
prompt = "Here are the user's historical interactions: {inters}, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order."

seqrec_prompt.append(prompt)

#####——3
prompt = "Based on the items that the user has interacted with: {inters}, can you determine what item would be recommended to him next?"

seqrec_prompt.append(prompt)

#####——4
prompt = "The user has interacted with the following items in order: {inters}. What else do you think the user need?"

seqrec_prompt.append(prompt)

#####——5
prompt = "Here is the item interaction history of the user: {inters}, what to recommend to the user next?"

seqrec_prompt.append(prompt)

#####——6
prompt = "Which item would the user be likely to interact with next after interacting with items {inters}?"

seqrec_prompt.append(prompt)

#####——7
prompt = "By analyzing the user's historical interactions with items {inters}, what is the next expected interaction item?"

seqrec_prompt.append(prompt)

#####——8
prompt = "After interacting with items {inters}, what is the next item that could be recommended for the user?"

seqrec_prompt.append(prompt)

#####——9
prompt = "Given the user's historical interactive items arranged in chronological order: {inters}, can you recommend a suitable item for the user?"

seqrec_prompt.append(prompt)

#####——10
prompt = "Considering the user has interacted with items {inters}. What is the next recommendation for the user?"

seqrec_prompt.append(prompt)

#####——11
prompt = "What is the top recommended item for the user who has previously interacted with items {inters} in order?"

seqrec_prompt.append(prompt)

#####——12
prompt = "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user most desires based on the given interaction records."

seqrec_prompt.append(prompt)

#####——13
prompt = "Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. The historical interactions are provided as follows: {inters}."

seqrec_prompt.append(prompt)

#####——14
prompt = "You can access the user's historical item interaction records: {inters}. Now your task is to recommend the next potential item to him, considering his past interactions."

seqrec_prompt.append(prompt)

#####——15
prompt = "You have observed that the user has interacted with the following items: {inters}, please recommend a next item that you think would be suitable for the user."

seqrec_prompt.append(prompt)

#####——16
prompt = "You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. Using this history as a reference, please select the next item to recommend to the user."

seqrec_prompt.append(prompt)